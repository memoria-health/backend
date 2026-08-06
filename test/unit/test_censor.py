"""Unit tests for `src.services.detection.censor`.

Like test_mp4_remux.py these shell out to real ffmpeg/ffprobe (the
censor does at runtime) and are skipped when the binaries are missing.
No pod is contacted — `pod_masks.fetch_window` and `_person_mask` are
monkeypatched, so CI needs neither network nor a GPU host.

Contracts under test:

  - AUDIO IS KEPT (2026-07-28 product call): a fixture clip WITH an
    audio track comes out of both censor tiers with its audio stream
    intact (ffprobe-verified) — visuals censored, sound stays.
  - No-mask frames of a sensitive clip are pixelated FULL-FRAME
    (`_censor_frame` with mask=None), and masked frames keep pixels
    outside the dilated mask untouched.
  - An unreachable pod (`PodMaskError`) falls back to tier 2 rather
    than shipping anything unmasked — the fail-closed rule survived the
    2026-08-06 switch from local YOLO to pod masks.
  - Tier 1 runs in an ISOLATED subprocess under a hard timeout
    (2026-07-28 native-torch-hang hardening): success uses the child's
    output; timeout kills the child; timeout/nonzero-exit both fall
    back to the pure-ffmpeg full-frame tier in-process. Tested with
    scripted stand-in children via the `_person_mask_subprocess_cmd`
    seam — real spawn/kill mechanics.
  - The pod TOKEN never reaches the child's argv (it would be visible
    in `ps`); it rides the environment.
  - When even tier 2 can't work (undecodable input), `CensorError`
    propagates so the caller drops the clip.
"""

from __future__ import annotations

import asyncio as aio
import json
import os
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src.services.detection import censor
from src.services.detection.censor import CensorError, censor_clip
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.mp4_remux import frames_to_mp4

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _i420_blob(width: int, height: int, *, fill: int = 0x80) -> bytes:
    return bytes([fill]) * (width * height * 3 // 2)


async def _fixture_clip_with_audio() -> bytes:
    """A tiny real MP4 with BOTH a video and an audio stream — the
    thing the censor must strip."""
    frames = [
        FrameSnapshot(
            width=320, height=240,
            data=_i420_blob(320, 240),
            captured_at=100.0 + i / 4,
        )
        for i in range(8)  # 2s @ 4fps
    ]
    pcm = b"\x00\x08" * 32000  # 2s int16 mono @ 16 kHz
    return await frames_to_mp4(frames, fps=4, pcm=pcm, pcm_sample_rate=16000)


def _probe_streams(path: str) -> list[dict]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-i", path],
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)["streams"]


def _codec_types(path: str) -> set[str]:
    return {s["codec_type"] for s in _probe_streams(path)}


def _count_frames(path: str) -> int:
    """Decoded frame count of the video stream (ffprobe -count_frames)."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames",
         "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames",
         "-print_format", "json", "-i", path],
        capture_output=True,
        check=True,
    )
    return int(json.loads(proc.stdout)["streams"][0]["nb_read_frames"])


async def _write_fixture(tmp_path) -> str:
    mp4 = await _fixture_clip_with_audio()
    in_path = str(tmp_path / "in.mp4")
    with open(in_path, "wb") as f:
        f.write(mp4)
    # Sanity: the fixture really does carry audio, or the audio-kept
    # assertions below are vacuous.
    assert _codec_types(in_path) == {"video", "audio"}
    return in_path


# ── full-frame tier (tier 2) ─────────────────────────────────────────


async def test_full_frame_censor_keeps_audio(tmp_path) -> None:
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await _censor(in_path, out_path, full_frame=True)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_full_frame_censor_raises_on_undecodable_input(tmp_path) -> None:
    """Tier 2 dying is the drop-the-clip signal for the caller."""
    in_path = str(tmp_path / "garbage.mp4")
    with open(in_path, "wb") as f:
        f.write(b"this is not an mp4")
    with pytest.raises(CensorError):
        await _censor(in_path, str(tmp_path / "out.mp4"), full_frame=True)


# ── person-mask tier (tier 1) — sync pipeline, pod seams mocked ─────

CAMERA_ID = "cafe1234-0000-4000-8000-000000000000"
CLIP_START = 1_786_000_000.0


def _run_person_mask_inprocess(monkeypatch) -> None:
    """Route censor_clip's tier 1 back in-process (thread, not
    subprocess) so the mask-seam monkeypatches apply. The subprocess
    mechanics themselves are covered by the subprocess-seam tests
    below; these tests pin the PIPELINE contract (frame treatment +
    audio kept)."""

    async def _inprocess(
        in_path: str, out_path: str, *, mode: str = "pixelate",
        camera_id: str = "", clip_start: float = 0.0,
    ) -> None:
        await aio.to_thread(
            censor._censor_person_mask_sync, in_path, out_path, mode=mode,
            camera_id=camera_id, clip_start=clip_start,
            pod_url="http://pod.invalid", pod_token="",
        )

    monkeypatch.setattr(censor, "_censor_person_mask_subprocess", _inprocess)


def _stub_window(monkeypatch, mask_fn) -> None:
    """Replace the pod fetch with an empty window, and `_person_mask`
    with `mask_fn(shape)`. Keeps the real loop, indexing and encoder."""
    monkeypatch.setattr(
        censor.pod_masks, "fetch_window",
        lambda *a, **kw: censor.pod_masks.MaskWindow([]),
    )
    monkeypatch.setattr(
        censor, "_person_mask",
        lambda window, frame_time, shape: mask_fn(shape),
    )


async def _censor(in_path: str, out_path: str, **kw) -> None:
    await censor_clip(
        in_path, out_path, camera_id=CAMERA_ID, clip_start=CLIP_START, **kw
    )


async def test_person_mask_tier_no_person_pixelates_and_keeps_audio(
    tmp_path, monkeypatch,
) -> None:
    """Sensitive clip, the pod retained no mask for any frame → every
    frame is censored full-frame (never left raw); audio muxed back."""
    _run_person_mask_inprocess(monkeypatch)
    _stub_window(monkeypatch, lambda shape: None)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await _censor(in_path, out_path)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_person_mask_tier_with_mask_keeps_audio(
    tmp_path, monkeypatch,
) -> None:
    def _mask(shape):
        h, w = shape
        m = np.zeros((h, w), dtype=np.uint8)
        m[h // 4: h // 2, w // 4: w // 2] = 1
        return m

    _run_person_mask_inprocess(monkeypatch)
    _stub_window(monkeypatch, _mask)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await _censor(in_path, out_path)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_person_mask_streaming_preserves_frame_count(
    tmp_path, monkeypatch,
) -> None:
    """The streaming tier-1 pipeline (frame → mask → censor → piped
    encoder, O(1-frame) memory) still yields a VALID MP4 with exactly
    the input's frame count — no frame dropped or duplicated at the
    open/close boundaries of the incremental encoder."""
    _run_person_mask_inprocess(monkeypatch)
    _stub_window(monkeypatch, lambda shape: None)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await _censor(in_path, out_path)
    n_in = _count_frames(in_path)
    assert n_in == 8  # fixture sanity
    assert _count_frames(out_path) == n_in


async def test_frame_time_advances_with_the_output_grid(
    tmp_path, monkeypatch,
) -> None:
    """Each frame is looked up at clip_start + i/clip_fps.

    This is the whole basis for indexing pod masks by capture time: the
    VFR encode materializes the window onto a uniform fps grid whose
    playback duration equals wall clock. If the censor ever stopped
    advancing (or advanced at the wrong rate) every frame would silently
    get the first frame's mask.
    """
    seen: list[float] = []

    def _record(window, frame_time, shape):
        seen.append(frame_time)
        return None

    _run_person_mask_inprocess(monkeypatch)
    monkeypatch.setattr(
        censor.pod_masks, "fetch_window",
        lambda *a, **kw: censor.pod_masks.MaskWindow([]),
    )
    monkeypatch.setattr(censor, "_person_mask", _record)
    in_path = await _write_fixture(tmp_path)
    await _censor(in_path, str(tmp_path / "out.mp4"))
    fps = float(censor.get_settings().clip_fps)
    assert seen == [CLIP_START + i / fps for i in range(len(seen))]
    assert len(seen) == 8


async def test_pod_unreachable_falls_back_to_full_frame(
    tmp_path, monkeypatch,
) -> None:
    """Pod down / tunnel dead → tier 1 raises → tier 2 (pure ffmpeg)
    still produces a censored clip. Nothing unmasked ever ships."""
    def _boom(*a, **kw):
        raise censor.pod_masks.PodMaskError("tunnel is dead")

    _run_person_mask_inprocess(monkeypatch)
    monkeypatch.setattr(censor.pod_masks, "fetch_window", _boom)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await _censor(in_path, out_path)  # must not raise
    assert _codec_types(out_path) == {"video", "audio"}


def test_the_pod_token_never_reaches_argv() -> None:
    """argv is world-readable via `ps`, so the credential must travel in
    the child's environment instead. Belt-and-braces against someone
    later 'simplifying' the env plumbing into a flag."""
    argv = censor._person_mask_subprocess_cmd(
        "in.mp4", "out.mp4", camera_id=CAMERA_ID, clip_start=CLIP_START,
        pod_url="http://pod.invalid",
    )
    secret = "s3cret-token-value"
    assert secret not in " ".join(argv)
    env = censor._censor_child_env(secret)
    assert env["NEMARA_PERCEPTION_TOKEN"] == secret
    # And an empty token UNSETS rather than blanks it, so "no token"
    # means the same thing on both sides of the wire.
    assert "NEMARA_PERCEPTION_TOKEN" not in censor._censor_child_env("")


def test_clip_start_survives_the_argv_round_trip() -> None:
    """Epoch seconds carry ~9 significant digits before the point. A
    float formatted with '%f'-style rounding would land the window whole
    milliseconds off; `repr` round-trips exactly."""
    argv = censor._person_mask_subprocess_cmd(
        "in.mp4", "out.mp4", camera_id=CAMERA_ID,
        clip_start=1786012345.6789012, pod_url="http://pod.invalid",
    )
    parsed = float(argv[argv.index("--clip-start") + 1])
    assert parsed == 1786012345.6789012


# ── tier-1 subprocess isolation (2026-07-28 hardening) ───────────────
#
# Real child processes via the `_person_mask_subprocess_cmd` seam —
# scripted stand-ins (exit codes, sleeps, or the real CLI in
# --full-frame mode) so no pod is contacted, while the
# spawn / wait_for / kill machinery under test is the production code.


def _fake_cmd(monkeypatch, *argv_tail: str) -> None:
    monkeypatch.setattr(
        censor,
        "_person_mask_subprocess_cmd",
        lambda in_path, out_path, **kw: [sys.executable, *argv_tail],
    )


async def test_subprocess_success_end_to_end(tmp_path, monkeypatch) -> None:
    """censor_clip tier 1 via a REAL child running the REAL module CLI
    (--full-frame stand-in for the mask pass, so no pod is contacted):
    child exits 0, its output is used, audio is stripped. Also proves
    `python -m src.services.detection.censor` imports headless."""
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    monkeypatch.setattr(
        censor,
        "_person_mask_subprocess_cmd",
        lambda i, o, **kw: [
            sys.executable, "-m", "src.services.detection.censor",
            i, o, "--full-frame",
        ],
    )
    await _censor(in_path, out_path)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_subprocess_timeout_kills_child_and_falls_back(
    tmp_path, monkeypatch,
) -> None:
    """A wedged child (the native-torch-hang scenario) is killed at the
    deadline and tier 2 still produces a censored, audio-free clip."""
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    pid_file = str(tmp_path / "child.pid")
    # Child records its pid then hangs far beyond the deadline.
    _fake_cmd(
        monkeypatch,
        "-c",
        "import os, sys, time\n"
        f"open({pid_file!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)",
    )
    monkeypatch.setattr(
        censor,
        "get_settings",
        lambda: SimpleNamespace(
            censor_timeout_seconds=1.5,
            perception_mask_url="http://pod.invalid",
            perception_mask_token="",
        ),
    )
    start = time.monotonic()
    await _censor(in_path, out_path)  # must not raise, must not hang
    elapsed = time.monotonic() - start
    assert elapsed < 30  # deadline tripped; we never sat out the sleep(60)
    # The child was killed AND reaped (proc.wait() ran before fallback).
    pid = int(open(pid_file).read())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_subprocess_nonzero_exit_falls_back(tmp_path, monkeypatch) -> None:
    """Child crash (import error, decode error, ... → exit 1..N) → tier
    2 fallback, same contract as the old in-process tier-1 exception."""
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    _fake_cmd(monkeypatch, "-c", "import sys; sys.exit(3)")
    await _censor(in_path, out_path)  # must not raise
    assert _codec_types(out_path) == {"video", "audio"}


async def test_subprocess_failure_and_tier2_failure_raises(
    tmp_path, monkeypatch,
) -> None:
    """Both tiers dead (child exits nonzero, input undecodable for
    ffmpeg) → CensorError propagates → caller drops the clip."""
    in_path = str(tmp_path / "garbage.mp4")
    with open(in_path, "wb") as f:
        f.write(b"this is not an mp4")
    _fake_cmd(monkeypatch, "-c", "import sys; sys.exit(2)")
    with pytest.raises(CensorError):
        await _censor(in_path, str(tmp_path / "out.mp4"))


def test_cli_nonzero_exit_on_bad_input(tmp_path) -> None:
    """The CLI child itself: undecodable input → nonzero exit (the
    signal the parent maps onto tier-2 fallback). --full-frame so the
    check needs no YOLO weights."""
    in_path = str(tmp_path / "garbage.mp4")
    with open(in_path, "wb") as f:
        f.write(b"nope")
    proc = subprocess.run(
        [
            sys.executable, "-m", "src.services.detection.censor",
            in_path, str(tmp_path / "out.mp4"), "--full-frame",
        ],
        capture_output=True,
    )
    assert proc.returncode != 0


# ── _censor_frame (pure) ─────────────────────────────────────────────


def _detail_frame(h: int = 120, w: int = 160) -> np.ndarray:
    """A high-detail frame (random noise) so pixelation measurably
    changes it."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def test_censor_frame_no_mask_pixelates_everything() -> None:
    frame = _detail_frame()
    out = censor._censor_frame(frame, None)
    assert out.shape == frame.shape
    # Whole frame censored: noise cannot survive 24x pixelation.
    assert np.abs(out.astype(int) - frame.astype(int)).mean() > 5.0


def test_censor_frame_masked_region_censored_rest_untouched() -> None:
    frame = _detail_frame()
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:20, 0:20] = 1  # person in the top-left corner
    out = censor._censor_frame(frame, mask)
    # Inside the mask: censored (changed).
    assert not np.array_equal(out[0:20, 0:20], frame[0:20, 0:20])
    # Far outside the dilated mask (dilation kernel is ≤ ~15-30 px at
    # this size): untouched.
    assert np.array_equal(out[h - 20:, w - 20:], frame[h - 20:, w - 20:])


def test_censor_frame_empty_mask_is_full_frame() -> None:
    """An all-zero mask means 'no person' — same treatment as None."""
    frame = _detail_frame()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    out = censor._censor_frame(frame, mask)
    assert np.array_equal(out, censor._censor_frame(frame, None))


class TestScratchSweep:
    """Startup sweep of the clip scratch dir (SIGKILL-mid-censor
    residue — see clip_archival.sweep_scratch_dir)."""

    def test_sweep_removes_stranded_files_and_counts(self, tmp_path, monkeypatch):
        from src.services import clip_archival

        monkeypatch.setattr(
            clip_archival.tempfile, "gettempdir", lambda: str(tmp_path)
        )
        scratch = clip_archival._scratch_dir()
        for name in ("stranded_raw.mp4", "stranded_censored.mp4", "list.txt"):
            with open(f"{scratch}/{name}", "wb") as f:
                f.write(b"raw bytes")
        assert clip_archival.sweep_scratch_dir() == 3
        import os as _os

        assert _os.listdir(scratch) == []
        # Idempotent + empty-dir safe.
        assert clip_archival.sweep_scratch_dir() == 0

    def test_tempfiles_are_created_inside_scratch_dir(self, tmp_path, monkeypatch):
        import asyncio

        from src.services import clip_archival
        from src.services.detection import censor as censor_mod

        monkeypatch.setattr(
            clip_archival.tempfile, "gettempdir", lambda: str(tmp_path)
        )
        seen: dict[str, str] = {}

        async def fake_censor(
            in_path, out_path, *, full_frame, keep_audio=True,
            mode="pixelate", camera_id="", clip_start=0.0,
        ):
            seen["in"], seen["out"] = in_path, out_path
            with open(out_path, "wb") as f:
                f.write(b"censored")

        monkeypatch.setattr(censor_mod, "censor_clip", fake_censor)
        out = asyncio.run(clip_archival._censor_bytes(
            b"raw", full_frame=True, camera_id=CAMERA_ID,
            clip_start=CLIP_START,
        ))
        assert out == b"censored"
        scratch = clip_archival._scratch_dir()
        assert seen["in"].startswith(scratch)
        assert seen["out"].startswith(scratch)
        import os as _os

        assert _os.listdir(scratch) == []  # unlinked in the finally
