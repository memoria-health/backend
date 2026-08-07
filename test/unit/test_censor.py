"""Unit tests for `src.services.detection.censor`.

Like test_mp4_remux.py these shell out to real ffmpeg/ffprobe (the
censor does at runtime) and are skipped when the binaries are missing.
The YOLO seg model is NEVER loaded here — the seg seams
(`_get_seg_model` / `_person_mask`) are monkeypatched so no weights
download in CI.

Contracts under test:

  - AUDIO IS KEPT (2026-07-28 product call): a fixture clip WITH an
    audio track comes out of both censor tiers with its audio stream
    intact (ffprobe-verified) — visuals censored, sound stays.
  - No-person frames of a sensitive clip are pixelated FULL-FRAME
    (`_censor_frame` with mask=None), and masked frames keep pixels
    outside the dilated mask untouched.
  - Tier 1 runs in an ISOLATED subprocess under a hard timeout
    (2026-07-28 native-torch-hang hardening): success uses the child's
    output; timeout kills the child; timeout/nonzero-exit both fall
    back to the pure-ffmpeg full-frame tier in-process. Tested with
    scripted stand-in children via the `_person_mask_subprocess_cmd`
    seam — real spawn/kill mechanics, no YOLO weights ever loaded.
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
    await censor_clip(in_path, out_path, full_frame=True)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_full_frame_censor_raises_on_undecodable_input(tmp_path) -> None:
    """Tier 2 dying is the drop-the-clip signal for the caller."""
    in_path = str(tmp_path / "garbage.mp4")
    with open(in_path, "wb") as f:
        f.write(b"this is not an mp4")
    with pytest.raises(CensorError):
        await censor_clip(in_path, str(tmp_path / "out.mp4"), full_frame=True)


# ── person-mask tier (tier 1) — sync pipeline, seg seams mocked ─────


def _run_person_mask_inprocess(monkeypatch) -> None:
    """Route censor_clip's tier 1 back in-process (thread, not
    subprocess) so the seg-seam monkeypatches apply. The subprocess
    mechanics themselves are covered by the subprocess-seam tests
    below; these tests pin the PIPELINE contract (frame treatment +
    audio kept)."""

    async def _inprocess(in_path: str, out_path: str) -> None:
        await aio.to_thread(censor._censor_person_mask_sync, in_path, out_path)

    monkeypatch.setattr(censor, "_censor_person_mask_subprocess", _inprocess)


async def test_person_mask_tier_no_person_pixelates_and_keeps_audio(
    tmp_path, monkeypatch,
) -> None:
    """Sensitive clip, seg finds nobody in any frame → every frame is
    censored full-frame (never left raw); source audio muxed back."""
    _run_person_mask_inprocess(monkeypatch)
    monkeypatch.setattr(censor, "_get_seg_model", lambda: object())
    monkeypatch.setattr(censor, "_person_mask", lambda model, frame: None)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await censor_clip(in_path, out_path)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_person_mask_tier_with_mask_keeps_audio(
    tmp_path, monkeypatch,
) -> None:
    def _mask(model, frame):
        h, w = frame.shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        m[h // 4: h // 2, w // 4: w // 2] = 1
        return m

    _run_person_mask_inprocess(monkeypatch)
    monkeypatch.setattr(censor, "_get_seg_model", lambda: object())
    monkeypatch.setattr(censor, "_person_mask", _mask)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await censor_clip(in_path, out_path)
    assert _codec_types(out_path) == {"video", "audio"}


async def test_person_mask_streaming_preserves_frame_count(
    tmp_path, monkeypatch,
) -> None:
    """The streaming tier-1 pipeline (frame → mask → censor → piped
    encoder, O(1-frame) memory) still yields a VALID MP4 with exactly
    the input's frame count — no frame dropped or duplicated at the
    open/close boundaries of the incremental encoder."""
    _run_person_mask_inprocess(monkeypatch)
    monkeypatch.setattr(censor, "_get_seg_model", lambda: object())
    monkeypatch.setattr(censor, "_person_mask", lambda model, frame: None)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await censor_clip(in_path, out_path)
    n_in = _count_frames(in_path)
    assert n_in == 8  # fixture sanity
    assert _count_frames(out_path) == n_in


async def test_seg_stack_failure_falls_back_to_full_frame(
    tmp_path, monkeypatch,
) -> None:
    """ultralytics missing / model load crash (now: inside the child
    process) → tier 1 raises → tier 2 (pure ffmpeg) still produces a
    censored, audio-free clip."""
    def _boom():
        raise ImportError("ultralytics is broken on this host")

    _run_person_mask_inprocess(monkeypatch)
    monkeypatch.setattr(censor, "_get_seg_model", _boom)
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    await censor_clip(in_path, out_path)  # must not raise
    assert _codec_types(out_path) == {"video", "audio"}


# ── tier-1 subprocess isolation (2026-07-28 hardening) ───────────────
#
# Real child processes via the `_person_mask_subprocess_cmd` seam —
# scripted stand-ins (exit codes, sleeps, or the real CLI in
# --full-frame mode) so YOLO weights are never loaded, while the
# spawn / wait_for / kill machinery under test is the production code.


def _fake_cmd(monkeypatch, *argv_tail: str) -> None:
    monkeypatch.setattr(
        censor,
        "_person_mask_subprocess_cmd",
        lambda in_path, out_path, **kw: [sys.executable, *argv_tail],
    )


async def test_subprocess_success_end_to_end(tmp_path, monkeypatch) -> None:
    """censor_clip tier 1 via a REAL child running the REAL module CLI
    (--full-frame stand-in for the YOLO pass, so no weights load):
    child exits 0, its output is used, audio is stripped. Also proves
    `python -m src.services.detection.censor` imports headless."""
    in_path = await _write_fixture(tmp_path)
    out_path = str(tmp_path / "out.mp4")
    monkeypatch.setattr(
        censor,
        "_person_mask_subprocess_cmd",
        lambda i, o: [
            sys.executable, "-m", "src.services.detection.censor",
            i, o, "--full-frame",
        ],
    )
    await censor_clip(in_path, out_path)
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
        lambda: SimpleNamespace(censor_timeout_seconds=1.5),
    )
    start = time.monotonic()
    await censor_clip(in_path, out_path)  # must not raise, must not hang
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
    await censor_clip(in_path, out_path)  # must not raise
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
        await censor_clip(in_path, str(tmp_path / "out.mp4"))


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

        async def fake_censor(in_path, out_path, *, full_frame, keep_audio=True, mode="pixelate"):
            seen["in"], seen["out"] = in_path, out_path
            with open(out_path, "wb") as f:
                f.write(b"censored")

        monkeypatch.setattr(censor_mod, "censor_clip", fake_censor)
        out = asyncio.run(clip_archival._censor_bytes(b"raw", full_frame=True))
        assert out == b"censored"
        scratch = clip_archival._scratch_dir()
        assert seen["in"].startswith(scratch)
        assert seen["out"].startswith(scratch)
        import os as _os

        assert _os.listdir(scratch) == []  # unlinked in the finally


# ── segmentation quality knobs (2026-08-07) ──────────────────────────


def test_person_mask_refines_by_re_segmenting_the_person_crop() -> None:
    """Two passes: find the person, then re-segment their own box.

    Measured against yolo11x-seg@1280, standing clip / recumbent clip:

        one pass,  n-seg @  640   IoU 0.819 / 0.210    25 ms
        one pass,  s-seg @ 1280   IoU 0.944 / 0.596   113 ms
        TWO PASS,  n-seg @  640   IoU 0.903 / 0.392    57 ms

    The big single-pass model was tried and reverted the same day: it
    pushed a ~26s gate clip to ~30s of masking and the operator's page
    stopped loading inside its 60s budget. Two passes get most of the
    quality for half the cost because the gain comes from the subject
    filling the input, not from capacity or pixels.
    """
    calls: list = []

    class _Res:
        def __init__(self, m):
            self.masks = m

    class _Model:
        def predict(self, image, **kw):
            calls.append(image.shape[:2])
            h, w = image.shape[:2]
            arr = np.zeros((1, h, w), dtype=np.float32)
            arr[0, h // 4: h // 2, w // 4: w // 2] = 1.0

            class _M:
                data = _Tensor(arr)
            return [_Res(_M())]

    class _Tensor:
        def __init__(self, a): self._a = a
        def cpu(self): return self
        def numpy(self): return self._a
        def __len__(self): return len(self._a)

    out = censor._person_mask(_Model(), np.zeros((1080, 1920, 3), np.uint8))
    assert out is not None
    # Two passes, and the second saw a SMALLER image than the first.
    assert len(calls) == 2, calls
    assert calls[1] < calls[0], calls


def test_refined_mask_replaces_coarse_inside_the_crop() -> None:
    """Not a union.

    OR-ing the refined mask with the coarse one drags every imprecision
    of the coarse mask back in and cancels the refinement exactly —
    measured 0.821 IoU with the union versus 0.903 with the replace,
    i.e. no better than not refining at all. Caught by measuring the
    shipped function rather than trusting the bench harness.
    """
    import inspect
    src = inspect.getsource(censor._person_mask)
    assert "out = coarse.copy()" in src
    assert "out |= coarse" not in src


def test_person_mask_asks_for_full_resolution_masks(monkeypatch) -> None:
    """`retina_masks` and `imgsz` both reach ultralytics.

    Without retina_masks the silhouette is reconstructed from a ~160px
    prototype and staircases; without imgsz it is derived from a
    640-wide view of a 1080p frame however large the model is."""
    seen: dict = {}

    class _Res:
        masks = None

    class _Model:
        def predict(self, frame, **kw):
            seen.update(kw)
            return [_Res()]

    censor._person_mask(_Model(), np.zeros((1080, 1920, 3), np.uint8))
    assert seen["retina_masks"] is True
    assert seen["imgsz"] == censor._SEG_IMGSZ
    assert seen["conf"] == censor._SEG_CONFIDENCE
    assert seen["classes"] == [censor._PERSON_CLASS_ID]
