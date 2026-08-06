"""Inline masking encoder — raw frames in, masked MP4 out.

The point of doing this inline rather than over the finished clip: the
recorder already holds decoded frames, so there is no decode and no
re-encode, and raw footage never lands on disk.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.services.data_collection.streaming_encoder import (
    MaskingMp4Writer,
    StreamingMp4Writer,
)
from src.services.detection import censor

CAMERA_ID = "cafe1234-0000-4000-8000-000000000000"
POD_URL = "http://pod.invalid"


def _writer(path, **kw):
    """The masking writer needs a camera and a pod URL since 2026-08-06:
    its child asks the perception pod for that camera's masks."""
    return MaskingMp4Writer(
        path, camera_id=CAMERA_ID, pod_url=POD_URL, **kw
    )


class _FakeFeed:
    """Stand-in for `pod_masks.MaskFeed` — no HTTP, no thread."""

    def __init__(self, mask=None):
        self._mask = mask

    def freshest(self, shape):
        return self._mask

    def close(self):
        pass

_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)

_W, _H, _FPS = 64, 48, 15


def test_masking_writer_routes_frames_through_the_censor(tmp_path) -> None:
    cmd = _writer(tmp_path / "v.mp4", width=_W, height=_H, fps=_FPS)._build_command()
    assert cmd[0] == sys.executable
    assert "src.services.detection.censor" in cmd
    assert f"{_W}x{_H}@{_FPS}" in cmd


def test_plain_writer_still_goes_straight_to_ffmpeg(tmp_path) -> None:
    """The raw path must be untouched by the subclass existing."""
    cmd = StreamingMp4Writer(tmp_path / "v.mp4")._build_command()
    assert cmd[0] == "ffmpeg"
    assert "libx264" in cmd


def test_masking_inherits_the_timeline(tmp_path) -> None:
    """It must report the same clock-truthful timeline as the plain
    writer, or annotations would not line up on masked clips."""
    w = _writer(tmp_path / "v.mp4", width=_W, height=_H, fps=_FPS)
    assert w.first_frame_mono is None
    assert w.duration_seconds == 0.0
    assert hasattr(w, "deficit_frames")


@_ffmpeg
def test_stream_masking_produces_a_playable_mp4(tmp_path, monkeypatch) -> None:
    """End to end through the real frame encoder, with the pod feed
    stubbed — the mask arithmetic itself is covered by the research-mode
    and pod_masks tests."""
    out = tmp_path / "masked.mp4"

    # A person occupying the middle of the frame.
    mask = np.zeros((_H, _W), dtype=np.uint8)
    mask[12:36, 16:48] = 1
    monkeypatch.setattr(
        censor, "get_settings",
        lambda: type("S", (), {"censor_stream_segment_every": 5})(),
    )

    frames = 30
    payload = bytes(_W * _H * 3 // 2) * frames
    censor.mask_stream_sync(
        _W, _H, _FPS, str(out), stdin=io.BytesIO(payload),
        camera_id=CAMERA_ID, pod_url=POD_URL, feed=_FakeFeed(mask),
    )

    assert out.exists() and out.stat().st_size > 0
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, check=True,
    ).stdout
    assert "video" in probed


def test_no_subject_still_covers_every_frame(tmp_path, monkeypatch) -> None:
    """The pod having no mask must not leak a raw frame. Keep it
    simple: cover the frame and carry on, no clip-dropping special case."""
    out = tmp_path / "masked.mp4"
    monkeypatch.setattr(
        censor, "get_settings",
        lambda: type("S", (), {"censor_stream_segment_every": 5})(),
    )

    written: list = []

    class _Enc:
        def __init__(self, *a, **kw) -> None: ...
        def write(self, frame) -> None: written.append(frame)
        def close(self) -> None: ...
        def abort(self) -> None: ...

    monkeypatch.setattr(censor, "_FrameEncoder", _Enc)
    payload = bytes(_W * _H * 3 // 2) * 5
    censor.mask_stream_sync(
        _W, _H, _FPS, str(out), stdin=io.BytesIO(payload),
        camera_id=CAMERA_ID, pod_url=POD_URL, feed=_FakeFeed(None),
    )

    assert len(written) == 5
    fill = np.array(censor._RESEARCH_FILL_BGR, np.uint8)
    assert all((f == fill).all() for f in written)


async def test_a_failed_child_marks_the_clip_untrustworthy(tmp_path) -> None:
    """Nothing checked the child's exit code before. A child that dies
    (unreachable pod, bad args) would have produced a broken file that
    got uploaded with nothing flagged anywhere."""
    w = _writer(tmp_path / "v.mp4", width=_W, height=_H, fps=_FPS)
    w._build_command = lambda: [sys.executable, "-c", "import sys; sys.exit(3)"]
    await w.start()
    w.feed(_W, _H, bytes(_W * _H * 3 // 2))
    await w.aclose()

    assert w.exit_code == 3
    assert w.ok is False


def test_torn_final_frame_is_not_encoded(tmp_path, monkeypatch) -> None:
    """A half-written frame at EOF must be dropped, not reshaped into
    garbage."""
    out = tmp_path / "masked.mp4"
    mask = np.ones((_H, _W), dtype=np.uint8)
    monkeypatch.setattr(
        censor, "get_settings",
        lambda: type("S", (), {"censor_stream_segment_every": 5})(),
    )

    written: list = []

    class _Enc:
        def __init__(self, *a, **kw) -> None: ...
        def write(self, frame) -> None: written.append(frame)
        def close(self) -> None: ...
        def abort(self) -> None: ...

    monkeypatch.setattr(censor, "_FrameEncoder", _Enc)
    frame_bytes = _W * _H * 3 // 2
    payload = bytes(frame_bytes) * 2 + bytes(frame_bytes // 3)  # torn tail
    censor.mask_stream_sync(
        _W, _H, _FPS, str(out), stdin=io.BytesIO(payload),
        camera_id=CAMERA_ID, pod_url=POD_URL, feed=_FakeFeed(mask),
    )
    assert len(written) == 2
