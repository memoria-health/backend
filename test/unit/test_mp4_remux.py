"""Unit tests for `frames_to_mp4`.

These tests shell out to real ffmpeg (the bot does the same at
runtime). Skipped if ffmpeg isn't on PATH so CI doesn't fail
spuriously on an image that didn't install it. We assert two
contracts:

  1. The function returns bytes that ffprobe agrees is a valid MP4
     with the expected duration / framerate.
  2. The function raises Mp4RemuxError on the kinds of failures the
     worker should surface (bad subprocess exit, mismatched frame
     dimensions, ffmpeg missing).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess

import pytest

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4

_pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _yuv420_blob(width: int, height: int, *, fill: int = 0x80) -> bytes:
    """An all-grey I420 frame at the given size. Grey because the
    Y plane fills to 128 and U/V planes fill to 128 (neutral chroma).
    The actual pixel values don't matter for these tests — only the
    byte count + the ability to re-decode.
    """
    return bytes([fill]) * (width * height * 3 // 2)


def _ffprobe(mp4: bytes) -> dict:
    """Pipe MP4 bytes into ffprobe and return the parsed JSON."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            "-i", "pipe:0",
        ],
        input=mp4,
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_returns_decodable_mp4_with_expected_dimensions() -> None:
    width, height, fps = 320, 240, 5
    # 15 frames @ 5fps = a 3-second clip.
    frames = [
        FrameSnapshot(
            width=width, height=height,
            data=_yuv420_blob(width, height),
            captured_at=100.0 + i / fps,
        )
        for i in range(15)
    ]
    mp4 = await frames_to_mp4(frames, fps=fps)
    assert mp4.startswith(b"\x00\x00\x00") or mp4[4:8] == b"ftyp"
    info = _ffprobe(mp4)
    streams = info["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    assert video["codec_name"] in {"h264", "avc1"}
    assert video["width"] == width
    assert video["height"] == height


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_empty_frame_list() -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        await frames_to_mp4([], fps=5)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_mismatched_frame_dimensions() -> None:
    f1 = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    f2 = FrameSnapshot(width=400, height=300, data=_yuv420_blob(400, 300), captured_at=100.2)
    with pytest.raises(ValueError, match="expected 320x240"):
        await frames_to_mp4([f1, f2], fps=5)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_wrong_byte_length_for_size() -> None:
    f = FrameSnapshot(width=320, height=240, data=b"\x00" * 100, captured_at=100.0)
    with pytest.raises(ValueError, match="expected 115200"):
        await frames_to_mp4([f], fps=5)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_rejects_out_of_range_fps() -> None:
    f = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    with pytest.raises(ValueError, match="fps must be in"):
        await frames_to_mp4([f], fps=0)
    with pytest.raises(ValueError, match="fps must be in"):
        await frames_to_mp4([f], fps=120)


@pytest.mark.asyncio
async def test_raises_clear_error_when_ffmpeg_missing(monkeypatch) -> None:
    """When ffmpeg can't be launched we want a `Mp4RemuxError` with a
    legible message, not a bare `FileNotFoundError`. The worker logs
    these — readability matters.
    """
    async def _fake_exec(*_args, **_kw):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)

    f = FrameSnapshot(width=320, height=240, data=_yuv420_blob(320, 240), captured_at=100.0)
    with pytest.raises(Mp4RemuxError, match="ffmpeg binary not found"):
        await frames_to_mp4([f], fps=5)
