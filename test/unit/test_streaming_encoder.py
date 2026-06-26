"""Task 5 — StreamingMp4Writer (CFR pacer + H.264 streaming encoder).

The sync-logic tests (drop accounting, off-size scaling, bad-length
rejection) need no ffmpeg and no real time — they exercise the
non-blocking `feed()` slot directly. The end-to-end test shells out to
real ffmpeg/ffprobe to prove the pacer produces a valid H.264 MP4, and
is skipped when ffmpeg isn't on PATH.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services.data_collection.streaming_encoder import StreamingMp4Writer

_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)


def _yuv420_blob(width: int, height: int, *, fill: int = 0x80) -> bytes:
    return bytes([fill]) * (width * height * 3 // 2)


def _ffprobe_file(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-print_format", "json",
            "-show_streams", "-show_format",
            "-i", str(path),
        ],
        capture_output=True,
        check=True,
    )
    return json.loads(proc.stdout)


# ── sync, non-blocking feed() logic (no ffmpeg, no time) ─────────────


def test_feed_is_nonblocking_and_counts_drops_when_unconsumed() -> None:
    w = StreamingMp4Writer("/tmp/unused.mp4", width=320, height=240, fps=15)
    blob = _yuv420_blob(320, 240)
    # No pacer running => nothing consumes the slot. First feed sets the
    # latest frame; each subsequent feed overwrites an unconsumed frame
    # and is counted as a drop.
    for _ in range(5):
        w.feed(320, 240, blob)
    assert w.dropped == 4
    assert w._latest is not None


def test_off_size_frame_is_scaled_to_output_size() -> None:
    w = StreamingMp4Writer("/tmp/unused.mp4", width=1280, height=720, fps=15)
    w.feed(640, 480, _yuv420_blob(640, 480))
    assert w._latest is not None
    # Stored frame is normalized to the fixed 720p I420 byte count.
    assert len(w._latest) == 1280 * 720 * 3 // 2


def test_wrong_byte_length_for_declared_size_is_ignored() -> None:
    w = StreamingMp4Writer("/tmp/unused.mp4", width=320, height=240, fps=15)
    w.feed(320, 240, b"\x00" * 100)  # too short for 320x240 I420
    assert w._latest is None
    assert w.dropped == 0


# ── end-to-end through real ffmpeg ───────────────────────────────────


@_ffmpeg
@pytest.mark.asyncio
async def test_streams_a_valid_h264_mp4(tmp_path) -> None:
    out = tmp_path / "clip.mp4"
    w = StreamingMp4Writer(out, width=320, height=240, fps=15)
    await w.start()
    blob = _yuv420_blob(320, 240)
    # Feed over ~0.6s of wall-clock; the 15fps pacer writes several frames.
    for _ in range(20):
        w.feed(320, 240, blob)
        await asyncio.sleep(0.03)
    path = await w.aclose()

    assert path == out
    data = out.read_bytes()
    assert len(data) > 0
    assert data[4:8] == b"ftyp"  # valid MP4 container
    assert w.written >= 1

    info = _ffprobe_file(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert video["codec_name"] in {"h264", "avc1"}
    assert video["width"] == 320
    assert video["height"] == 240


@_ffmpeg
@pytest.mark.asyncio
async def test_aclose_is_idempotent_and_returns_path(tmp_path) -> None:
    out = tmp_path / "clip.mp4"
    w = StreamingMp4Writer(out, width=320, height=240, fps=15)
    await w.start()
    w.feed(320, 240, _yuv420_blob(320, 240))
    await asyncio.sleep(0.1)
    assert await w.aclose() == out
    # Second close must not raise even though ffmpeg is already gone.
    assert await w.aclose() == out
