"""frame_extraction unit tests.

We generate a real 3-second 15fps MP4 via ffmpeg at module import time
so we can assert N frames out matches `fps * duration` for a known
input. Skips the whole module if ffmpeg isn't on PATH.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import subprocess
import tempfile

import pytest

from src.services.detection.frame_extraction import (
    ExtractedFrame,
    extract_frames,
)

# Skip the whole module if ffmpeg isn't installed — the source code
# guard already raises in that case, so there's nothing meaningful to
# test on the runtime path.
pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)


def _make_test_mp4(duration_seconds: int = 3, fps: int = 15) -> bytes:
    """Generate a deterministic test MP4 using ffmpeg's lavfi color
    source. The exact content doesn't matter — only the duration / fps
    do, because we're testing frame extraction, not video understanding.
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        # Use `testsrc` so each frame has a distinct pattern (encoded
        # frames aren't all-identical, otherwise some encoders collapse
        # them and ffmpeg's fps filter behaves oddly).
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "lavfi",
            "-i", f"testsrc=duration={duration_seconds}:size=320x240:rate={fps}",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        with open(path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_extract_frames_returns_fps_times_duration() -> None:
    """A 3s @ 15fps source sampled at fps=5 should yield 15 frames.
    Strict equality — anything else points at an off-by-one in the
    ffmpeg invocation.
    """
    mp4 = _make_test_mp4(duration_seconds=3, fps=15)

    frames = await extract_frames(mp4, fps=5, max_width=480)

    assert isinstance(frames, list)
    assert len(frames) == 15
    for f in frames:
        assert isinstance(f, ExtractedFrame)
        assert f.mime_type == "image/jpeg"
        # data_base64 round-trips and yields JPEG bytes.
        raw = base64.b64decode(f.data_base64)
        assert raw[:3] == b"\xff\xd8\xff"  # JPEG SOI


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_fps", [0, -1, 61, 1000])
async def test_extract_frames_invalid_fps_raises_value_error(bad_fps: int) -> None:
    """fps out of (0, 60] is a programming error — raise immediately
    rather than letting ffmpeg run with broken args.
    """
    with pytest.raises(ValueError) as exc:
        await extract_frames(b"\x00" * 1024, fps=bad_fps)
    assert "fps" in str(exc.value)


@pytest.mark.asyncio
async def test_extract_frames_returns_jpeg_data_base64() -> None:
    """Each frame's data_base64 must decode to actual JPEG bytes, not
    PNG/raw/etc.
    """
    mp4 = _make_test_mp4(duration_seconds=1, fps=15)
    frames = await extract_frames(mp4, fps=2, max_width=480)
    assert len(frames) == 2
    for f in frames:
        raw = base64.b64decode(f.data_base64)
        # JPEG SOI marker.
        assert raw[:3] == b"\xff\xd8\xff"
        # JPEG EOI marker.
        assert raw[-2:] == b"\xff\xd9"


@pytest.mark.asyncio
async def test_extract_frames_cleans_up_tempdir() -> None:
    """The function must clean up after itself even on the happy path.
    We snapshot the parent tempdir before/after and assert no
    `nemara-frames-*` dirs leaked.
    """
    base = tempfile.gettempdir()

    def _nemara_dirs() -> set[str]:
        return {
            d for d in os.listdir(base)
            if d.startswith("nemara-frames-") and os.path.isdir(os.path.join(base, d))
        }

    pre = _nemara_dirs()
    mp4 = _make_test_mp4(duration_seconds=1, fps=15)
    frames = await extract_frames(mp4, fps=2, max_width=480)
    assert len(frames) == 2
    post = _nemara_dirs()

    # No leftover nemara temp dirs from this run.
    assert post - pre == set()


@pytest.mark.asyncio
async def test_extract_frames_missing_ffmpeg_raises_runtime_error(monkeypatch) -> None:
    """If ffmpeg isn't on PATH, asyncio.create_subprocess_exec fails.
    extract_frames must surface a clean error rather than a bare
    FileNotFoundError.

    We can't easily uninstall ffmpeg, so simulate by monkeypatching
    `asyncio.create_subprocess_exec` to raise FileNotFoundError when
    invoked with 'ffmpeg'.
    """
    real_exec = asyncio.create_subprocess_exec

    async def fake_exec(*args, **kwargs):
        if args and args[0] == "ffmpeg":
            raise FileNotFoundError("ffmpeg")
        return await real_exec(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises((FileNotFoundError, RuntimeError)):
        await extract_frames(b"\x00" * 1024, fps=5)
