"""Encode an in-memory sequence of I420 frames into an MP4 byte string.

Used by the subscriber-bot at tick time: pull the last 15s of
`FrameSnapshot`s out of `FrameBuffer`, hand them here, get back an
MP4 that drops straight into `gemini_provider(mp4=...)` /
`openai_provider(mp4=...)`. This is the byte-shape equivalent of
what the wall phone used to upload via `POST /v1/clips/ingest`, so
the existing provider callables don't change.

ffmpeg pipeline (matches the dashboard / `frame_extraction.py` flags
where it makes sense):

    ffmpeg -f rawvideo -pix_fmt yuv420p -s WxH -r FPS -i pipe:0
           -an
           -c:v libx264 -preset ultrafast -tune zerolatency
           -pix_fmt yuv420p -movflags +faststart
           -f mp4 pipe:1

Why ultrafast + zerolatency: every tick re-muxes a fresh window, so
encoding latency dominates wall-clock; quality is bound by the LLM's
robustness to compression, not the file's archival use.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from src.services.detection.frame_buffer import FrameSnapshot


class Mp4RemuxError(RuntimeError):
    """ffmpeg returned non-zero, isn't on PATH, or didn't produce any
    bytes. Raised from `frames_to_mp4` with the captured stderr in
    the message so the worker can log a meaningful failure.
    """


async def frames_to_mp4(
    frames: Sequence[FrameSnapshot],
    *,
    fps: int = 5,
) -> bytes:
    """Stream raw YUV420P frames into ffmpeg on stdin, read the MP4
    off stdout. Returns the MP4 bytes.

    Every frame in `frames` must have the same `width` and `height`;
    raises ValueError otherwise (ffmpeg requires a constant input
    geometry for `-f rawvideo`).

    Raises `Mp4RemuxError` on any subprocess-side failure (binary
    missing, non-zero exit, empty stdout).
    """
    if not frames:
        raise ValueError("frames_to_mp4 requires at least one frame")
    if not (0 < fps <= 60):
        raise ValueError(f"fps must be in (0, 60], got {fps}")
    width = frames[0].width
    height = frames[0].height
    for i, f in enumerate(frames[1:], start=1):
        if f.width != width or f.height != height:
            raise ValueError(
                f"frame {i} has size {f.width}x{f.height}; expected "
                f"{width}x{height} to match frame 0"
            )

    expected_bytes_per_frame = width * height * 3 // 2  # I420 = 1.5 bpp
    for i, f in enumerate(frames):
        if len(f.data) != expected_bytes_per_frame:
            raise ValueError(
                f"frame {i} has {len(f.data)} bytes; expected "
                f"{expected_bytes_per_frame} for I420 at {width}x{height}"
            )

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart+empty_moov",
            "-f", "mp4",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise Mp4RemuxError("ffmpeg binary not found on PATH") from exc

    # Pass every frame's bytes in order; ffmpeg's `-f rawvideo` reads
    # exactly `width*height*1.5` bytes per frame.
    payload = b"".join(f.data for f in frames)
    try:
        stdout, stderr = await proc.communicate(input=payload)
    except Exception:
        # The subprocess may have exited mid-write — make sure we
        # don't leave a zombie around.
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise Mp4RemuxError(
            f"ffmpeg exited {proc.returncode}: {err or '(no stderr)'}"
        )
    if not stdout:
        raise Mp4RemuxError("ffmpeg produced 0 bytes of MP4 output")
    return stdout
