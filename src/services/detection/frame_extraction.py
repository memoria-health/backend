"""ffmpeg subprocess frame extraction.

Verbatim port of `nemara-temp-dashboard/lib/extract-frames.ts`. Used
by the OpenAI provider — Gemini takes the MP4 inline directly. Same
args as the dashboard: `fps=5, scale='min(480,iw)':-2, q:v=5`.
"""

import asyncio
import base64
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedFrame:
    """A single JPEG-encoded frame from the input MP4."""
    data_base64: str
    mime_type: str = "image/jpeg"


async def extract_frames(
    mp4: bytes,
    *,
    fps: int = 5,
    max_width: int = 480,
    jpeg_quality: int = 5,
) -> Sequence[ExtractedFrame]:
    """Decode the MP4 with ffmpeg and emit one JPEG per sampled frame.

    `jpeg_quality` is ffmpeg's `-q:v` — 2 best, 31 worst. 5 matches the
    dashboard's default and is fine for fall detection at typical room
    range.

    Raises RuntimeError if ffmpeg isn't on PATH, returns 0 frames, or
    exits non-zero. Cleans up its temp dir even on error.
    """
    if not (0 < fps <= 60):
        raise ValueError(f"fps must be in (0, 60], got {fps}")

    workdir = tempfile.mkdtemp(prefix="nemara-frames-")
    in_path = os.path.join(workdir, "input.mp4")
    frame_pattern = os.path.join(workdir, "frame_%04d.jpg")
    try:
        with open(in_path, "wb") as f:
            f.write(mp4)

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-vf", f"fps={fps},scale='min({max_width},iw)':-2",
            "-q:v", str(jpeg_quality),
            frame_pattern,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ffmpeg exited {proc.returncode}: {err.strip() or '(no stderr)'}"
            )

        files = sorted(
            f for f in os.listdir(workdir)
            if f.startswith("frame_") and f.endswith(".jpg")
        )
        if not files:
            raise RuntimeError(
                f"ffmpeg produced 0 frames from {len(mp4)}-byte input"
            )

        frames: list[ExtractedFrame] = []
        for fname in files:
            with open(os.path.join(workdir, fname), "rb") as f:
                frames.append(
                    ExtractedFrame(data_base64=base64.b64encode(f.read()).decode())
                )
        return frames
    finally:
        # Best-effort cleanup; never raise from finally.
        try:
            for fname in os.listdir(workdir):
                try:
                    os.remove(os.path.join(workdir, fname))
                except OSError:
                    pass
            os.rmdir(workdir)
        except OSError:
            pass
