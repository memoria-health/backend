"""Finalize a recorded session into one uploadable MP4.

The streaming encoder wrote video to a fragmented MP4 and detection's
audio was appended to a raw PCM side-file (s16le, 16 kHz mono — the
format detection already captures). ``mux_av`` combines them in one
ffmpeg pass: video is **stream-copied** (it's already H.264, so no
re-encode) and audio is AAC-encoded into a single ``+faststart`` MP4.

Sync is approximate — both streams are anchored to the session start,
which is good enough for offline analysis (no PyAV in v1). When there's
no audio (the buffer was warming, or the file is empty/missing) the
output is video-only.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MuxError(RuntimeError):
    """ffmpeg failed to mux the session (non-zero exit or missing binary)."""


def _has_audio(audio_path: Path | str | None) -> bool:
    if audio_path is None:
        return False
    p = Path(audio_path)
    return p.exists() and p.stat().st_size > 0


async def mux_av(
    video_path: Path | str,
    audio_path: Path | str | None,
    out_path: Path | str,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Mux ``video_path`` (+ optional ``audio_path`` PCM) into ``out_path``.

    Returns ``out_path`` on success; raises :class:`MuxError` on a
    non-zero ffmpeg exit or a missing ffmpeg binary.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    has_audio = _has_audio(audio_path)

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
    ]
    if has_audio:
        cmd += [
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-i", str(audio_path),
        ]
    cmd += ["-c:v", "copy"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", "-f", "mp4", str(out_path)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise MuxError("ffmpeg binary not found") from exc

    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace")[:500] if stderr else ""
        raise MuxError(f"ffmpeg mux failed (exit {proc.returncode}): {detail}")
    return out_path
