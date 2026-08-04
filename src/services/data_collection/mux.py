"""Finalize a recorded session into one uploadable MP4.

The streaming encoder wrote video to a fragmented MP4 and detection's
audio was appended to a raw PCM side-file (s16le, 16 kHz mono — the
format detection already captures). ``mux_av`` combines them in one
ffmpeg pass: video is **stream-copied** (it's already H.264, so no
re-encode) and audio is AAC-encoded into a single ``+faststart`` MP4.

Alignment is measured, not assumed. The session times both streams off
one clock and passes ``audio_offset_seconds``: audio that started before
the video's first frame is trimmed by that much, audio that started
after is pushed back by that much with leading silence. Assuming both
began together was wrong in practice — the audio stream and the encoder
warm up at their own pace.

``duration_seconds`` caps the output exactly, the way
``mp4_remux.frames_to_vfr_mp4`` does for event clips. It replaces
``-shortest``, which cut the output down to whichever stream happened to
be shorter and so quietly threw away the tail of the audio.

When there's no audio (the buffer was warming, or the file is
empty/missing) the output is video-only.
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
    audio_offset_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> Path:
    """Mux ``video_path`` (+ optional ``audio_path`` PCM) into ``out_path``.

    ``audio_offset_seconds`` is how far the audio has to move to line up
    with the video: positive trims that much off the front, negative
    delays it by that much. ``duration_seconds`` caps the output.

    Returns ``out_path`` on success; raises :class:`MuxError` on a
    non-zero ffmpeg exit or a missing ffmpeg binary.
    """
    video_path = Path(video_path)
    out_path = Path(out_path)
    has_audio = _has_audio(audio_path)
    offset = audio_offset_seconds or 0.0

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
    ]
    if has_audio:
        # Trimming happens on the INPUT (before -i), so ffmpeg seeks
        # past those bytes instead of decoding and discarding them.
        if offset > 0:
            cmd += ["-ss", f"{offset:.3f}"]
        cmd += [
            "-f", "s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-i", str(audio_path),
        ]
    cmd += ["-c:v", "copy"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "64k"]
        if offset < 0:
            # Audio started late: hold it back by the gap so it still
            # lands under the video it belongs to. `all=1` covers every
            # channel; without it only the first is delayed.
            cmd += ["-af", f"adelay={int(round(-offset * 1000))}:all=1"]
    else:
        cmd += ["-an"]
    if duration_seconds and duration_seconds > 0:
        # Applies to every output stream, so the container's duration is
        # the truth rather than whichever stream ran longest.
        cmd += ["-t", f"{duration_seconds:.3f}"]
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
