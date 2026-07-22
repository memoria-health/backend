"""Encode an in-memory sequence of I420 frames (plus optional PCM
audio) into an MP4 byte string.

Used by the subscriber-bot at tick time: pull the last 15s of
`FrameSnapshot`s out of `FrameBuffer` (and optionally PCM out of
`AudioBuffer`), hand them here, get back an MP4 that drops straight
into `gemini_provider(mp4=...)` / `openai_provider(mp4=...)`.

ffmpeg pipeline:

    ffmpeg -f rawvideo -pix_fmt yuv420p -s WxH -r FPS -i pipe:0
           [-f s16le -ar RATE -ac CH -i <audio.pcm>]
           -c:v libx264 -preset ultrafast -tune zerolatency -pix_fmt yuv420p
           [-c:a aac -b:a 64k -shortest] | -an
           -movflags +faststart+empty_moov -f mp4 pipe:1

Why ultrafast + zerolatency: every tick re-muxes a fresh window, so
encoding latency dominates wall-clock; quality is bound by the LLM's
robustness to compression, not the file's archival use.

The PCM goes through a tempfile because asyncio subprocess only
exposes stdin/stdout/stderr; piping two distinct streams to one
ffmpeg process is doable with `pass_fds=` but the temp-file route is
much simpler and the audio blob is tiny (~32 kB/s).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import Sequence

from src.services.detection import frame_codec
from src.services.detection.frame_buffer import FrameSnapshot

log = logging.getLogger(__name__)


class Mp4RemuxError(RuntimeError):
    """ffmpeg returned non-zero, isn't on PATH, or didn't produce any
    bytes. Raised from `frames_to_mp4` with the captured stderr in
    the message so the worker can log a meaningful failure.
    """


async def frames_to_mp4(
    frames: Sequence[FrameSnapshot],
    *,
    fps: int = 5,
    pcm: bytes | None = None,
    pcm_sample_rate: int = 16000,
    pcm_num_channels: int = 1,
) -> bytes:
    """Stream raw YUV420P frames into ffmpeg on stdin, optionally
    layer in a PCM audio track from `pcm`, and read the MP4 off
    stdout.

    `pcm`, when provided, must be int16 little-endian PCM at the
    given sample rate and channel count. The exact byte length isn't
    validated (audio rarely lines up perfectly with the video window
    and ffmpeg handles the discrepancy via `-shortest`).

    Raises `Mp4RemuxError` on any subprocess-side failure (binary
    missing, non-zero exit, empty stdout). Raises ValueError on
    inconsistent frame shapes or invalid parameters.
    """
    if not frames:
        raise ValueError("frames_to_mp4 requires at least one frame")
    if not (0 < fps <= 60):
        raise ValueError(f"fps must be in (0, 60], got {fps}")
    if pcm is not None:
        if pcm_sample_rate <= 0:
            raise ValueError(f"pcm_sample_rate must be > 0, got {pcm_sample_rate}")
        if pcm_num_channels <= 0:
            raise ValueError(f"pcm_num_channels must be > 0, got {pcm_num_channels}")

    # WebRTC's adaptive bitrate can change publish resolution mid-
    # stream — the buffer may contain a mix of (e.g.) 320x180 startup
    # frames + later 1280x720 steady-state frames. Both input modes
    # need constant dimensions (rawvideo strictly; mjpeg for a stable
    # h264 encode), so filter the window to only frames matching the
    # MOST RECENT frame's size and drop the rest. Last frame's
    # resolution is what the LLM should see — the current publish state.
    width = frames[-1].width
    height = frames[-1].height
    filtered = [f for f in frames if f.width == width and f.height == height]
    if len(filtered) < len(frames):
        log.info(
            "mp4_remux: dropped %d/%d frames at non-current size; "
            "kept %d at %dx%d",
            len(frames) - len(filtered), len(frames), len(filtered), width, height,
        )
    frames = filtered  # type: ignore[assignment]
    if not frames:
        raise ValueError("frames_to_mp4: no frames at the most-recent resolution")

    # Format sniff (JPEG-in-buffer, 2026-07-19): the live buffer now
    # stores JPEG frames; ffmpeg consumes them directly as an MJPEG
    # stream (its C decoder, no Python-side decode). Legacy raw-I420
    # windows (tests, transitional) keep the rawvideo path. A window
    # never legitimately mixes formats — reject if it does.
    jpeg_input = frame_codec.is_jpeg(frames[0].data)
    if jpeg_input:
        for i, f in enumerate(frames):
            if not frame_codec.is_jpeg(f.data):
                raise ValueError(
                    f"frame {i} is not JPEG in a JPEG window",
                )
    else:
        expected_bytes_per_frame = width * height * 3 // 2  # I420 = 1.5 bpp
        for i, f in enumerate(frames):
            if len(f.data) != expected_bytes_per_frame:
                raise ValueError(
                    f"frame {i} has {len(f.data)} bytes; expected "
                    f"{expected_bytes_per_frame} for I420 at {width}x{height}"
                )

    audio_path: str | None = None
    if pcm:
        # Write the PCM blob to a unique tempfile; ffmpeg reads it as
        # a second input. delete=False because ffmpeg won't have the
        # fd; we unlink in the finally block below.
        af = tempfile.NamedTemporaryFile(suffix=".pcm", delete=False)
        try:
            af.write(pcm)
        finally:
            af.close()
        audio_path = af.name

    try:
        cmd: list[str] = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
        if jpeg_input:
            cmd += ["-f", "mjpeg", "-r", str(fps), "-i", "pipe:0"]
        else:
            cmd += [
                "-f", "rawvideo",
                "-pix_fmt", "yuv420p",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "pipe:0",
            ]
        if audio_path is not None:
            cmd += [
                "-f", "s16le",
                "-ar", str(pcm_sample_rate),
                "-ac", str(pcm_num_channels),
                "-i", audio_path,
            ]
        # veryfast+crf20 replaced ultrafast (2026-07-19): this encode
        # is what the VLM actually SEES, and ultrafast at default crf
        # was the lossiest hop in the whole pipeline. ~2× encode time
        # (~0.5s/window at 2K) buys visibly cleaner frames for
        # detection accuracy.
        cmd += [
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
        ]
        if audio_path is not None:
            cmd += [
                "-c:a", "aac",
                "-b:a", "64k",
                "-shortest",
            ]
        else:
            cmd += ["-an"]
        cmd += [
            "-movflags", "+faststart+empty_moov",
            "-f", "mp4",
            "pipe:1",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise Mp4RemuxError("ffmpeg binary not found on PATH") from exc

        payload = b"".join(f.data for f in frames)
        try:
            stdout, stderr = await proc.communicate(input=payload)
        except Exception:
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
    finally:
        if audio_path is not None:
            try:
                os.unlink(audio_path)
            except FileNotFoundError:
                pass
