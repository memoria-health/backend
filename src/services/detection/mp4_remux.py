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
import itertools
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


async def frames_to_vfr_mp4(
    frames: Sequence[FrameSnapshot],
    *,
    fps: int = 8,
    end_at: float | None = None,
    pcm: bytes | None = None,
    pcm_sample_rate: int = 16000,
    pcm_num_channels: int = 1,
    pcm_starts_at: float | None = None,
) -> bytes:
    """Encode frames into a wall-clock-faithful VARIABLE-frame-rate MP4.

    `frames_to_mp4` above declares a fixed input rate (`-r fps`), which
    throws every frame's real `captured_at` away — fine for the VLM
    (it doesn't watch in real time), but for INCIDENT CLIPS it plays
    back frame-count/fps seconds regardless of how much wall-clock the
    window actually covered. Frames arrive well below clip_fps (WebRTC
    delivers fewer frames the stiller the scene), so clips ran fast,
    ran short, teleported across still stretches, and drifted against
    the wall-clock-true audio (2026-07-28 soak: 10-12s wandering clips
    for a ~26s window; a fall clip whose still-on-the-couch lead-in
    collapsed into a 2s teleport).

    This encoder keeps the timestamps instead: each frame's display
    duration is its real gap to the next frame (ffmpeg concat demuxer
    with per-frame `duration` entries), so playback duration == wall
    clock, motion plays at true speed, and a quiet stretch shows its
    real sparse frames for their real durations — exactly how a live
    WebRTC player renders the same feed.

    The true timestamps are materialized onto a uniform `fps` output
    grid (`-vf fps=N` + an exact `-t` cap — raw `-vsync vfr`
    passthrough writes broken per-sample durations that make the
    container's duration metadata lie to scrubbers, and the concat
    demuxer pads past EOF). Grid duplicates of a held frame encode as
    skip-frames (~0 bytes), and the CFR container scrubs correctly in
    every player, AVPlayer included.

    `end_at` (monotonic, same clock as `captured_at`) extends the last
    frame to the true end of the clip window; omitted, the last frame
    gets the window's median gap.

    Audio: `pcm` is muxed WITHOUT `-shortest` (video now spans true
    wall-clock, so nothing needs truncating). `pcm_starts_at` (monotonic)
    anchors alignment: leading PCM is trimmed (or silence prepended) so
    audio t=0 coincides with the first frame's `captured_at`.

    Only JPEG-in-buffer windows skip re-encoding (the live path since
    2026-07-19); raw-I420 windows (tests, transitional) are converted
    per-frame via `frame_codec.i420_to_jpeg`.
    """
    if not frames:
        raise ValueError("frames_to_vfr_mp4 requires at least one frame")
    if not (0 < fps <= 60):
        raise ValueError(f"fps must be in (0, 60], got {fps}")
    if pcm is not None:
        if pcm_sample_rate <= 0:
            raise ValueError(f"pcm_sample_rate must be > 0, got {pcm_sample_rate}")
        if pcm_num_channels <= 0:
            raise ValueError(f"pcm_num_channels must be > 0, got {pcm_num_channels}")

    # Same constant-dimensions rule as frames_to_mp4: keep only frames
    # at the most-recent resolution (mid-stream ABR switches).
    width, height = frames[-1].width, frames[-1].height
    filtered = [f for f in frames if f.width == width and f.height == height]
    if len(filtered) < len(frames):
        log.info(
            "mp4_remux.vfr: dropped %d/%d frames at non-current size; "
            "kept %d at %dx%d",
            len(frames) - len(filtered), len(frames), len(filtered),
            width, height,
        )
    frames = filtered  # type: ignore[assignment]
    if not frames:
        raise ValueError("frames_to_vfr_mp4: no frames at the most-recent resolution")

    # Strictly increasing timestamps — a duplicate/out-of-order
    # captured_at would yield a zero/negative duration entry.
    ordered = sorted(frames, key=lambda f: f.captured_at)

    gaps = [
        max(0.001, b.captured_at - a.captured_at)
        for a, b in itertools.pairwise(ordered)
    ]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 0.125
    last_duration = median_gap
    if end_at is not None:
        tail = end_at - ordered[-1].captured_at
        if tail > 0:
            last_duration = tail
    durations = [*gaps, last_duration]

    # Audio alignment: shift the PCM so its t=0 == first frame's
    # captured_at. `pcm_starts_at` is the MEASURED capture time of the
    # first audio chunk — never inferred from the byte length, which
    # silently displaces the whole track whenever the buffer has a gap.
    bytes_per_second = pcm_sample_rate * 2 * pcm_num_channels
    if pcm and pcm_starts_at is not None:
        audio_start = pcm_starts_at
        lead = ordered[0].captured_at - audio_start
        if lead > 0:
            cut = int(lead * pcm_sample_rate) * 2 * pcm_num_channels
            pcm = pcm[cut:]
        elif lead < 0:
            pad = int(-lead * pcm_sample_rate) * 2 * pcm_num_channels
            pcm = b"\x00" * pad + pcm

    # Scratch MUST live in the clip scratch dir: these are RAW frames,
    # written before the sensitivity screen and before any censoring,
    # and the sweep at startup only cleans that one directory. Using
    # the system temp root left uncensored bedroom/bathroom JPEGs
    # behind permanently whenever the process was hard-killed
    # mid-encode (found in review 2026-07-29).
    from src.services.clip_archival import _scratch_dir

    with tempfile.TemporaryDirectory(
        prefix="nemara_vfr_", dir=_scratch_dir(),
    ) as tmpdir:
        def _materialize() -> list[str]:
            """JPEG-convert (when needed) and write every frame, then
            build the concat listing. Runs in a THREAD: this is tens of
            MB of synchronous file writes plus per-frame encoding, and
            the event loop it would otherwise sit on is the one feeding
            every camera's frames."""
            out = ["ffconcat version 1.0"]
            for i, (f, dur) in enumerate(zip(ordered, durations, strict=True)):
                data = f.data
                if not frame_codec.is_jpeg(data):
                    data = frame_codec.i420_to_jpeg(
                        f.width, f.height, data, quality=90,
                    )
                path = os.path.join(tmpdir, f"f{i:05d}.jpg")
                with open(path, "wb") as fh:
                    fh.write(data)
                out.append(f"file '{path}'")
                out.append(f"duration {dur:.6f}")
            return out

        listing = await asyncio.to_thread(_materialize)
        # ffmpeg's concat demuxer ignores the LAST entry's duration, so
        # the final frame contributes nothing to the stream. With many
        # frames that just clips the tail; with ONE frame it produces a
        # zero-length, unplayable MP4 that still passes the "did ffmpeg
        # emit bytes?" check and gets uploaded with has_clip=true —
        # a caregiver taps a fall clip and nothing plays. Repeating the
        # last file (the documented slideshow idiom) gives it a real
        # duration. Reachable whenever a resolution switch or an empty
        # post-roll leaves a single surviving frame.
        listing.append(listing[-2])
        list_path = os.path.join(tmpdir, "frames.ffconcat")
        with open(list_path, "w") as fh:
            fh.write("\n".join(listing) + "\n")

        audio_path: str | None = None
        if pcm:
            audio_path = os.path.join(tmpdir, "audio.pcm")
            with open(audio_path, "wb") as fh:
                fh.write(pcm)

        cmd: list[str] = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", list_path,
        ]
        if audio_path is not None:
            cmd += [
                "-f", "s16le",
                "-ar", str(pcm_sample_rate),
                "-ac", str(pcm_num_channels),
                "-i", audio_path,
            ]
        # Exact clip length: first frame -> end of the window. `-t` caps
        # every output stream (video AND audio) at precisely this, which
        # both trims the concat demuxer's EOF padding and guarantees the
        # container's duration metadata is honest.
        total = sum(durations)
        cmd += [
            "-vf", f"fps={fps}",
            "-t", f"{total:.3f}",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
        ]
        cmd += ["-c:a", "aac", "-b:a", "64k"] if audio_path is not None else ["-an"]
        cmd += ["-movflags", "+faststart+empty_moov", "-f", "mp4", "pipe:1"]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise Mp4RemuxError("ffmpeg binary not found on PATH") from exc
        try:
            stdout, stderr = await proc.communicate()
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
