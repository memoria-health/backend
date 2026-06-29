"""Streaming H.264 MP4 writer with a constant-frame-rate pacer.

A presence session can't be buffered in RAM, so frames stream straight
into a long-lived ``ffmpeg`` that reads raw I420 from its stdin and
writes a fragmented MP4 to a temp file. Fragmented output is crash-safe:
a process kill mid-session still leaves a playable MP4 up to the last
flushed fragment.

Two problems this solves:

1. **Variable input rate.** LiveKit delivers frames at a variable rate
   (and the 4fps lead-in is slower still), but ffmpeg's rawvideo input
   assumes every frame is exactly ``1/fps`` apart. A **CFR pacer** runs
   on the event loop and writes the *latest* frame to ffmpeg once per
   ``1/fps`` wall-clock tick — duplicating the last frame when nothing
   newer arrived. Output duration then tracks real elapsed time.

2. **Back-pressure must never reach detection.** ``feed()`` is a
   synchronous, non-blocking slot overwrite — it never awaits ffmpeg.
   Only the pacer writes to ffmpeg; if the pipe back-pressures, the
   pacer falls behind and ``feed()`` keeps dropping (counted) onto the
   single latest-frame slot. The detection frame callback is never
   blocked.

Off-size frames are scaled to the fixed output size (LiveKit publishes
720p with simulcast off, so this is a rare safety net) so the rawvideo
input dimensions never change mid-stream.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 15
_FFMPEG_EXIT_TIMEOUT = 10.0


def _i420_size(width: int, height: int) -> int:
    """Byte length of one I420 (yuv420p) frame: Y + quarter-size U + V."""
    return width * height * 3 // 2


class StreamingMp4Writer:
    """Streams I420 frames into ffmpeg as a fixed-fps H.264 MP4.

    Lifecycle: ``start()`` launches ffmpeg + the pacer; ``feed()`` is
    called (sync, non-blocking) for every frame; ``aclose()`` flushes
    and finalizes, returning the output path.
    """

    def __init__(
        self,
        out_path: Path | str,
        *,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps: int = _DEFAULT_FPS,
    ) -> None:
        self._out_path = Path(out_path)
        self._width = width
        self._height = height
        self._fps = fps
        self._frame_bytes = _i420_size(width, height)

        self._latest: bytes | None = None
        self._unconsumed = False
        self.dropped = 0
        self.written = 0

        self._proc: asyncio.subprocess.Process | None = None
        self._pacer_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def out_path(self) -> Path:
        return self._out_path

    # ── sync, non-blocking frame intake ─────────────────────────────

    def feed(self, width: int, height: int, data: bytes) -> None:
        """Store ``data`` as the latest frame. Never blocks, never raises.

        Off-size frames are scaled to the fixed output size. A frame
        whose byte length doesn't match its declared size is dropped
        silently (can't be trusted). If the pacer hasn't consumed the
        previous latest frame, that frame is counted as dropped — we
        only ever hold the newest.
        """
        try:
            if width != self._width or height != self._height:
                data = self._scale_to_output(data, width, height)
            elif len(data) != self._frame_bytes:
                return  # wrong byte count for declared size — untrustworthy
            if self._unconsumed:
                self.dropped += 1
            self._latest = data
            self._unconsumed = True
        except Exception:  # must never propagate into the detection callback
            logger.exception("StreamingMp4Writer.feed failed; frame dropped")

    def _scale_to_output(self, data: bytes, width: int, height: int) -> bytes:
        """Resample an off-size I420 frame to the fixed output size.

        Imported lazily — the common path (LiveKit's fixed 720p,
        simulcast off) never hits this, so cv2/numpy load only if an
        off-size frame actually arrives.
        """
        import cv2
        import numpy as np

        if len(data) != _i420_size(width, height):
            raise ValueError("frame byte length does not match its declared size")
        yuv = np.frombuffer(data, dtype=np.uint8).reshape((height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        bgr = cv2.resize(
            bgr, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        out = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        return out.tobytes()

    # ── lifecycle ───────────────────────────────────────────────────

    async def start(self) -> None:
        """Launch ffmpeg and the CFR pacer task."""
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-pix_fmt", "yuv420p",
            # Fragmented MP4 — a partial file (crash mid-session) is still
            # a playable MP4 up to the last flushed fragment.
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4",
            str(self._out_path),
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._pacer_task = asyncio.create_task(self._pace())

    async def _pace(self) -> None:
        """Write the latest frame to ffmpeg once per 1/fps wall-clock
        tick, duplicating it when nothing newer has arrived (CFR)."""
        interval = 1.0 / self._fps
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                frame = self._latest
                if frame is None:
                    continue  # output starts once the first frame arrives
                self._unconsumed = False
                proc = self._proc
                if proc is None or proc.stdin is None:
                    break
                try:
                    proc.stdin.write(frame)
                    await proc.stdin.drain()
                    self.written += 1
                except (BrokenPipeError, ConnectionResetError):
                    break
        except asyncio.CancelledError:
            pass

    async def aclose(self) -> Path:
        """Stop the pacer, flush + close ffmpeg's stdin, await exit.

        Idempotent: a second call (or one after ffmpeg already exited)
        is a no-op. Returns the output path.
        """
        self._closed = True
        if self._pacer_task is not None:
            self._pacer_task.cancel()
            try:
                await self._pacer_task
            except asyncio.CancelledError:
                pass
            self._pacer_task = None

        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=_FFMPEG_EXIT_TIMEOUT)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
        return self._out_path
