"""Streaming H.264 MP4 writer with a constant-frame-rate pacer.

A presence session can't be buffered in RAM, so frames stream straight
into a long-lived ``ffmpeg`` that reads raw I420 from its stdin and
writes a fragmented MP4 to a temp file. Fragmented output is crash-safe:
a process kill mid-session still leaves a playable MP4 up to the last
flushed fragment.

Two problems this solves:

1. **Variable input rate.** LiveKit delivers frames at a variable rate,
   but ffmpeg's rawvideo input assumes every frame is exactly ``1/fps``
   apart. A pacer runs on the event loop and writes the *latest* frame
   to ffmpeg, duplicating it when nothing newer arrived, so that output
   duration tracks real elapsed time.

   The pacer asks the CLOCK how many frames should exist, never its own
   wake count. Waking up N times does NOT mean N/fps seconds passed:
   ``asyncio.sleep`` sleeps *at least* the interval, and the event loop
   this shares with detection adds more on top. Counting wakeups made
   every session shorter than real life — measured 2026-08-04, a 20s
   session produced an 18.7s clip on a completely idle loop and a 6.6s
   clip under moderate load — because ffmpeg was told the input was
   exactly 15fps while the pacer had actually delivered fewer frames
   than that. Video came out time-compressed and drifted steadily away
   from the real-time audio it was muxed with.

   So each wake computes ``round(elapsed * fps)`` — the frame count the
   wall clock says should exist by now — and writes however many
   duplicates it takes to get there. This is the streaming form of what
   ``mp4_remux.frames_to_vfr_mp4`` does in one shot for event clips
   (keep true timestamps, then materialize them onto a uniform fps
   grid). Duplicates of a held frame cost almost nothing: x264 encodes
   them as skip frames. Average bytes/second is unchanged — we write
   the frames we always should have written, just in catch-up bursts
   when the loop stalls instead of one at a time.

   Because output duration now equals real elapsed time, an offset into
   the finished clip is exactly ``captured_at - clip_start``. That is
   what makes time-indexed annotation of this footage possible at all.

2. **Back-pressure must never reach detection.** ``feed()`` is a
   synchronous, non-blocking slot overwrite — it never awaits ffmpeg.
   Only the pacer writes to ffmpeg; if the pipe back-pressures, the
   pacer falls behind and ``feed()`` keeps dropping (counted) onto the
   single latest-frame slot. The detection frame callback is never
   blocked. Catch-up is bounded per wake for the same reason: a long
   stall must not turn into one enormous blocking write.

Off-size frames are scaled to the fixed output size (LiveKit publishes
720p with simulcast off, so this is a rare safety net) so the rawvideo
input dimensions never change mid-stream.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 15
_FFMPEG_EXIT_TIMEOUT = 10.0
# Most catch-up frames one pacer wake may write, as a multiple of fps.
# 1.0 = at most one second of video per wake. Bounds a single blocking
# write after a stall; the deficit that doesn't fit is carried to the
# next wake, and recovery still outruns real time by a wide margin.
_MAX_BURST_SECONDS = 1.0
# The final flush at close may write more, since it happens once and the
# session is ending anyway. Anything still outstanding after this is
# reported as `deficit_frames` rather than silently shortening the clip.
_MAX_CLOSING_BURST_SECONDS = 30.0


def _i420_size(width: int, height: int) -> int:
    """Byte length of one I420 (yuv420p) frame: Y + quarter-size U + V."""
    return width * height * 3 // 2


class StreamingMp4Writer:
    """Streams I420 frames into ffmpeg as a fixed-fps H.264 MP4.

    Lifecycle: ``start()`` launches ffmpeg + the pacer; ``feed()`` is
    called (sync, non-blocking) for every frame; ``aclose()`` flushes
    and finalizes, returning the output path.
    """

    # How long to wait for the child to finish after stdin closes.
    # Plain ffmpeg drains almost immediately; a subclass doing real work
    # per frame needs longer, and killing it early truncates the clip.
    _exit_timeout = _FFMPEG_EXIT_TIMEOUT

    def __init__(
        self,
        out_path: Path | str,
        *,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps: int = _DEFAULT_FPS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._out_path = Path(out_path)
        self._width = width
        self._height = height
        self._fps = fps
        self._clock = clock
        self._frame_bytes = _i420_size(width, height)

        self._latest: bytes | None = None
        self._unconsumed = False
        self.dropped = 0
        self.written = 0
        # Frames the wall clock called for that we never managed to
        # write. Non-zero means this clip IS shorter than the session it
        # recorded — the drift we used to have on every clip, now
        # measured and rare instead of silent and constant.
        self.deficit_frames = 0

        # Real time that output position 0 corresponds to. Set on the
        # first frame actually written; every later frame's position is
        # measured from here, so the clip's timeline is anchored to one
        # known instant. The session uses it to line audio up.
        self._grid_start: float | None = None

        self._proc: asyncio.subprocess.Process | None = None
        self._pacer_task: asyncio.Task[None] | None = None
        self._closed = False
        # How the encoder child ended. Anything other than 0 means the
        # file it wrote cannot be trusted — with plain ffmpeg that is
        # rare, but the masking subclass exits non-zero when the person
        # finder fails to load, and uploading that clip would ship a
        # broken or unmasked file with nothing flagged anywhere.
        self.exit_code: int | None = None

    @property
    def out_path(self) -> Path:
        return self._out_path

    @property
    def first_frame_mono(self) -> float | None:
        """Monotonic time that output position 0 corresponds to, or None
        if nothing was ever written."""
        return self._grid_start

    @property
    def duration_seconds(self) -> float:
        """Length of the encoded video. ffmpeg is told the input is
        exactly ``fps``, so this is frames/fps by construction — and
        because the pacer fills to the clock, it also equals the real
        elapsed time of the session."""
        return self.written / self._fps if self._fps else 0.0

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

    def _build_command(self) -> list[str]:
        """argv for the child that turns piped raw I420 into the output
        MP4. Overridden by `MaskingMp4Writer` to route frames through the
        research censor instead of straight into ffmpeg — everything else
        about this class (the pacer, the timeline, back-pressure) is
        identical either way."""
        return [
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

    async def start(self) -> None:
        """Launch the encoder child and the CFR pacer task."""
        cmd = self._build_command()
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        self._pacer_task = asyncio.create_task(self._pace())

    async def _pace(self) -> None:
        """Keep the encoded frame count level with the wall clock.

        Each wake works out how many frames *should* exist by now and
        writes enough copies of the latest frame to get there, so a late
        or skipped wake costs nothing: the next one makes it up.
        """
        interval = 1.0 / self._fps
        max_burst = max(1, int(self._fps * _MAX_BURST_SECONDS))
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                frame = self._latest
                if frame is None:
                    continue  # output starts once the first frame arrives
                self._unconsumed = False
                if not await self._fill_to_clock(frame, max_burst):
                    break
        except asyncio.CancelledError:
            pass

    async def _fill_to_clock(self, frame: bytes, max_burst: int) -> bool:
        """Write frames until the encoded count matches the wall clock.

        Returns False when ffmpeg's pipe is gone and pacing should stop.
        """
        proc = self._proc
        if proc is None or proc.stdin is None:
            return False

        now = self._clock()
        if self._grid_start is None:
            # First frame out: it defines output position 0, so exactly
            # one frame is due right now.
            self._grid_start = now
            due = 1
        else:
            due = round((now - self._grid_start) * self._fps) + 1 - self.written
        if due <= 0:
            return True

        count = min(due, max_burst)
        try:
            for _ in range(count):
                proc.stdin.write(frame)
                self.written += 1
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return False
        return True

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

        # Final top-up: cover the time between the pacer's last write and
        # right now, so the clip ends when the session ended rather than
        # wherever the last wake happened to land.
        frame = self._latest
        # Guarded on `_proc` so a second aclose() stays a true no-op —
        # otherwise the extra wall-clock time since the first call would
        # be reported as a deficit that never existed.
        if frame is not None and self._grid_start is not None and self._proc:
            await self._fill_to_clock(
                frame, max(1, int(self._fps * _MAX_CLOSING_BURST_SECONDS))
            )
            outstanding = (
                round((self._clock() - self._grid_start) * self._fps)
                + 1
                - self.written
            )
            if outstanding > 0:
                self.deficit_frames = outstanding
                logger.warning(
                    "StreamingMp4Writer: clip is %.1fs short of the session "
                    "it recorded (%d frames); the encoder could not keep up",
                    outstanding / self._fps,
                    outstanding,
                )

        proc = self._proc
        self._proc = None
        if proc is not None:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=self._exit_timeout)
                self.exit_code = proc.returncode
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                self.exit_code = -1  # killed: output is truncated
            if self.exit_code:
                logger.error(
                    "%s child exited %s for %s — output cannot be trusted",
                    type(self).__name__, self.exit_code, self._out_path,
                )
        return self._out_path

    @property
    def ok(self) -> bool:
        """True when the encoder child finished cleanly. False means the
        file must not be uploaded."""
        return self.exit_code == 0


class MaskingMp4Writer(StreamingMp4Writer):
    """Same writer, but every frame is masked on the way out.

    Frames go to an isolated child process that segments the people in
    them and fills those regions flat, then encodes the result. Raw
    footage never reaches disk.

    Doing it here rather than over the finished clip is what makes it
    affordable: the recorder already holds decoded frames, so masking
    inline skips both the decode a post-pass would need and its
    re-encode. It also spreads the work across the session instead of
    stalling on a burst at the end.

    A child process rather than a thread, for the reason `censor.py`
    already isolates tier 1: a native OpenMP deadlock in torch froze the
    whole backend on 2026-07-28, event loop included. That hang lives
    below Python, so a thread could not be recovered but a child can
    always be killed.

    The child dying takes the session's video with it — the pacer sees a
    broken pipe and stops, and ``ok`` goes False so the clip is never
    uploaded.
    """

    # Segmenting is real work, and at close the child still has whatever
    # the pipe buffered to get through. Killing it at ffmpeg's 10s would
    # truncate the clip.
    _exit_timeout = 120.0

    def _build_command(self) -> list[str]:
        import sys

        return [
            sys.executable,
            "-m", "src.services.detection.censor",
            "-",  # in_path unused in stream mode
            str(self._out_path),
            "--stream-i420", f"{self._width}x{self._height}@{self._fps}",
        ]
