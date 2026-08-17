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

Off-size frames are scaled to the fixed output size so the rawvideo
input dimensions never change mid-stream. In production the publisher
sends native resolution (2K), so this scaling runs on every recorded
frame — see `_scale_to_output` for why it resizes I420 planes in
place instead of round-tripping through BGR.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
# Most unwritten time the pacer will keep chasing. Past this the encoder
# is not merely late, it is too slow for the stream, and the missing
# frames are written off rather than chased forever. See
# `_forgive_impossible_debt`.
#
# Deliberately generous. A tighter cap (5s, tried 2026-08-04) wrote off
# time the pacer would have recovered on its own: a busy event loop
# stalls in bursts far longer than a frame, and catching up afterwards
# is exactly what the pacer is for. Measured, 5s turned a clean 20.00s
# clip into 15.20s under moderate load. This is a backstop for an
# encoder that genuinely cannot keep up, not a pacing knob.
_MAX_OUTSTANDING_SECONDS = 30.0


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

        In production this is NOT a rare path: the publisher sends the
        camera's native resolution (2K on Tapo), so every recorded
        frame lands here. It runs synchronously inside the detection
        frame callback, so it must allocate as little as possible.
        Resizing each I420 plane directly (Y, then quarter-size U and
        V) skips the YUV→BGR→YUV round-trip an earlier version did —
        that round-trip allocated ~14 MB of transient BGR per 2K frame
        at the delivered rate, which fragmented the allocator badly
        enough to ratchet RSS for the life of the process.
        """
        import cv2
        import numpy as np

        if len(data) != _i420_size(width, height):
            raise ValueError("frame byte length does not match its declared size")
        src = np.frombuffer(data, dtype=np.uint8)
        out = np.empty(self._frame_bytes, dtype=np.uint8)
        dw, dh = self._width, self._height
        # Plane offsets in I420: full-res Y, then U and V at half
        # resolution in each dimension.
        planes = (
            (0, width, height, 0, dw, dh),
            (width * height, width // 2, height // 2,
             dw * dh, dw // 2, dh // 2),
            (width * height * 5 // 4, width // 2, height // 2,
             dw * dh * 5 // 4, dw // 2, dh // 2),
        )
        for s_off, sw, sh, d_off, pdw, pdh in planes:
            cv2.resize(
                src[s_off:s_off + sw * sh].reshape(sh, sw),
                (pdw, pdh),
                dst=out[d_off:d_off + pdw * pdh].reshape(pdh, pdw),
                interpolation=cv2.INTER_AREA,
            )
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

    def _forgive_impossible_debt(self, now: float) -> int:
        """Frames owed right now, with unreachable ones written off.

        A child that encodes slower than real time falls behind by a
        fixed amount every second, and no amount of catching up fixes
        that — the shortfall is the child's throughput, not our pacing.
        Chasing it forever just grows a queue.

        So past `_MAX_OUTSTANDING_SECONDS` the excess is written off:
        counted in `deficit_frames` and the timeline re-anchored so we
        stop trying to make up time we cannot encode.

        The honest cost: re-anchoring shifts everything after it, so once
        `deficit_frames` is non-zero, offsets later in THIS clip are no
        longer exactly `captured_at - clip_start`. That is why the
        counter exists — a clip that lost time says so, rather than
        quietly lying about when things happened.
        """
        due = round((now - self._grid_start) * self._fps) + 1 - self.written
        cap = int(self._fps * _MAX_OUTSTANDING_SECONDS)
        if due <= cap:
            return due
        forgiven = due - cap
        self.deficit_frames += forgiven
        self._grid_start += forgiven / self._fps
        logger.warning(
            "%s cannot keep up: wrote off %.1fs (%d frames) for %s",
            type(self).__name__, forgiven / self._fps, forgiven,
            self._out_path,
        )
        return cap

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
            due = self._forgive_impossible_debt(now)
        if due <= 0:
            return True

        count = min(due, max_burst)
        try:
            for _ in range(count):
                proc.stdin.write(frame)
                self.written += 1
                # Drain per frame, not per burst. The stdin transport
                # buffers everything handed to write() and only applies
                # back-pressure at the next drain(), so a burst-level
                # drain let a catch-up write park `count × frame_bytes`
                # in one bytearray — ~620 MB at the closing burst's 450
                # frames. Per-frame drain caps the buffer at the
                # transport's high-water mark; when the pipe is free it
                # returns immediately, so the healthy path pays nothing.
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
                # Add, never replace: the pacer may already have written
                # off time mid-session (`_forgive_impossible_debt`), and
                # overwriting here would hide it.
                self.deficit_frames += outstanding
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
    def deficit_seconds(self) -> float:
        """Wall clock this clip could not represent at all. 0.0 on a
        healthy clip; anything else means the file is that much shorter
        than the session it recorded."""
        return self.deficit_frames / self._fps if self._fps else 0.0

    @property
    def ok(self) -> bool:
        """True when the encoder child finished cleanly. False means the
        file must not be uploaded."""
        return self.exit_code == 0
