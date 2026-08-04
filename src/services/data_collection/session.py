"""Per-camera recording session state machine.

A ``SessionRecorder`` owns at most one in-flight recording at a time and
drives it through:

    IDLE ──collect──▶ RECORDING ──!collect──▶ LINGERING ──collect──▶ RECORDING
                          │                       │
                       roll │                       │ linger expiry
                          ▼                       ▼
                     (finalize + restart)      (finalize → IDLE)

``on_tick(collect, ...)`` is called once per detection tick (~10s) and
is the only thing that changes state. ``feed_video`` / ``feed_audio``
are called per-frame in between ticks and route to the active session's
encoder + audio side-file (both no-ops when idle). On finalize the
recorder calls ``on_finalized(FinalizedSession)`` — the manager handles
the audio mux + B2 upload from there.

Time is injected (``clock`` = monotonic seconds for elapsed math,
``utcnow`` = wall-clock for the object key), and the video encoder comes
from an injected factory, so the whole machine is testable with fakes.
"""

from __future__ import annotations

import enum
import logging
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

logger = logging.getLogger(__name__)


class SessionState(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    LINGERING = "lingering"


@dataclass(frozen=True)
class RecorderCtx:
    """Everything needed to name + tag a clip, captured at session start."""

    caregiver_id: UUID
    patient_id: UUID
    # Which camera the session records — carried through to the manager's
    # upload-boundary privacy re-check (per-camera privacy_enabled,
    # 2026-07-27). The manager keys sessions by camera_id already; this
    # puts the id on the finalized payload so the fire-and-forget upload
    # task can re-check without reaching back into the session dict.
    camera_id: UUID
    room: str
    gate: str  # "poi" (person-of-interest matched) | "person" (presence only)
    confidence: float | None


@dataclass(frozen=True)
class FinalizedSession:
    """Handed to ``on_finalized`` when a session closes."""

    video_path: Path
    audio_path: Path | None
    session_start_utc: datetime
    ctx: RecorderCtx
    # How far the audio needs to move to line up with the video, in
    # seconds. Positive means audio began BEFORE the video's first frame,
    # so that much has to come off the front; negative means it began
    # after, so that much silence goes on the front. None when either
    # side never produced a measured start time (no audio at all, or a
    # test encoder that doesn't report one) — the muxer then leaves the
    # audio where it is, which is the old behaviour.
    audio_offset_seconds: float | None = None
    # Encoded video length. Used to cap the muxed output exactly, so the
    # container's duration is honest instead of whatever the longer
    # stream happened to be.
    video_duration_seconds: float | None = None


class _Encoder(Protocol):
    async def start(self) -> None: ...
    def feed(self, width: int, height: int, data: bytes) -> None: ...
    async def aclose(self) -> Path: ...


# Shape of the audio detection captures and hands us (see room_worker's
# audio consumer). Two bytes per sample, one channel, 16 kHz.
_AUDIO_SAMPLE_RATE = 16000
_AUDIO_CHANNELS = 1
_AUDIO_BYTES_PER_SAMPLE = 2
# Slack before we treat a late audio chunk as a real gap. Normal network
# jitter moves chunks around by tens of milliseconds; only a stall long
# enough to shift everything after it audibly is worth patching.
_AUDIO_GAP_TOLERANCE_SECONDS = 0.25


class _LeadInFrame(Protocol):
    width: int
    height: int
    data: bytes


class SessionRecorder:
    def __init__(
        self,
        *,
        camera_id: UUID,
        ctx: RecorderCtx,
        temp_dir: Path | str,
        encoder_factory: Callable[[Path], _Encoder],
        on_finalized: Callable[[FinalizedSession], Awaitable[None]],
        clock: Callable[[], float],
        utcnow: Callable[[], datetime],
        roll_seconds: float = 900.0,
        linger_seconds: float = 45.0,
    ) -> None:
        self.camera_id = camera_id
        self._ctx = ctx
        self._temp_dir = Path(temp_dir)
        self._encoder_factory = encoder_factory
        self._on_finalized = on_finalized
        self._clock = clock
        self._utcnow = utcnow
        self._roll_seconds = roll_seconds
        self._linger_seconds = linger_seconds

        self.state = SessionState.IDLE
        self._encoder: _Encoder | None = None
        self._audio_fh: Any = None
        self._video_path: Path | None = None
        self._audio_path: Path | None = None
        self._session_start_mono: float = 0.0
        self._session_start_utc: datetime | None = None
        self._linger_start_mono: float = 0.0
        # Real time the first audio byte was captured, and how many bytes
        # we've written since. Together these say what moment any point
        # in the audio file belongs to — measured, never inferred from
        # the file's length, because a stall in the audio stream would
        # otherwise shift everything after it out of place for good.
        # (`mp4_remux` learned this the hard way on event clips.)
        self._audio_start_mono: float | None = None
        self._audio_bytes: int = 0

    @property
    def is_active(self) -> bool:
        return self.state is not SessionState.IDLE

    # ── tick-driven state machine ───────────────────────────────────

    async def on_tick(
        self,
        collect: bool,
        *,
        lead_in_frames: Sequence[_LeadInFrame] | None = None,
        lead_in_audio: bytes | None = None,
    ) -> None:
        now = self._clock()
        if self.state is SessionState.IDLE:
            if collect:
                await self._start_session(lead_in_frames, lead_in_audio)
        elif self.state is SessionState.RECORDING:
            if not collect:
                self.state = SessionState.LINGERING
                self._linger_start_mono = now
            elif now - self._session_start_mono >= self._roll_seconds:
                # Still collecting but the session hit the max roll —
                # finalize and immediately start a fresh, contiguous one
                # (no lead-in; the stream didn't pause).
                await self._finalize()
                await self._start_session(None, None)
        elif self.state is SessionState.LINGERING:
            if collect:
                self.state = SessionState.RECORDING  # presence resumed
            elif now - self._linger_start_mono >= self._linger_seconds:
                await self._finalize()

    # ── per-frame intake (sync, non-blocking) ───────────────────────

    def feed_video(self, width: int, height: int, data: bytes) -> None:
        if self._encoder is not None and self.state in (
            SessionState.RECORDING,
            SessionState.LINGERING,
        ):
            self._encoder.feed(width, height, data)

    def feed_audio(self, pcm: bytes, captured_at: float | None = None) -> None:
        """Append captured audio. ``captured_at`` is when the chunk was
        received (same monotonic clock as the video frames); pass it so a
        stall in the audio stream can be patched with silence instead of
        sliding everything after it earlier.
        """
        if self._audio_fh is None or self.state not in (
            SessionState.RECORDING,
            SessionState.LINGERING,
        ):
            return
        try:
            if captured_at is not None:
                if self._audio_start_mono is None:
                    self._audio_start_mono = captured_at
                else:
                    self._pad_audio_gap(captured_at)
            self._audio_fh.write(pcm)
            self._audio_bytes += len(pcm)
        except Exception:  # never propagate into the audio callback
            logger.exception("SessionRecorder.feed_audio write failed")

    def _pad_audio_gap(self, captured_at: float) -> None:
        """Insert silence for time the audio stream didn't cover.

        The stream can drop out and reconnect mid-session. Bytes alone
        can't tell: the file just runs on, so everything recorded after
        the gap plays too early, by the length of the gap, for the rest
        of the clip. Comparing where this chunk *belongs* against how
        much audio we hold exposes the hole so we can fill it.
        """
        assert self._audio_start_mono is not None
        bytes_per_second = (
            _AUDIO_SAMPLE_RATE * _AUDIO_BYTES_PER_SAMPLE * _AUDIO_CHANNELS
        )
        belongs_at = (captured_at - self._audio_start_mono) * bytes_per_second
        missing = int(belongs_at) - self._audio_bytes
        if missing <= _AUDIO_GAP_TOLERANCE_SECONDS * bytes_per_second:
            return
        # Whole samples only — a half-sample of padding would flip every
        # following sample's byte order and turn the rest into noise.
        missing -= missing % (_AUDIO_BYTES_PER_SAMPLE * _AUDIO_CHANNELS)
        self._audio_fh.write(b"\x00" * missing)
        self._audio_bytes += missing
        logger.info(
            "SessionRecorder: filled a %.2fs hole in the audio stream "
            "camera=%s", missing / bytes_per_second, self.camera_id,
        )

    # ── transitions ─────────────────────────────────────────────────

    async def _start_session(
        self,
        lead_in_frames: Sequence[_LeadInFrame] | None,
        lead_in_audio: bytes | None,
    ) -> None:
        sid = uuid.uuid4().hex
        self._video_path = self._temp_dir / f"{sid}.video.mp4"
        self._audio_path = self._temp_dir / f"{sid}.audio.pcm"
        self._session_start_mono = self._clock()
        self._session_start_utc = self._utcnow()
        # Fresh session, fresh audio timeline.
        self._audio_start_mono = None
        self._audio_bytes = 0

        encoder = self._encoder_factory(self._video_path)
        await encoder.start()
        self._encoder = encoder
        self._audio_fh = open(self._audio_path, "wb")

        # Seed the lead-in (the analyzed 4fps window + matching audio)
        # BEFORE flipping to RECORDING, so it lands at the head of the file.
        if lead_in_frames:
            for f in lead_in_frames:
                encoder.feed(f.width, f.height, f.data)
        if lead_in_audio:
            try:
                self._audio_fh.write(lead_in_audio)
            except Exception:
                logger.exception("SessionRecorder lead-in audio write failed")

        self.state = SessionState.RECORDING

    async def _finalize(self) -> None:
        # Detach state first so any concurrent feed_video/feed_audio
        # during the awaits below becomes a no-op (we're closing).
        encoder = self._encoder
        audio_fh = self._audio_fh
        audio_path = self._audio_path
        start_utc = self._session_start_utc
        self._encoder = None
        self._audio_fh = None
        self.state = SessionState.IDLE

        if audio_fh is not None:
            try:
                audio_fh.close()
            except Exception:
                logger.exception("SessionRecorder audio file close failed")

        audio_start = self._audio_start_mono
        self._audio_start_mono = None
        self._audio_bytes = 0

        video_path: Path | None = None
        if encoder is not None:
            try:
                video_path = await encoder.aclose()
            except Exception:
                logger.exception("SessionRecorder encoder aclose failed")

        if video_path is None or start_utc is None:
            return  # nothing usable to hand off

        # Line the two streams up against the one clock they share.
        # `first_frame_mono` is the moment the video's opening frame was
        # taken; `audio_start` the moment its first sample was. The gap
        # between them is how far the audio has to move. Both come from
        # measurement, so a slow start on either side is corrected rather
        # than baked in. Real encoders report these; test fakes don't,
        # and then we simply don't shift anything.
        video_start = getattr(encoder, "first_frame_mono", None)
        duration = getattr(encoder, "duration_seconds", None)
        offset = (
            video_start - audio_start
            if video_start is not None and audio_start is not None
            else None
        )

        try:
            await self._on_finalized(
                FinalizedSession(
                    video_path=video_path,
                    audio_path=audio_path,
                    session_start_utc=start_utc,
                    ctx=self._ctx,
                    audio_offset_seconds=offset,
                    video_duration_seconds=duration,
                )
            )
        except Exception:
            logger.exception("SessionRecorder on_finalized callback failed")

    async def abort(self) -> None:
        """Force-ABORT the in-flight session: tear down the encoder and
        DISCARD both temp files WITHOUT muxing or uploading.

        This is the privacy path. The normal gate-off route
        (RECORDING → LINGERING → linger expiry → ``_finalize`` → upload)
        would push footage captured *before* privacy turned on straight
        to storage. ``abort`` instead drops the bytes on the device:
        ``_on_finalized`` (the upload hand-off) is never called.
        """
        encoder = self._encoder
        audio_fh = self._audio_fh
        video_path = self._video_path
        audio_path = self._audio_path
        self._encoder = None
        self._audio_fh = None
        self._video_path = None
        self._audio_path = None
        self._audio_start_mono = None
        self._audio_bytes = 0
        self.state = SessionState.IDLE

        if audio_fh is not None:
            try:
                audio_fh.close()
            except Exception:
                logger.exception("SessionRecorder abort audio close failed")

        # Tear the encoder down (terminates the ffmpeg subprocess) but
        # ignore the returned path — we delete it instead of handing off.
        if encoder is not None:
            try:
                await encoder.aclose()
            except Exception:
                logger.exception("SessionRecorder abort encoder aclose failed")

        for path in (video_path, audio_path):
            if path is not None:
                try:
                    Path(path).unlink(missing_ok=True)
                except Exception:
                    logger.exception(
                        "SessionRecorder abort temp cleanup failed: %s", path
                    )

    async def aclose(self) -> None:
        """Force-finalize any in-flight session (graceful shutdown)."""
        if self.state is not SessionState.IDLE:
            await self._finalize()
