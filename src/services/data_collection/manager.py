"""RecorderManager — owns one SessionRecorder per actively-collecting
camera and handles the finalize → mux → B2 upload → cleanup tail.

The detection hooks talk only to this object:
  - ``on_tick(camera_id, collect, ctx=...)`` once per detection tick,
  - ``feed_video`` / ``feed_audio`` per frame.
Everything else (encoding, state, upload) is downstream of here.

Uploads are fire-and-forget tasks with bounded retry; a failed upload
never blocks detection and always cleans up its temp files. A
concurrent-session cap bounds CPU + temp disk per worker process —
scale past it by sharding the worker fleet, not by separating recording
into its own service (that would re-subscribe and double LiveKit's
bandwidth bill).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from src.services.data_collection import b2 as b2_module
from src.services.data_collection.keys import build_clip_key
from src.services.data_collection.mux import mux_av
from src.services.data_collection.session import (
    FinalizedSession,
    RecorderCtx,
    SessionRecorder,
)
from src.services.data_collection.streaming_encoder import (
    MaskingMp4Writer,
    StreamingMp4Writer,
)

logger = logging.getLogger(__name__)

# Type aliases for the injectable collaborators.
_Muxer = Callable[..., Awaitable[Path]]
_Uploader = Callable[..., Awaitable[str]]
# Called with the CAMERA id (per-camera room-derived privacy, 2026-07-27
# — the retired patient-level check took a patient id).
_PrivacyCheck = Callable[[UUID], Awaitable[bool]]
_RegionLookup = Callable[[UUID], Awaitable[str]]


def temp_root() -> Path:
    """The one directory recorder tempfiles live in.

    Kept to itself so the startup sweep below is safe: nothing else
    writes here, so whatever is present at boot is stranded.
    """
    path = Path(tempfile.gettempdir()) / "memoria-data-collection"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sweep_temp_dir() -> int:
    """Delete every recorder tempfile left over from a previous process.

    Called once at startup, before any recording begins. The normal and
    error paths both unlink their own temps, so anything sitting here at
    boot was stranded by a SIGKILL or an OOM kill mid-session.

    This matters more than ordinary tidying. Those files are RAW,
    UNCENSORED video and audio — written before any sensitivity screen —
    and without a sweep they stay on disk forever, growing by up to a
    session's worth (hundreds of MB) per crash until the disk fills.
    Event clips have had this sweep since 2026-07-29
    (`clip_archival.sweep_scratch_dir`); the recorder never got one.

    Best-effort; returns how many files were removed.
    """
    removed = 0
    try:
        root = temp_root()
        for name in os.listdir(root):
            try:
                os.unlink(os.path.join(root, name))
                removed += 1
            except OSError:
                continue
    except OSError:
        logger.warning("data_collection.temp_sweep failed", exc_info=True)
        return removed
    if removed:
        logger.warning(
            "data_collection.temp_sweep removed %d stranded recorder "
            "tempfile(s) — raw footage left by a hard-killed process",
            removed,
        )
    return removed


def _default_encoder_factory(path: Path) -> StreamingMp4Writer:
    return StreamingMp4Writer(path)


def _masking_encoder_factory(path: Path) -> MaskingMp4Writer:
    """Encoder for a session judged sensitive: identical timing, but
    every frame is masked on the way out and raw never touches disk."""
    return MaskingMp4Writer(path)


async def _never_private(_camera_id: UUID) -> bool:
    """Default privacy check — used when no DB-backed check is injected
    (e.g. unit tests). Always permits the upload."""
    return False


async def _always_us(_patient_id: UUID) -> str:
    """Default region lookup — used when no DB-backed resolver is injected
    (e.g. unit tests). Routes to the default (US) bucket."""
    return "US"


class RecorderManager:
    def __init__(
        self,
        *,
        settings: Any,
        temp_dir: Path | str,
        encoder_factory: Callable[[Path], Any] = _default_encoder_factory,
        masking_encoder_factory: Callable[[Path], Any] = _masking_encoder_factory,
        muxer: _Muxer = mux_av,
        uploader: _Uploader = b2_module.upload_file_to_b2,
        clock: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        roll_seconds: float = 900.0,
        linger_seconds: float = 45.0,
        upload_retries: int = 3,
        retry_base_delay: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        privacy_check: _PrivacyCheck = _never_private,
        region_lookup: _RegionLookup = _always_us,
    ) -> None:
        self._settings = settings
        self._masking_encoder_factory = masking_encoder_factory
        # Cameras whose current session must be masked. Set by a
        # sensitive verdict, cleared when the session ends.
        self._masked: dict[UUID, bool] = {}
        self._privacy_check = privacy_check
        self._region_lookup = region_lookup
        self._temp_dir = Path(temp_dir)
        self._encoder_factory = encoder_factory
        self._muxer = muxer
        self._uploader = uploader
        self._clock = clock
        self._utcnow = utcnow
        self._roll_seconds = roll_seconds
        self._linger_seconds = linger_seconds
        self._max_concurrent = int(getattr(settings, "data_collection_max_concurrent", 8))
        self._upload_retries = upload_retries
        self._retry_base_delay = retry_base_delay
        self._sleep = sleep

        # Running tally of what the recorder did. Without this, collection
        # can stop completely — cap reached, B2 unconfigured, every upload
        # failing — and nothing anywhere says so. Read it via `stats`.
        self.stats: dict[str, int] = {
            "sessions_started": 0,
            "sessions_masked": 0,
            "clips_uploaded": 0,
            "dropped_cap": 0,
            "dropped_privacy": 0,
            "dropped_encoder": 0,
            "dropped_mux": 0,
            "dropped_no_b2": 0,
            "dropped_upload_failed": 0,
            "clips_short": 0,
        }

        self._sessions: dict[UUID, SessionRecorder] = {}
        self._pending_uploads: set[asyncio.Task] = set()
        # Detection ticks for one camera can OVERLAP (the tick loop is
        # fire-and-forget and the LLM call outlasts the tick interval),
        # so on_tick must be serialized per camera or two overlapping
        # ticks could both try to start/transition the same session.
        self._tick_locks: dict[UUID, asyncio.Lock] = {}

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._sessions.values() if r.is_active)

    # ── tick + frame entry points (called from the detection hooks) ──

    def _lock_for(self, camera_id: UUID) -> asyncio.Lock:
        lock = self._tick_locks.get(camera_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tick_locks[camera_id] = lock
        return lock

    async def on_tick(
        self,
        camera_id: UUID,
        collect: bool,
        *,
        ctx: RecorderCtx,
        lead_in_frames: Sequence[Any] | None = None,
        lead_in_audio: bytes | None = None,
        sensitive: bool | None = None,
    ) -> None:
        """``sensitive`` is the tick's verdict on whether this moment
        needs masking, or None when the tick didn't ask (it runs on a
        cadence). A True verdict is the ONLY thing that turns masking on.
        """
        async with self._lock_for(camera_id):
            if sensitive:
                await self._go_masked(camera_id)
            rec = self._sessions.get(camera_id)
            if rec is None:
                if not collect:
                    return  # idle camera, nothing to record
                if self.active_count >= self._max_concurrent:
                    logger.warning(
                        "data_collection: session cap %d reached; "
                        "not recording camera %s",
                        self._max_concurrent,
                        camera_id,
                    )
                    self.stats["dropped_cap"] += 1
                    return
                rec = self._make_recorder(
                    camera_id, ctx, masked=self._masked.get(camera_id, False)
                )
                self._sessions[camera_id] = rec
                self.stats["sessions_started"] += 1
                if self._masked.get(camera_id):
                    self.stats["sessions_masked"] += 1

            await rec.on_tick(
                collect, lead_in_frames=lead_in_frames, lead_in_audio=lead_in_audio
            )
            # Drop fully-idle recorders so the dict (and the cap math)
            # stays tight — a session that just finalized is back to IDLE.
            if not rec.is_active:
                self._sessions.pop(camera_id, None)
                # Session over. The next one starts unmasked again — a
                # verdict applies to the moment it judged, not forever.
                self._masked.pop(camera_id, None)

    async def _go_masked(self, camera_id: UUID) -> None:
        """Switch this camera to masked recording, discarding what the
        current session has captured so far.

        A verdict describes footage that is already on disk — the tick
        judges a window that has passed. Masking forward would leave that
        prefix sitting there unmasked, so it is thrown away instead. We
        lose the seconds before the verdict; nothing sensitive survives
        unmasked. Discarding is also far cheaper than going back and
        masking it.

        Sticky for the rest of the session: once a moment has been judged
        sensitive, the session stays masked rather than flapping back and
        forth on each verdict, which would chop it into fragments.

        Caller holds the camera's lock.
        """
        if self._masked.get(camera_id):
            return  # already masked, nothing to switch
        self._masked[camera_id] = True
        rec = self._sessions.pop(camera_id, None)
        if rec is None or not rec.is_active:
            return  # nothing captured yet; the next session starts masked
        logger.info(
            "data_collection: sensitive verdict for camera %s — discarding "
            "the unmasked session and restarting masked", camera_id,
        )
        try:
            await rec.abort()
        except Exception:
            logger.exception(
                "data_collection: abort while going masked failed for %s",
                camera_id,
            )

    async def finalize_camera(self, camera_id: UUID) -> None:
        """Force-finalize a single camera's in-flight session (called when
        the camera is removed / vanishes mid-session). The clip's upload
        still proceeds via the normal fire-and-forget path."""
        async with self._lock_for(camera_id):
            rec = self._sessions.pop(camera_id, None)
        if rec is not None and rec.is_active:
            try:
                await rec.aclose()
            except Exception:
                logger.exception(
                    "data_collection: finalize_camera failed for %s", camera_id
                )

    async def abort_camera(self, camera_id: UUID) -> None:
        """Force-ABORT a camera's in-flight session, discarding temp files
        WITHOUT muxing or uploading.

        This is the privacy path: when the camera's ``privacy_enabled``
        turns on mid-session (room classifier finalizing, support
        override), the per-tick drive calls this instead of the normal
        gate-off route so the already-captured footage never reaches B2.
        No-op if the camera has no active session.
        """
        async with self._lock_for(camera_id):
            rec = self._sessions.pop(camera_id, None)
        if rec is not None and rec.is_active:
            try:
                await rec.abort()
            except Exception:
                logger.exception(
                    "data_collection: abort_camera failed for %s", camera_id
                )

    def wants_video(self, camera_id: UUID) -> bool:
        """Cheap probe: does this camera have a session that would
        consume a fed frame right now? Lets the receive path skip the
        native-res convert entirely when neither the recorder nor the
        frame buffer needs the frame (JPEG-in-buffer rework 2026-07-19)."""
        return self._sessions.get(camera_id) is not None

    def feed_video(self, camera_id: UUID, width: int, height: int, data: bytes) -> None:
        rec = self._sessions.get(camera_id)
        if rec is not None:
            rec.feed_video(width, height, data)

    def feed_audio(
        self, camera_id: UUID, pcm: bytes, captured_at: float | None = None
    ) -> None:
        """Route a chunk of captured audio to the camera's session.

        ``captured_at`` is when the chunk arrived, on the same monotonic
        clock as the video frames. It's what lets the session line the
        two streams up and notice a stall in the audio stream.
        """
        rec = self._sessions.get(camera_id)
        if rec is not None:
            rec.feed_audio(pcm, captured_at)

    # ── session wiring ──────────────────────────────────────────────

    def _make_recorder(
        self, camera_id: UUID, ctx: RecorderCtx, *, masked: bool = False,
    ) -> SessionRecorder:
        return SessionRecorder(
            camera_id=camera_id,
            ctx=ctx,
            temp_dir=self._temp_dir,
            encoder_factory=(
                self._masking_encoder_factory if masked
                else self._encoder_factory
            ),
            on_finalized=self._handle_finalized,
            clock=self._clock,
            utcnow=self._utcnow,
            roll_seconds=self._roll_seconds,
            linger_seconds=self._linger_seconds,
            masked=masked,
        )

    async def _handle_finalized(self, fs: FinalizedSession) -> None:
        """Mux the session, then upload it (fire-and-forget) with cleanup.

        Runs inline up to spawning the upload task so the mux is done
        before the next tick reuses temp space, but the upload itself
        never blocks the tick.
        """
        final_path = fs.video_path.with_suffix(".final.mp4")
        # A masked clip is where sound carries most of what happened, so
        # it is kept by default. This switch is ours to flip; it can only
        # DISCARD audio, never cause any to be captured (that is severed
        # at the publisher by `patients.audio_capture_enabled`).
        audio_path = fs.audio_path
        if fs.masked and not getattr(
            self._settings, "data_collection_keep_sensitive_audio", True
        ):
            logger.info(
                "data_collection: dropping audio from a masked clip "
                "(keep_sensitive_audio is off) camera=%s", fs.ctx.camera_id,
            )
            audio_path = None
        if not fs.encoder_ok:
            # The encoder child died. Its file is truncated at best, and
            # for a masked session there is no telling how much of it got
            # masked — so it never leaves the box.
            logger.error(
                "data_collection: encoder failed for camera %s; discarding "
                "the clip", fs.ctx.camera_id,
            )
            self.stats["dropped_encoder"] += 1
            self._cleanup(fs.video_path, fs.audio_path)
            return
        try:
            await self._muxer(
                fs.video_path,
                audio_path,
                final_path,
                audio_offset_seconds=fs.audio_offset_seconds,
                duration_seconds=fs.video_duration_seconds,
            )
        except Exception:
            self.stats["dropped_mux"] += 1
            logger.exception("data_collection: mux failed; dropping session")
            self._cleanup(fs.video_path, fs.audio_path, final_path)
            return

        key = build_clip_key(
            caregiver_id=fs.ctx.caregiver_id,
            patient_id=fs.ctx.patient_id,
            room=fs.ctx.room,
            session_start=fs.session_start_utc,
        )
        metadata = {
            "patient_id": str(fs.ctx.patient_id),
            "room": fs.ctx.room,
            "gate": fs.ctx.gate,
            "confidence": "" if fs.ctx.confidence is None else str(fs.ctx.confidence),
            # The clip's own time axis, carried on the object. With the
            # encoder now filling to the wall clock, a moment inside this
            # file sits at `captured_at - started_at` seconds — which is
            # what lets anything time-indexed point into the footage
            # without opening it.
            "started_at": fs.session_start_utc.isoformat(),
            "duration_seconds": (
                "" if fs.video_duration_seconds is None
                else f"{fs.video_duration_seconds:.3f}"
            ),
            # Whether this clip's timeline can be trusted. "true" means
            # duration equals the real session and an offset inside the
            # file is exactly `captured_at - started_at`. "false" means
            # the encoder fell too far behind to catch up, so the clip is
            # short and later offsets have shifted — the consumer needs to
            # know that without opening the file or reading our logs.
            "timeline_exact": "false" if fs.deficit_seconds else "true",
            "lost_seconds": f"{fs.deficit_seconds:.1f}",
            # Frames overwritten before they could be encoded. Their time
            # is still represented; the intermediate picture is not.
            # A sharpness signal for whoever trains on this.
            "dropped_frames": str(fs.dropped_frames),
        }

        task = asyncio.create_task(
            self._upload_and_cleanup(
                final_path,
                key,
                metadata,
                fs.video_path,
                fs.audio_path,
                camera_id=fs.ctx.camera_id,
                patient_id=fs.ctx.patient_id,
            )
        )
        self._pending_uploads.add(task)
        task.add_done_callback(self._pending_uploads.discard)

    async def _upload_and_cleanup(
        self,
        final_path: Path,
        key: str,
        metadata: dict[str, str],
        video_path: Path,
        audio_path: Path | None,
        *,
        camera_id: UUID,
        patient_id: UUID,
    ) -> None:
        try:
            # Defense-in-depth: the CAMERA's privacy can flip on (room
            # classifier finalizing bathroom/bedroom, support override)
            # AFTER a session finalized but BEFORE its bytes leave the
            # box. Re-check per-camera at the upload boundary and drop
            # the clip if so — nothing private ever reaches B2 even on
            # that race. Fail-safe: if the check itself errors, do NOT
            # upload.
            try:
                if await self._privacy_check(camera_id):
                    logger.info(
                        "data_collection: camera privacy on at upload; "
                        "discarding clip %s",
                        key,
                    )
                    self.stats["dropped_privacy"] += 1
                    return
            except Exception:
                logger.exception(
                    "data_collection: privacy check failed for camera %s; "
                    "discarding clip %s (fail-safe)",
                    camera_id,
                    key,
                )
                return
            if not b2_module.b2_configured(self._settings):
                logger.warning(
                    "data_collection: B2 not configured; discarding clip %s", key
                )
                self.stats["dropped_no_b2"] += 1
                return
            # Data-residency: resolve the patient's region so India clips land
            # in the India bucket. A lookup failure defaults to US (never block
            # an upload on it), matching the fail-open posture for routing.
            try:
                region = await self._region_lookup(patient_id)
            except Exception:
                logger.exception(
                    "data_collection: region lookup failed for %s; "
                    "defaulting to US bucket for clip %s",
                    patient_id, key,
                )
                region = "US"
            await self._upload_with_retry(final_path, key, metadata, region)
        finally:
            self._cleanup(video_path, audio_path, final_path)

    async def _upload_with_retry(
        self, final_path: Path, key: str, metadata: dict[str, str], region: str,
    ) -> None:
        delay = self._retry_base_delay
        for attempt in range(1, self._upload_retries + 1):
            try:
                await self._uploader(
                    final_path, key, metadata=metadata, settings=self._settings,
                    region=region,
                )
                self.stats["clips_uploaded"] += 1
                if metadata.get("timeline_exact") == "false":
                    self.stats["clips_short"] += 1
                logger.info(
                    "data_collection: uploaded %s (%s) stats=%s",
                    key, metadata.get("duration_seconds"), self.stats,
                )
                return
            except Exception:
                if attempt >= self._upload_retries:
                    logger.exception(
                        "data_collection: upload failed after %d attempts key=%s",
                        attempt,
                        key,
                    )
                    self.stats["dropped_upload_failed"] += 1
                    return
                logger.warning(
                    "data_collection: upload attempt %d failed key=%s; retrying",
                    attempt,
                    key,
                )
                await self._sleep(delay)
                delay *= 2

    def _cleanup(self, *paths: Path | None) -> None:
        for p in paths:
            if p is None:
                continue
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                logger.exception("data_collection: temp cleanup failed for %s", p)

    async def drain(self) -> None:
        """Await all in-flight uploads (tests + graceful shutdown)."""
        if self._pending_uploads:
            await asyncio.gather(*list(self._pending_uploads), return_exceptions=True)

    async def aclose(self) -> None:
        """Finalize every in-flight session, then await pending uploads.

        Best-effort on graceful shutdown; a hard crash loses in-flight
        sessions (acceptable for v1).
        """
        for camera_id, rec in list(self._sessions.items()):
            try:
                await rec.aclose()
            except Exception:
                logger.exception(
                    "data_collection: session aclose failed for camera %s", camera_id
                )
            self._sessions.pop(camera_id, None)
        await self.drain()
