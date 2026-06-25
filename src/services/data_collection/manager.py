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
from src.services.data_collection.streaming_encoder import StreamingMp4Writer

logger = logging.getLogger(__name__)

# Type aliases for the injectable collaborators.
_Muxer = Callable[..., Awaitable[Path]]
_Uploader = Callable[..., Awaitable[str]]


def _default_encoder_factory(path: Path) -> StreamingMp4Writer:
    return StreamingMp4Writer(path)


class RecorderManager:
    def __init__(
        self,
        *,
        settings: Any,
        temp_dir: Path | str,
        encoder_factory: Callable[[Path], Any] = _default_encoder_factory,
        muxer: _Muxer = mux_av,
        uploader: _Uploader = b2_module.upload_file_to_b2,
        clock: Callable[[], float] = time.monotonic,
        utcnow: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        roll_seconds: float = 900.0,
        linger_seconds: float = 45.0,
        upload_retries: int = 3,
        retry_base_delay: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
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
    ) -> None:
        async with self._lock_for(camera_id):
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
                    return
                rec = self._make_recorder(camera_id, ctx)
                self._sessions[camera_id] = rec

            await rec.on_tick(
                collect, lead_in_frames=lead_in_frames, lead_in_audio=lead_in_audio
            )
            # Drop fully-idle recorders so the dict (and the cap math)
            # stays tight — a session that just finalized is back to IDLE.
            if not rec.is_active:
                self._sessions.pop(camera_id, None)

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

        This is the privacy path: when ``privacy_mode_enabled`` turns on
        mid-session, the per-tick drive calls this instead of the normal
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

    def feed_video(self, camera_id: UUID, width: int, height: int, data: bytes) -> None:
        rec = self._sessions.get(camera_id)
        if rec is not None:
            rec.feed_video(width, height, data)

    def feed_audio(self, camera_id: UUID, pcm: bytes) -> None:
        rec = self._sessions.get(camera_id)
        if rec is not None:
            rec.feed_audio(pcm)

    # ── session wiring ──────────────────────────────────────────────

    def _make_recorder(self, camera_id: UUID, ctx: RecorderCtx) -> SessionRecorder:
        return SessionRecorder(
            camera_id=camera_id,
            ctx=ctx,
            temp_dir=self._temp_dir,
            encoder_factory=self._encoder_factory,
            on_finalized=self._handle_finalized,
            clock=self._clock,
            utcnow=self._utcnow,
            roll_seconds=self._roll_seconds,
            linger_seconds=self._linger_seconds,
        )

    async def _handle_finalized(self, fs: FinalizedSession) -> None:
        """Mux the session, then upload it (fire-and-forget) with cleanup.

        Runs inline up to spawning the upload task so the mux is done
        before the next tick reuses temp space, but the upload itself
        never blocks the tick.
        """
        final_path = fs.video_path.with_suffix(".final.mp4")
        try:
            await self._muxer(fs.video_path, fs.audio_path, final_path)
        except Exception:
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
        }

        task = asyncio.create_task(
            self._upload_and_cleanup(
                final_path, key, metadata, fs.video_path, fs.audio_path
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
    ) -> None:
        try:
            if not b2_module.b2_configured(self._settings):
                logger.warning(
                    "data_collection: B2 not configured; discarding clip %s", key
                )
                return
            await self._upload_with_retry(final_path, key, metadata)
        finally:
            self._cleanup(video_path, audio_path, final_path)

    async def _upload_with_retry(
        self, final_path: Path, key: str, metadata: dict[str, str]
    ) -> None:
        delay = self._retry_base_delay
        for attempt in range(1, self._upload_retries + 1):
            try:
                await self._uploader(
                    final_path, key, metadata=metadata, settings=self._settings
                )
                return
            except Exception:
                if attempt >= self._upload_retries:
                    logger.exception(
                        "data_collection: upload failed after %d attempts key=%s",
                        attempt,
                        key,
                    )
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
