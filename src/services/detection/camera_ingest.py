"""Per-camera detector tick — the subscriber-bot's body.

Each `CameraIngest` instance owns:
  - one `FrameBuffer` of recently subscribed video frames,
  - one `asyncio.Task` that fires every `tick_seconds`,
  - one `asyncio.Lock` that serializes ticks against in-flight ones,
  - a reference to the room-worker-scoped `MotionCache` and
    `Sessionmaker`.

The tick body **is the same orchestration `src/api/clips.py` used to
run** — it calls the exact same registry / state-machine /
analyses-store / event-promotion functions. Only two inputs differ
from the HTTP-clip path:

  1. `motion_in_clip` is read from the motion cache (populated by
     the room-worker's data-packet handler) instead of trusted from a
     form field.
  2. `mp4_bytes` comes from `frames_to_mp4` over the most recent
     window of buffered frames, instead of from a multipart upload.

Concurrency safety: the lock guarantees no two ticks for the same
camera overlap on the state-machine row's read-modify-write. The
worker is the only writer for `cameras.state` / `no_people_count`
once Phase 2 ships — the deleted HTTP route can no longer touch it.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from src.db.camera import Camera
from src.services.detection.analyses_store import add_analysis
from src.services.detection.event_promotion import maybe_promote_analysis
from src.services.detection.frame_buffer import FrameBuffer
from src.services.detection.motion_cache import MotionCache
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.providers.openai import openai_provider
from src.services.detection.providers.types import ProviderResult
from src.services.detection.registry import get_enabled_detectors_for_camera
from src.services.detection.state_machine import apply_tick_update

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)


class CameraIngest:
    """One per camera. Owns the frame buffer, the tick scheduler, and
    the per-camera lock. Lifecycle is controlled by the parent
    `RoomWorker`: `start()` spawns the tick task, `stop()` cancels
    + awaits it.
    """

    def __init__(
        self,
        *,
        camera_id: UUID,
        patient_id: UUID,
        sessionmaker: "async_sessionmaker",
        motion_cache: MotionCache,
        settings: "Settings",
    ) -> None:
        self.camera_id = camera_id
        self.patient_id = patient_id
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._settings = settings
        self.frame_buffer = FrameBuffer(
            window_seconds=settings.livekit_worker_window_seconds,
            target_fps=settings.livekit_worker_target_fps,
        )
        self._lock = asyncio.Lock()
        self._tick_task: asyncio.Task[None] | None = None
        self._log_tag = f"worker cam={camera_id}"

    async def start(self) -> None:
        if self._tick_task is not None:
            return
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        if self._tick_task is None:
            return
        self._tick_task.cancel()
        try:
            await self._tick_task
        except asyncio.CancelledError:
            pass
        self._tick_task = None
        self._motion_cache.drop(self.camera_id)

    async def _tick_loop(self) -> None:
        tick_seconds = self._settings.livekit_worker_tick_seconds
        try:
            while True:
                await asyncio.sleep(tick_seconds)
                # If a prior tick is still running (slow LLM, etc.),
                # skip this cycle rather than queueing — matches the
                # plan's "no overlap" guarantee.
                if self._lock.locked():
                    log.info("%s SKIP previous tick still in flight", self._log_tag)
                    continue
                async with self._lock:
                    try:
                        await self._run_tick()
                    except Exception as exc:
                        # Tick failures must NEVER kill the loop. Log
                        # and move on; next tick will get another shot.
                        log.exception("%s tick failed: %s", self._log_tag, exc)
        except asyncio.CancelledError:
            raise

    async def _run_tick(self) -> None:
        """One tick — mirrors `src/api/clips.py:ingest_clip` body."""
        motion_in_clip = self._motion_cache.get(self.camera_id)
        captured_at = datetime.now(timezone.utc)

        async with self._sessionmaker() as session:
            detectors = await get_enabled_detectors_for_camera(session, self.camera_id)

            # Closed-state short-circuit. Fetch the camera's CURRENT
            # state to decide whether to short-circuit the LLM.
            cam = await session.get(Camera, self.camera_id)
            if cam is None:
                log.warning("%s camera row vanished mid-flight", self._log_tag)
                return

            log.info(
                "%s START detectors=%s motion=%s prior_state=%s buffered_frames=%d",
                self._log_tag,
                [d.identifier for d in detectors],
                motion_in_clip,
                cam.state,
                len(self.frame_buffer),
            )

            if cam.state == "closed" and not motion_in_clip:
                state_after = await apply_tick_update(
                    session,
                    camera_id=self.camera_id,
                    people_present=False,
                    motion_in_clip=False,
                )
                synthetic_fields = {
                    "people_present": False,
                    "skipped_reason": "camera_closed",
                }
                await add_analysis(
                    session,
                    camera_id=self.camera_id,
                    provider="skipped",
                    description="(closed — LLM skipped)",
                    detector_fields=synthetic_fields,
                    people_present=False,
                    people_count=0,
                    motion_in_clip=False,
                    state_after=state_after.state,
                    captured_at=captured_at,
                )
                await session.commit()
                log.info(
                    "%s SKIP closed-state, no motion — counter=%s/%s",
                    self._log_tag, state_after.no_people_count, state_after.close_threshold,
                )
                return

            # Build the MP4 from the buffered window.
            window = self.frame_buffer.snapshot_last_seconds(
                self._settings.livekit_worker_window_seconds,
            )
            if not window:
                log.info(
                    "%s SKIP no frames in window (worker just joined, "
                    "or publisher hasn't sent any frames yet)",
                    self._log_tag,
                )
                return
            try:
                mp4_bytes = await frames_to_mp4(
                    window,
                    fps=self._settings.livekit_worker_target_fps,
                )
            except (Mp4RemuxError, ValueError) as exc:
                log.warning("%s mp4_remux failed: %s", self._log_tag, exc)
                return

            # Provider fallback: Gemini → OpenAI.
            used_provider: str
            result: ProviderResult
            try:
                used_provider = "gemini"
                result = await gemini_provider(
                    mp4=mp4_bytes, detectors=detectors,
                    log_tag=self._log_tag, settings=self._settings,
                )
            except Exception as gem_err:
                log.warning("%s gemini failed, trying openai: %s", self._log_tag, gem_err)
                try:
                    used_provider = "openai"
                    result = await openai_provider(
                        mp4=mp4_bytes, detectors=detectors,
                        log_tag=self._log_tag, settings=self._settings,
                    )
                except Exception as oa_err:
                    log.error(
                        "%s both providers failed: gemini=%s openai=%s",
                        self._log_tag, gem_err, oa_err,
                    )
                    return

            # State machine + analyses + event promotion. Same session,
            # one commit at the end.
            state_after = await apply_tick_update(
                session,
                camera_id=self.camera_id,
                people_present=result.people_present,
                motion_in_clip=motion_in_clip,
            )
            await add_analysis(
                session,
                camera_id=self.camera_id,
                provider=used_provider,
                description=result.description,
                detector_fields=result.detector_fields,
                people_present=result.people_present,
                people_count=result.people_count,
                motion_in_clip=motion_in_clip,
                state_after=state_after.state,
                captured_at=captured_at,
            )
            promotion = await maybe_promote_analysis(
                session,
                patient_id=self.patient_id,
                camera_id=self.camera_id,
                occurred_at=captured_at,
                description=result.description,
                detector_fields=result.detector_fields,
            )
            await session.commit()

            log.info(
                "%s DONE provider=%s people=%s(%s) motion=%s state=%s/%s "
                "events_fired=%d dedup=%d",
                self._log_tag, used_provider, result.people_present, result.people_count,
                motion_in_clip, state_after.state, state_after.no_people_count,
                len(promotion.fired), len(promotion.skipped_for_dedup),
            )
