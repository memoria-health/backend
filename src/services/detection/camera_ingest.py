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
from src.db.patient import Patient
from src.services import clip_archival, push
from src.services.detection.analyses_store import add_analysis
from src.services.detection.audio_buffer import AudioBuffer
from src.services.detection.event_promotion import maybe_promote_analysis
from src.services.detection.frame_buffer import FrameBuffer
from src.services.detection.motion_cache import MotionCache
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.providers.openai import openai_provider
from src.services.detection.providers.types import ProviderResult
from src.services.detection.registry import get_enabled_detectors_for_camera
from src.services.detection.state_machine import apply_tick_update
from src.services.reid.inference import ReIDInference

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
        reid: ReIDInference,
    ) -> None:
        self.camera_id = camera_id
        self.patient_id = patient_id
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._settings = settings
        self._reid = reid
        self.frame_buffer = FrameBuffer(
            window_seconds=settings.livekit_worker_window_seconds,
            target_fps=settings.livekit_worker_target_fps,
        )
        self.audio_buffer = AudioBuffer(
            window_seconds=settings.livekit_worker_window_seconds,
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
        # Fire-and-forget: every `tick_seconds`, kick off a tick as its
        # own task and immediately go back to sleeping. The previous
        # tick's Gemini call may still be in flight — that's intentional,
        # so cadence stays at exactly `tick_seconds` regardless of LLM
        # latency. Combined with the 15s window setting this means
        # consecutive ticks overlap by `window - tick_seconds` (default
        # 5s of overlap).
        #
        # Trade-off: the old serializing lock prevented overlapping
        # state-machine writes. With concurrent ticks, two `apply_tick_update`
        # calls can race on `cameras.state` / `no_people_count`. For testing
        # this is acceptable; the analyses uniq-by-(camera, captured_at)
        # still prevents duplicate rows since captured_at is microsecond-precise.
        tick_seconds = self._settings.livekit_worker_tick_seconds
        try:
            while True:
                await asyncio.sleep(tick_seconds)
                asyncio.create_task(
                    self._run_tick_safely(),
                    name=f"tick-{self.camera_id}",
                )
        except asyncio.CancelledError:
            raise

    async def _run_tick_safely(self) -> None:
        try:
            await self._run_tick()
        except Exception as exc:
            # Tick failures must NEVER kill the loop. Log
            # and move on; next tick will get another shot.
            log.exception("%s tick failed: %s", self._log_tag, exc)

    async def _run_tick(self) -> None:
        """One tick — mirrors `src/api/clips.py:ingest_clip` body."""
        motion_in_clip = self._motion_cache.get(self.camera_id)
        captured_at = datetime.now(timezone.utc)

        async with self._sessionmaker() as session:
            detectors = await get_enabled_detectors_for_camera(session, self.camera_id)

            # Closed-state short-circuit. Fetch the camera's CURRENT
            # state to decide whether to short-circuit the LLM. Also
            # fetch the patient so we can honor the per-patient
            # "cameras_paused" toggle (global monitoring off).
            cam = await session.get(Camera, self.camera_id)
            if cam is None:
                log.warning("%s camera row vanished mid-flight", self._log_tag)
                return
            patient = await session.get(Patient, self.patient_id)
            if patient is None:
                log.warning("%s patient row vanished mid-flight", self._log_tag)
                return

            log.info(
                "%s START detectors=%s motion=%s prior_state=%s "
                "buffered_frames=%d buffered_audio_chunks=%d "
                "disabled=%s paused=%s",
                self._log_tag,
                [d.identifier for d in detectors],
                motion_in_clip,
                cam.state,
                len(self.frame_buffer),
                len(self.audio_buffer),
                cam.disabled,
                patient.cameras_paused,
            )

            # User-facing "off" short-circuit. Either the per-camera
            # disable flag OR the patient-wide pause flag suppresses the
            # LLM call. We still write a synthetic "skipped" row so the
            # debug feed shows ticks happening (and the state machine
            # stays consistent — counters keep advancing). The LiveKit
            # connection + frame buffer stay alive in case the user
            # re-enables mid-tick.
            if cam.disabled or patient.cameras_paused:
                reason = "camera_disabled" if cam.disabled else "monitoring_paused"
                description = (
                    "(disabled — LLM skipped)"
                    if cam.disabled
                    else "(monitoring paused — LLM skipped)"
                )
                state_after = await apply_tick_update(
                    session,
                    camera_id=self.camera_id,
                    people_present=False,
                    motion_in_clip=False,
                )
                await add_analysis(
                    session,
                    camera_id=self.camera_id,
                    provider="skipped",
                    description=description,
                    detector_fields={
                        "people_present": False,
                        "skipped_reason": reason,
                    },
                    people_present=False,
                    people_count=0,
                    motion_in_clip=False,
                    state_after=state_after.state,
                    captured_at=captured_at,
                )
                await session.commit()
                log.info(
                    "%s SKIP user-off (%s) — state=%s/%s",
                    self._log_tag,
                    reason,
                    state_after.state,
                    state_after.no_people_count,
                )
                return

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

            # Build the MP4 from the buffered window (video + audio).
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
            pcm = self.audio_buffer.snapshot_last_seconds(
                self._settings.livekit_worker_window_seconds,
            )
            try:
                mp4_bytes = await frames_to_mp4(
                    window,
                    fps=self._settings.livekit_worker_target_fps,
                    pcm=pcm if pcm else None,
                    pcm_sample_rate=self.audio_buffer.sample_rate,
                    pcm_num_channels=self.audio_buffer.num_channels,
                )
            except (Mp4RemuxError, ValueError) as exc:
                log.warning("%s mp4_remux failed: %s", self._log_tag, exc)
                return

            # Re-ID identity check. Runs before the LLM call so the
            # patient_in_clip flag can flow into the prompt addendum.
            # Wrapped in try/except — Re-ID failures must NEVER break
            # the tick. Empty result = identity-agnostic (today's
            # behavior); the LLM still scores every detector.
            try:
                identity = await self._reid.label_clip(
                    mp4_bytes=mp4_bytes, patient_id=self.patient_id,
                )
            except Exception:
                log.exception("%s reid.label_clip crashed", self._log_tag)
                from src.services.reid.inference import IdentityResult
                identity = IdentityResult.empty()
            log.info(
                "%s reid patient_in_clip=%s confidence=%.3f face_count=%d",
                self._log_tag,
                identity.patient_in_clip,
                identity.confidence,
                identity.face_count,
            )

            # Provider fallback: OpenAI → Gemini. OpenAI's gpt-5.4-nano
            # is the cheaper of the two on a per-tick basis (the OpenAI
            # path also samples frames at lower fps + 480px width, vs
            # Gemini's full MP4 ingest), so we try it first and fall
            # back to Gemini only when OpenAI errors. Both providers
            # accept the same MP4 bytes — `openai_provider` extracts
            # frames internally via `extract_frames`.
            used_provider: str
            result: ProviderResult
            try:
                used_provider = "openai"
                result = await openai_provider(
                    mp4=mp4_bytes, detectors=detectors,
                    log_tag=self._log_tag, settings=self._settings,
                    patient_in_clip=identity.patient_in_clip,
                )
            except Exception as oa_err:
                log.warning("%s openai failed, trying gemini: %s", self._log_tag, oa_err)
                try:
                    used_provider = "gemini"
                    result = await gemini_provider(
                        mp4=mp4_bytes, detectors=detectors,
                        log_tag=self._log_tag, settings=self._settings,
                        patient_in_clip=identity.patient_in_clip,
                    )
                except Exception as gem_err:
                    log.error(
                        "%s both providers failed: openai=%s gemini=%s",
                        self._log_tag, oa_err, gem_err,
                    )
                    return

            # Stash identity_map in detector_fields so the analyses
            # row carries it for the dashboard / event_promotion path.
            # The dashboard can filter "patient agitation" vs anyone-
            # else's-agitation by reading identity_confidence here.
            result.detector_fields["identity_patient_in_clip"] = identity.patient_in_clip
            result.detector_fields["identity_confidence"] = round(identity.confidence, 4)

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

            # Fan APNs out after the events row is committed — a
            # rolled-back tick must not leave a phantom push trail.
            # Fire-and-forget: tick latency stays bound by Gemini, not
            # by Apple's push endpoint.
            if promotion.fired:
                fired_ids = [ev.id for ev in promotion.fired]
                asyncio.create_task(
                    push.dispatch_event_pushes(
                        sessionmaker=self._sessionmaker,
                        event_ids=fired_ids,
                        settings=self._settings,
                    ),
                    name=f"push-dispatch-{self.camera_id}",
                )
                # Clip archival: reuse the MP4 we already built for
                # the LLM as the pre-event clip; spawn one task per
                # fired event so the same camera can have two events
                # fire at the same tick (rare but possible — fall +
                # agitation) without one stomping the other's wait.
                for fired_event in promotion.fired:
                    asyncio.create_task(
                        clip_archival.archive_event(
                            event_id=fired_event.id,
                            pre_event_mp4=mp4_bytes,
                            frame_buffer=self.frame_buffer,
                            audio_buffer=self.audio_buffer,
                            sessionmaker=self._sessionmaker,
                            settings=self._settings,
                        ),
                        name=f"clip-archive-{fired_event.id}",
                    )

            log.info(
                "%s DONE provider=%s people=%s(%s) motion=%s state=%s/%s "
                "events_fired=%d dedup=%d",
                self._log_tag, used_provider, result.people_present, result.people_count,
                motion_in_clip, state_after.state, state_after.no_people_count,
                len(promotion.fired), len(promotion.skipped_for_dedup),
            )
