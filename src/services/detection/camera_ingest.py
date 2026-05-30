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
from src.services import clip_archival, email, push
from src.services.detection.analyses_store import add_analysis
from src.services.detection.audio_buffer import AudioBuffer
from src.services.detection.event_promotion import maybe_promote_analysis
from src.services.detection.frame_buffer import FrameBuffer
from src.services.detection.motion_cache import MotionCache
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.providers.openai import openai_provider
from src.services.detection.providers.types import ProviderResult
from src.services.detection.registry import (
    CameraVanishedError,
    get_enabled_detectors_for_camera,
)
from src.services.detection.state_machine import apply_tick_update
from src.repos.face_embedding_repo import has_ready_enrollment
from src.services.reid.inference import IdentityResult, ReIDInference

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)

# Bound on how long stop() will wait for in-flight tick bodies to
# finish before force-cancelling. Re-id-gated ticks (fast path) finish
# in ~3s; Gemini-running ticks take up to ~28s and we'd rather
# force-cancel them than block a shutdown — those would otherwise
# write an analysis row to a now-deleted camera (FK violation,
# silently swallowed).
_STOP_INFLIGHT_TIMEOUT: float = 2.0


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
        livekit_identity: str | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.patient_id = patient_id
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._settings = settings
        self._reid = reid
        # Retained for log identification; the actual reaper for a
        # vanished camera lives in `LiveKitRoomReconciler`.
        self._livekit_identity = livekit_identity
        self.frame_buffer = FrameBuffer(
            window_seconds=settings.livekit_worker_window_seconds,
            target_fps=settings.livekit_worker_target_fps,
        )
        self.audio_buffer = AudioBuffer(
            window_seconds=settings.livekit_worker_window_seconds,
        )
        self._lock = asyncio.Lock()
        self._tick_task: asyncio.Task[None] | None = None
        # Monotonic timestamp of the most recent frame pushed into
        # `frame_buffer` (set by RoomWorker._consume_video →
        # record_frame_arrival). Read by `LiveKitRoomReconciler` to
        # decide the second half of the two-signal online rule
        # (REST-presence AND fresh frames). Single-writer single-reader
        # within the asyncio loop — no lock required.
        self._last_frame_at: float = 0.0
        # In-flight tick bodies spawned by _tick_loop. stop() awaits these
        # with a 2s bound so the most recent tick(s) have a chance to
        # finish cleanly before we force-cancel — otherwise their
        # analyses writes would FK-violate after the row CASCADE-deletes.
        self._inflight_ticks: set[asyncio.Task[None]] = set()
        self._log_tag = f"worker cam={camera_id}"

    async def start(self) -> None:
        if self._tick_task is not None:
            return
        self._tick_task = asyncio.create_task(self._tick_loop())

    def record_frame_arrival(self, monotonic_ts: float) -> None:
        """Sync callback the RoomWorker's video consumer fires every
        time it pushes a frame to our buffer. Just stamps a monotonic
        timestamp; the actual `cameras.online` / `last_seen_at` writes
        live in `LiveKitRoomReconciler`, which reads this field on each
        of its periodic REST polls (every ~10s) and combines it with
        the REST-presence signal to decide both columns.
        """
        self._last_frame_at = monotonic_ts

    async def stop(self) -> None:
        # Cancel the loop FIRST so no new ticks get spawned while we wait
        # on the in-flight ones. Idempotent — second stop() is a no-op
        # because _tick_task is None after the first call.
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        # Drain in-flight tick bodies regardless of loop state. This
        # block runs even when stop() is called without a prior start()
        # (some tests do this) so any tasks added directly to the set
        # are still cleaned up.
        #
        # Re-id-gated ticks (~3s) usually finish before timeout;
        # Gemini-running ticks (up to ~28s) get force-cancelled — which
        # is exactly right, since their analysis writes would FK-violate
        # against a row that's about to be CASCADE-deleted.
        if self._inflight_ticks:
            # Snapshot to avoid mutation during iteration (done-callbacks
            # remove tasks from the set as they complete).
            in_flight = list(self._inflight_ticks)
            try:
                await asyncio.wait_for(
                    asyncio.gather(*in_flight, return_exceptions=True),
                    timeout=_STOP_INFLIGHT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                log.warning(
                    "%s stop: %d in-flight tick(s) still running after %.1fs — "
                    "force-cancelling",
                    self._log_tag, sum(1 for t in in_flight if not t.done()),
                    _STOP_INFLIGHT_TIMEOUT,
                )
                for task in in_flight:
                    if not task.done():
                        task.cancel()
                # Drain the cancellations so we don't leak warnings about
                # un-awaited tasks at process shutdown.
                await asyncio.gather(*in_flight, return_exceptions=True)
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
                task = asyncio.create_task(
                    self._run_tick_safely(),
                    name=f"tick-{self.camera_id}",
                )
                # Track so stop() can wait on or cancel in-flight ticks.
                # The done callback runs synchronously when the task
                # finishes (or is cancelled) and discards the reference,
                # so the set stays bounded.
                self._inflight_ticks.add(task)
                task.add_done_callback(self._inflight_ticks.discard)
        except asyncio.CancelledError:
            raise

    async def _run_tick_safely(self) -> None:
        """Wrap one tick body so that failures NEVER kill the loop.

        Vanished-row ticks (the camera row was deleted mid-flight) used
        to count toward a self-destruct threshold; that mechanism was
        removed when `LiveKitRoomReconciler` became responsible for
        spawning and tearing down ingests based on REST presence.
        The reconciler will stop calling our publisher's tracks (or
        observe LiveKit Cloud no longer holding the participant) and
        will call `RoomWorker.remove_ingest` on the next debounced
        miss. Until then, vanished-row ticks just log and return.
        """
        try:
            await self._run_tick()
        except CameraVanishedError:
            log.warning(
                "%s tick failed: camera row missing — reconciler will reap",
                self._log_tag,
            )
            return
        except Exception as exc:
            # Tick failures must NEVER kill the loop. Log
            # and move on; next tick will get another shot.
            log.exception("%s tick failed: %s", self._log_tag, exc)
            return

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
                # Raise so _run_tick_safely's vanished-row counter ticks.
                # Routes the same self-destruct path as the registry-
                # level miss above — single source of truth for "row is
                # gone, eventually reap me".
                raise CameraVanishedError(
                    f"[camera_ingest] camera {self.camera_id} vanished mid-tick"
                )
            patient = await session.get(Patient, self.patient_id)
            if patient is None:
                log.warning("%s patient row vanished mid-flight", self._log_tag)
                return

            log.info(
                "%s START detectors=%s motion=%s prior_state=%s "
                "buffered_frames=%d buffered_audio_chunks=%d "
                "disabled=%s ai_paused=%s paused=%s",
                self._log_tag,
                [d.identifier for d in detectors],
                motion_in_clip,
                cam.state,
                len(self.frame_buffer),
                len(self.audio_buffer),
                cam.disabled,
                cam.ai_paused,
                patient.cameras_paused,
            )

            # User-facing "off" short-circuit. Any of the three flags
            # suppresses the LLM call:
            #   - cam.disabled         → user hid + paused via Hide button
            #   - cam.ai_paused        → user paused AI only (video still flows)
            #   - patient.cameras_paused → patient-wide monitoring off
            # We still write a synthetic "skipped" row so the debug feed
            # shows ticks happening (and the state machine stays
            # consistent — counters keep advancing). The LiveKit
            # connection + frame buffer stay alive in all three cases so
            # re-enabling is instant.
            if cam.disabled or cam.ai_paused or patient.cameras_paused:
                if cam.disabled:
                    reason = "camera_disabled"
                    description = "(disabled — LLM skipped)"
                elif cam.ai_paused:
                    reason = "ai_paused"
                    description = "(AI paused — LLM skipped)"
                else:
                    reason = "monitoring_paused"
                    description = "(monitoring paused — LLM skipped)"
                # DO NOT advance the state machine here. We have no real
                # observation — we're explicitly skipping the LLM. The
                # previous version called apply_tick_update with
                # (people_present=False, motion_in_clip=False), which is
                # a fabrication, not a measurement. Over ~6 paused ticks
                # (~90s) that drove the state machine to "closed," and
                # the very next tick after resume would short-circuit
                # again on the closed-state check ("camera_closed") —
                # the user observed this as "I clicked resume but it's
                # still skipping." camera_repo.update_partial also
                # resets state to "open" on resume as a belt-and-
                # suspenders against any drift that snuck in.
                # Liveness write (online=true, last_seen_at) lives in
                # `_liveness_writer_loop` below — driven by actual
                # frame arrivals, not by tick scheduling. Calling
                # `apply_liveness_only` from here was a footgun: ticks
                # fire on a fixed cadence regardless of whether frames
                # are arriving, so an ai-paused camera with a long-
                # dead publisher would keep its heartbeat fresh forever
                # because the timer kept ticking and the SKIP branch
                # kept writing. Observed 2026-05-28 in Test 2 — Pi
                # service stopped but online stayed true past 55s.
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
                    state_after=cam.state,
                    captured_at=captured_at,
                )
                await session.commit()
                log.info(
                    "%s SKIP user-off (%s) — state preserved at %s/%s",
                    self._log_tag,
                    reason,
                    cam.state,
                    cam.no_people_count,
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
            # Wrapped in try/except, Re-ID failures must NEVER break
            # the tick. `IdentityResult.empty()` has reid_ran=False
            # which makes event_promotion bypass the identity gate
            # entirely (identity-agnostic / today's behavior).
            try:
                identity = await self._reid.label_clip(
                    mp4_bytes=mp4_bytes, patient_id=self.patient_id,
                )
            except Exception:
                log.exception("%s reid.label_clip crashed", self._log_tag)
                identity = IdentityResult.empty()
            log.info(
                "%s reid reid_ran=%s patient_in_clip=%s confidence=%.3f face_count=%d",
                self._log_tag,
                identity.reid_ran,
                identity.patient_in_clip,
                identity.confidence,
                identity.face_count,
            )

            # Upstream identity gate. When Re-ID has POSITIVE evidence
            # that the patient is NOT in this clip (faces detected,
            # none matched the gallery) AND an enrollment exists, skip
            # the LLM call entirely. Saves a per-tick provider call
            # (5-7s + cost) plus the archival path.
            #
            # Crucially, the gate only fires when `face_count > 0`.
            # `face_count == 0` means Re-ID saw no faces, which is NOT
            # evidence that the patient is absent. The patient may be
            # in frame with their face turned away, occluded, foreshort-
            # ened from a high-mounted camera, or motion-blurred during
            # a fall. We fall back to identity-agnostic (run the LLM)
            # in that case, same as we do for `reid_ran=False`.
            #
            # Safety contract preserved on every axis:
            #   reid_ran=False (timeout/crash/no-enrollment) -> bypass
            #   face_count=0 (no faces seen) -> bypass
            #   face_count>0, patient_in_clip=False -> suppress
            #   patient_in_clip=True -> run with patient marked
            #
            # The `has_ready_enrollment` check is the day-zero clause.
            # Without it, a patient who hasn't enrolled would have
            # every tick skipped because Re-ID always says "not in
            # clip" against an empty gallery.
            # The identity gate now considers TWO independent positive
            # signals for "everyone in frame is verified not-patient":
            #
            #   1. `face_count > 0`: at least one face was seen and
            #      embedded this clip; the face matcher didn't lock any
            #      track as PATIENT. Today's signal.
            #
            #   2. `has_only_verified_other_tracks`: every body track in
            #      the clip ended labeled OTHER (face_seen_attempts >=
            #      face_checks_for_other AND face_hits = 0). Catches the
            #      case where the helper's face is BRIEFLY visible at
            #      various points within the clip but `face_count` could
            #      be 0 if the brief moments didn't accumulate. The
            #      OTHER label is now epistemically honest (Phase 1
            #      semantic): only fires after enough verified match
            #      failures, never on "face turned away whole clip."
            #
            # Either signal alone is sufficient to short-circuit. The
            # `has_ready_enrollment` guard remains the day-zero clause —
            # without an enrollment we can't tell "non-patient" from
            # anything else.
            if (
                identity.reid_ran
                and not identity.patient_in_clip
                and (identity.face_count > 0 or identity.has_only_verified_other_tracks)
            ):
                if await has_ready_enrollment(session, self.patient_id):
                    state_after = await apply_tick_update(
                        session,
                        camera_id=self.camera_id,
                        # We saw bodies (and at least some face evidence)
                        # but no match. State machine treats this as
                        # "people present" so a household-member walking
                        # by doesn't trip the closed-state countdown.
                        people_present=True,
                        motion_in_clip=motion_in_clip,
                    )
                    synthetic_fields = {
                        "people_present": True,
                        "skipped_reason": "identity_not_patient",
                        "identity_reid_ran": True,
                        "identity_patient_in_clip": False,
                        "identity_confidence": round(identity.confidence, 4),
                        "identity_face_count": identity.face_count,
                        "identity_has_only_verified_other_tracks":
                            identity.has_only_verified_other_tracks,
                    }
                    await add_analysis(
                        session,
                        camera_id=self.camera_id,
                        provider="skipped",
                        description="(non-patient in frame, LLM skipped)",
                        detector_fields=synthetic_fields,
                        people_present=True,
                        people_count=max(
                            identity.face_count, identity.people_count_reid,
                        ),
                        motion_in_clip=motion_in_clip,
                        state_after=state_after.state,
                        captured_at=captured_at,
                    )
                    await session.commit()
                    log.info(
                        "%s SKIP identity-gated, faces=%d only_other=%s confidence=%.3f",
                        self._log_tag,
                        identity.face_count,
                        identity.has_only_verified_other_tracks,
                        identity.confidence,
                    )
                    return

            # Provider fallback: Gemini -> OpenAI. Gemini-3.5-flash
            # processes the MP4 as actual video (native temporal
            # reasoning + higher per-frame fidelity) which is
            # materially better at the safety-critical detectors:
            # wandering (did the body cross a doorway through a door
            # swing?) and fall (sudden downward motion at a specific
            # instant). gpt-5.4-nano on 75 low-detail jpegs handles
            # state-style detectors (agitation, medication) fine but
            # was producing false negatives on threshold-crossing
            # events. Cost is ~2-3x higher per tick on Gemini, paid
            # back by not missing real safety incidents. Keep OpenAI
            # in the fallback slot for when Gemini errors or rate-
            # limits. Both providers accept the same MP4 bytes;
            # `openai_provider` extracts frames internally via
            # `extract_frames`.
            used_provider: str
            result: ProviderResult
            try:
                used_provider = "gemini"
                result = await gemini_provider(
                    mp4=mp4_bytes, detectors=detectors,
                    log_tag=self._log_tag, settings=self._settings,
                    patient_in_clip=identity.patient_in_clip,
                    patient_bbox=identity.patient_bbox,
                )
            except Exception as gem_err:
                log.warning("%s gemini failed, trying openai: %s", self._log_tag, gem_err)
                try:
                    used_provider = "openai"
                    result = await openai_provider(
                        mp4=mp4_bytes, detectors=detectors,
                        log_tag=self._log_tag, settings=self._settings,
                        patient_in_clip=identity.patient_in_clip,
                        patient_bbox=identity.patient_bbox,
                        pre_extracted_frames=identity.extracted_frames,
                        per_frame_bboxes=identity.per_frame_bboxes,
                        sample_fps=identity.sample_fps,
                    )
                except Exception as oa_err:
                    log.error(
                        "%s both providers failed: gemini=%s openai=%s",
                        self._log_tag, gem_err, oa_err,
                    )
                    return

            # Stash identity in detector_fields so the analyses row
            # carries it for the dashboard / event_promotion path.
            # The `identity_reid_ran` flag is the gate signal that
            # tells event_promotion whether to apply identity
            # suppression at all (False = bypass, identity-agnostic).
            result.detector_fields["identity_reid_ran"] = identity.reid_ran
            result.detector_fields["identity_patient_in_clip"] = identity.patient_in_clip
            result.detector_fields["identity_confidence"] = round(identity.confidence, 4)
            result.detector_fields["identity_face_count"] = identity.face_count

            # Presence + count are now Re-ID-derived (YOLO + BoT-SORT
            # body tracking), not LLM-derived. The prompt no longer
            # asks the LLM for `people_present`/`people_count`; those
            # fields on `result` default to (False, 0) and we ignore
            # them whenever Re-ID returned a usable verdict.
            #
            # `reid_ran=True` path covers both the standard
            # (gallery-present) case and the no-enrollment case where
            # `body_pipeline` runs body tracking without face matching.
            # `reid_ran=False` only happens on real infra failures
            # (re-id disabled in settings, gallery load crashed,
            # frame extract failed, pipeline timeout). On that path we
            # conservatively keep the camera open (people_present=True,
            # people_count=0) so a Re-ID outage doesn't silently close
            # the state and stop the LLM-driven behavioral detectors.
            if identity.reid_ran:
                gate_people_present = identity.people_present_reid
                gate_people_count = identity.people_count_reid
                present_src = "reid"
            else:
                gate_people_present = True
                gate_people_count = 0
                present_src = "reid_failed_fallback"
            result.detector_fields["people_present_src"] = present_src
            result.detector_fields["people_count_reid"] = identity.people_count_reid

            # State machine + analyses + event promotion. Same session,
            # one commit at the end.
            state_after = await apply_tick_update(
                session,
                camera_id=self.camera_id,
                people_present=gate_people_present,
                motion_in_clip=motion_in_clip,
            )
            await add_analysis(
                session,
                camera_id=self.camera_id,
                provider=used_provider,
                description=result.description,
                detector_fields=result.detector_fields,
                people_present=gate_people_present,
                people_count=gate_people_count,
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
                # Same fire-and-forget fan-out for email alerts. No-ops
                # if Resend isn't configured or the caregiver hasn't
                # enabled email; independent of push so one failing
                # doesn't affect the other.
                asyncio.create_task(
                    email.dispatch_event_emails(
                        sessionmaker=self._sessionmaker,
                        event_ids=fired_ids,
                        settings=self._settings,
                    ),
                    name=f"email-dispatch-{self.camera_id}",
                )
                # Clip archival: reuse the MP4 we already built for
                # the LLM as the pre-event clip; spawn one task per
                # fired event so the same camera can have two events
                # fire at the same tick (rare but possible — fall +
                # agitation) without one stomping the other's wait.
                # The last frame in the pre snapshot anchors the post
                # snapshot's start. Without this, the archive task's
                # "wait then take last 15s" leaves a 5-10s hole equal
                # to the LLM think time between pre's end and post's
                # start.
                pre_end_captured_at = window[-1].captured_at
                for fired_event in promotion.fired:
                    asyncio.create_task(
                        clip_archival.archive_event(
                            event_id=fired_event.id,
                            patient_id=self.patient_id,
                            pre_event_mp4=mp4_bytes,
                            pre_end_captured_at=pre_end_captured_at,
                            frame_buffer=self.frame_buffer,
                            audio_buffer=self.audio_buffer,
                            sessionmaker=self._sessionmaker,
                            settings=self._settings,
                        ),
                        name=f"clip-archive-{fired_event.id}",
                    )

            log.info(
                "%s DONE provider=%s people=%s(%s) src=%s motion=%s "
                "state=%s/%s events_fired=%d dedup=%d",
                self._log_tag, used_provider, gate_people_present, gate_people_count,
                present_src, motion_in_clip, state_after.state, state_after.no_people_count,
                len(promotion.fired), len(promotion.skipped_for_dedup),
            )
