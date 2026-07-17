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
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.patient import Patient
from src.services import clip_archival, email, push
from src.services.detection.analyses_store import add_analysis
from src.services.detection.audio_buffer import AudioBuffer
from src.services.detection.event_promotion import maybe_promote_analysis
from src.services.detection.frame_buffer import FrameBuffer
from src.services.detection.id_evidence import dump_id_evidence
from src.services.detection.ingress_meter import IngressMeter
from src.services.detection.motion import (
    BlockMotion,
    BlockMotionTracker,
    sample_stats,
)
from src.services.detection.motion_cache import MotionCache
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.providers.types import ProviderResult
# v2 scheduler (P3) — the new architecture lives in scheduling/ (D1);
# these imports + the flag branch in the tick body are the only
# sanctioned coupling between the two.
from src.services.detection.scheduling import (
    ScheduledDetector,
    pack_calls,
    resolve_dims,
    resolve_due,
)
from src.services.detection.scheduling.executor import run_scheduled_calls
from src.services.detection.signals.tick_signals import TickSignals, compute_tick_signals
from src.services.detection.speech_filter import sanitize_speech_fields
from src.services.detection.vlm_routing import resolve_vlm_provider, run_vlm
from src.services.detection.registry import (
    CameraVanishedError,
    get_enabled_detectors_for_camera,
    get_enabled_loadout_for_camera,
)
from src.services.detection.state_machine import apply_tick_update
from src.repos.face_embedding_repo import has_ready_enrollment
from src.services.reid.inference import IdentityResult, ReIDInference
from src.services.data_collection.gate import should_collect
from src.services.data_collection.session import RecorderCtx

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings
    from src.services.data_collection.manager import RecorderManager

log = logging.getLogger(__name__)

# Bound on how long stop() will wait for in-flight tick bodies to
# finish before force-cancelling. Re-id-gated ticks (fast path) finish
# in ~3s; Gemini-running ticks take up to ~28s and we'd rather
# force-cancel them than block a shutdown — those would otherwise
# write an analysis row to a now-deleted camera (FK violation,
# silently swallowed).
_STOP_INFLIGHT_TIMEOUT: float = 2.0


@dataclass
class _RecorderDecision:
    """Per-tick verdict handed to the data-collection recorder.

    Created fresh each tick (ticks can overlap, so this must NOT be
    shared instance state). Defaults to "don't collect" so an early-
    return tick still drives any open session toward finalize; the
    fields are filled in as the tick learns presence + identity.
    """

    collect: bool = False
    ctx: "RecorderCtx | None" = None
    lead_frames: list | None = None
    lead_audio: bytes | None = None
    # Privacy mode for this patient (read off the patient row once it
    # loads). When True the drive ABORTS any in-flight session instead of
    # taking the linger→finalize→upload route, so pre-privacy footage is
    # dropped on the device rather than uploaded.
    privacy: bool = False


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
        recorder: "RecorderManager | None" = None,
    ) -> None:
        self.camera_id = camera_id
        self.patient_id = patient_id
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._settings = settings
        self._reid = reid
        # Backend wake authority (motion.py #3): verifies the gateway's
        # motion hint before a closed camera buys a VLM call. Owns the
        # per-block habituation baselines for this camera.
        self._block_motion = BlockMotionTracker()
        # Data-collection recorder (owned by the RoomWorker, shared
        # across the room's cameras). None when the feature is off /
        # not wired — then every recorder hook is a no-op.
        self._recorder = recorder
        # Retained for log identification; the actual reaper for a
        # vanished camera lives in `LiveKitRoomReconciler`.
        self._livekit_identity = livekit_identity
        # Caregiver-local timezone for when_scope windows (2026-07-14).
        # Resolved once on first use and cached for the worker's life —
        # a caregiver timezone change picks up on the next worker
        # restart, which matches the scheduled runner's tolerance.
        self._caregiver_tz: "ZoneInfo | None" = None
        # Previous tick's identity signature (n_tracks, labels, face
        # locks) — drives change-triggered evidence capture. None until
        # the first tick that runs tracking.
        self._last_id_sig: tuple | None = None
        # Retain enough history for the LONGEST consumer — the LLM
        # analysis window OR the incident-clip post-roll, whichever is
        # bigger — derived from clip length so the two can never drift.
        # The tick still snapshots only `livekit_worker_window_seconds`
        # for the LLM (cost unchanged); the extra retention exists purely
        # so the post-roll's post.mp4 stays contiguous with pre.mp4 and
        # the concat seam has no gap. (Before this, the buffer was tied to
        # the 11s analysis window while the post-roll asked for 15s, so the
        # post snapshot lost its oldest ~4s → a visible jump at ~11s.)
        retention_seconds = settings.frame_buffer_retention_seconds
        # RAPID detectors need native-rate frames; the tick retargets the
        # buffer to `rapid_fps` only while the camera's loadout carries a
        # RAPID-bucket detector (Ledger row 5 — no flag). Construction
        # starts at the plain target rate; every other consumer decimates
        # via `snapshot_last_seconds(at_fps=target_fps)` either way, so a
        # retarget never changes what they see.
        self.frame_buffer = FrameBuffer(
            window_seconds=retention_seconds,
            target_fps=settings.livekit_worker_target_fps,
        )
        # Fed by RoomWorker._consume_video for every SDK-delivered frame
        # (accepted or not); each tick stamps its snapshot on the row so
        # delivered-vs-published fps is auditable per tick (soak #1
        # finding 3: 4–6 fps delivered against a CFR-15 publish).
        self.ingress_meter = IngressMeter()
        self.audio_buffer = AudioBuffer(
            window_seconds=retention_seconds,
        )
        self._lock = asyncio.Lock()
        self._tick_task: asyncio.Task[None] | None = None
        # Monotonic per-camera tick counter — drives the scheduler's
        # sampled duty cycle (v2 P3). Single-writer within the loop.
        self._tick_counter = 0
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
        timestamp; the actual `cameras.online` write lives in
        `LiveKitRoomReconciler`, which reads this field on each of its
        periodic REST polls (every ~10s) and combines it with the
        REST-presence signal to decide the column.
        """
        self._last_frame_at = monotonic_ts

    def _stamp_ingress(self, fields: dict) -> None:
        """Stash the video-ingress snapshot on whatever row this tick
        writes. Starvation — the SDK delivering below even the buffer's
        own target rate — additionally logs at INFO so `railway logs`
        shows it without DB access. No-op while the meter is cold
        (paths that bypass the video consumer, e.g. tests pushing
        straight into FrameBuffer, must not stamp lying zeros)."""
        snap = self.ingress_meter.snapshot(time.monotonic())
        if snap is None:
            return
        fields.update(snap)
        recv = float(snap["ingress_recv_fps"])  # type: ignore[arg-type]
        target = self.frame_buffer.target_fps
        if recv < target:
            log.info(
                "%s video ingress STARVED: recv=%.1ffps target=%dfps "
                "kept=%s gap_max=%sms convert_max=%sms res=%s",
                self._log_tag, recv, target,
                snap["ingress_kept_fps"], snap["ingress_max_gap_ms"],
                snap["ingress_convert_ms_max"], snap["ingress_res"],
            )

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
            # Tick failures must NEVER kill the loop. Log and move on; next
            # tick will get another shot. Lead with a HIGH-SIGNAL one-liner so
            # "why did no analysis show up?" is greppable without wading
            # through the stack — a DB CheckViolation on a new provider/enum
            # value (e.g. 'dashscope') otherwise hides at its very bottom.
            reason = (str(exc).splitlines() or [""])[0] or type(exc).__name__
            log.error(
                "%s ⛔ TICK FAILED — %s: %s",
                self._log_tag, type(exc).__name__, reason,
            )
            log.exception("%s tick failed (full traceback): %s", self._log_tag, exc)
            await self._write_failed_row(exc)
            return

    async def _write_failed_row(self, exc: Exception) -> None:
        """Visibility row for a failed tick (Aryaman 2026-07-17: ANY
        tick error — model, DB, ffmpeg, anywhere — must show on the
        dashboard for that camera/tick, not just in logs).

        provider stays 'skipped' because the analyses CHECK constraint
        allows only gemini/openai/openrouter/skipped — the truth rides
        in skipped_reason='tick_failed' + error_class/error_msg. Uses a
        FRESH session (the tick's own session may be mid-rollback) and
        never raises: the visibility writer must not hurt the loop."""
        try:
            fields: dict = {
                "people_present": False,
                "skipped_reason": "tick_failed",
                "error_class": type(exc).__name__,
                "error_msg": ((str(exc).splitlines() or [""])[0] or "?")[:200],
            }
            self._stamp_ingress(fields)
            async with self._sessionmaker() as session:
                cam = await session.get(Camera, self.camera_id)
                state = (
                    cam.state
                    if cam is not None and cam.state in ("open", "closed")
                    else "open"
                )
                await add_analysis(
                    session,
                    camera_id=self.camera_id,
                    provider="skipped",
                    description=(
                        f"(tick FAILED: {type(exc).__name__} — see logs)"
                    ),
                    detector_fields=fields,
                    people_present=False,
                    people_count=0,
                    motion_in_clip=False,
                    state_after=state,
                    captured_at=datetime.now(timezone.utc),
                )
                await session.commit()
        except Exception:  # noqa: BLE001 — never fail the failure writer
            log.debug(
                "%s failed-row write failed", self._log_tag, exc_info=True,
            )

    async def _resolve_caregiver_tz(self, session, patient: "Patient") -> ZoneInfo:
        """Caregiver's IANA timezone for when_scope windows, cached for
        the worker's life. Any failure falls back to the scheduled
        runner's convention (America/New_York) — window gating degrades,
        never crashes a tick."""
        if self._caregiver_tz is not None:
            return self._caregiver_tz
        name = None
        try:
            cg = await session.get(Caregiver, patient.caregiver_id)
            name = cg.timezone if cg else None
        except Exception:  # noqa: BLE001
            log.warning(
                "%s caregiver tz lookup failed — using fallback",
                self._log_tag,
            )
        try:
            tz = ZoneInfo(name) if name else ZoneInfo("America/New_York")
        except ZoneInfoNotFoundError:
            log.warning("%s bad caregiver tz %r — using fallback", self._log_tag, name)
            tz = ZoneInfo("America/New_York")
        self._caregiver_tz = tz
        return tz

    async def _verify_wake(self) -> "BlockMotion | None":
        """Backend wake authority (motion.py #3): assess the last ~3s
        of buffered frames off the event loop. Returns None when
        verification could not run (too few frames, or any error) —
        the caller then honors a SAD hint if one exists (fail open on
        the hint path; the hint-less closed path stays closed, exactly
        like the pre-consolidation backend wake on a still frame)."""
        try:
            recent = self.frame_buffer.snapshot_last_seconds(3.0)
            if len(recent) < 3:
                return None
            return await asyncio.to_thread(self._block_motion.assess, recent)
        except Exception:  # noqa: BLE001
            log.debug(
                "%s wake verification failed open",
                self._log_tag, exc_info=True,
            )
            return None

    async def _run_tick(self) -> None:
        """Run one tick, then drive the data-collection recorder.

        The recorder is driven in a `finally` so EVERY tick path —
        including the early-return short-circuits below — gets a chance
        to advance an open recording session toward its linger/finalize.
        `_drive_recorder` is a no-op unless the feature is wired and the
        tick got far enough to know which patient it's for.
        """
        decision = _RecorderDecision()
        try:
            await self._run_tick_inner(decision)
        finally:
            await self._drive_recorder(decision)

    async def _drive_recorder(self, decision: "_RecorderDecision") -> None:
        """Hand the per-tick decision to the recorder. Never raises into
        the tick — recording must never affect detection."""
        if self._recorder is None or decision.ctx is None:
            return
        try:
            if decision.privacy:
                # Privacy on → drop any in-flight clip immediately, no
                # mux/upload. Idempotent no-op when nothing is recording.
                await self._recorder.abort_camera(self.camera_id)
                return
            await self._recorder.on_tick(
                self.camera_id,
                decision.collect,
                ctx=decision.ctx,
                lead_in_frames=decision.lead_frames,
                lead_in_audio=decision.lead_audio,
            )
        except Exception:
            log.exception("%s data-collection recorder tick failed", self._log_tag)

    async def _run_tick_inner(self, decision: "_RecorderDecision") -> None:
        """One tick — mirrors `src/api/clips.py:ingest_clip` body."""
        self._tick_counter += 1
        # Phone/gateway SAD packet, demoted to a HINT (v2 P2): backend-
        # owned motion (tick_signals over the buffered window) is now
        # the authoritative reading and `motion_in_clip` is the union
        # of the two — computed after the signals pass below. The hint
        # still matters: it's free, it survives windows with no usable
        # frames, and gateway (Tapo) cameras that never publish it no
        # longer lose motion gating entirely.
        sad_motion = self._motion_cache.get(self.camera_id)
        motion_in_clip = sad_motion
        # Set iff this tick was woken out of the closed state by the
        # backend motion pass (whole-frame env fraction that fired).
        backend_wake_env: float | None = None
        captured_at = datetime.now(timezone.utc)
        # One monotonic anchor for EVERY buffer snapshot this tick takes —
        # the main window and the RAPID native window must be the same
        # slice of time even though seconds of VLM latency separate the
        # calls that read them.
        tick_mono = time.monotonic()

        async with self._sessionmaker() as session:
            # v2 scheduler path (P3, flag-gated) needs each spec's row for
            # the dimension columns; the legacy path needs only specs.
            loadout: list = []
            if self._settings.detection_v2_scheduler_enabled:
                loadout = await get_enabled_loadout_for_camera(
                    session, self.camera_id,
                )
                detectors = [spec for spec, _row in loadout]
            else:
                detectors = await get_enabled_detectors_for_camera(
                    session, self.camera_id,
                )
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

            # Base recorder context, refined once presence is known
            # below. Set here (after the patient + camera rows load) so
            # that even an early-return tick can drive an open session
            # toward finalize — `collect` stays False on those paths.
            decision.ctx = RecorderCtx(
                caregiver_id=patient.caregiver_id,
                patient_id=self.patient_id,
                room=cam.room,
                gate="person",
                confidence=None,
            )
            # Privacy is known the moment the patient row loads — set it
            # now so EVERY tick path (including early returns) aborts an
            # open session rather than letting it linger toward upload.
            decision.privacy = patient.privacy_mode_enabled

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

            # User-facing "off" short-circuit. A camera's OWN flags
            # suppress the LLM call:
            #   - cam.disabled  → user hid + paused via Hide button
            #   - cam.ai_paused → user paused AI only (video still flows)
            #
            # The patient-level `cameras_paused` (the bulk disconnect-all /
            # reconnect-all switch) is deliberately NOT a detection gate.
            # Detection follows each camera's own state: disconnect-all
            # already stops detection the correct way — by disconnecting the
            # cameras (no stream → no frames → no ticks). It must NOT also
            # suppress a camera that's been individually brought back online
            # while the patient flag still reads paused. (That mismatch is
            # exactly the "camera's open but no analyses" bug.)
            # We still write a synthetic "skipped" row so the debug feed
            # shows ticks happening (and the state machine stays consistent
            # — counters keep advancing). The LiveKit connection + frame
            # buffer stay alive so re-enabling is instant.
            if cam.disabled or cam.ai_paused:
                if cam.disabled:
                    reason = "camera_disabled"
                    description = "(disabled — LLM skipped)"
                else:
                    reason = "ai_paused"
                    description = "(AI paused — LLM skipped)"
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
                # Liveness write (online=true) lives in
                # `_liveness_writer_loop` below, driven by actual
                # frame arrivals, not by tick scheduling. Calling
                # `apply_liveness_only` from here was a footgun: ticks
                # fire on a fixed cadence regardless of whether frames
                # are arriving, so an ai-paused camera with a long-
                # dead publisher would keep its heartbeat fresh forever
                # because the timer kept ticking and the SKIP branch
                # kept writing. Observed 2026-05-28 in Test 2 — Pi
                # service stopped but online stayed true past 55s.
                # Ingress health rides even fully-paused rows: frames
                # keep flowing while AI is off, so a paused rig measures
                # SDK delivery at zero VLM cost (the reid-off arm of the
                # soak-finding-3 bisect comes free).
                paused_fields = {
                    "people_present": False,
                    "skipped_reason": reason,
                }
                self._stamp_ingress(paused_fields)
                await add_analysis(
                    session,
                    camera_id=self.camera_id,
                    provider="skipped",
                    description=description,
                    detector_fields=paused_fields,
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

            if cam.state == "closed":
                # Closed-state wake — ONE authority for hint and
                # no-hint alike (motion consolidation 2026-07-17):
                # block motion (motion.py #3). The SAD packet and the
                # old whole-frame env fraction are demoted to hints/
                # telemetry — the global fraction is exactly the metric
                # IR night grain defeats (Nagpur 2026-07-16: the hint
                # tripped on 97.4% of night ticks ≈ $31/day of Gemini
                # calls on a verified-empty office), while block motion
                # crushes grain ~60× inside each block and habituates
                # stationary blinkers without ever blinding a block
                # (BASELINE_CAP).
                wake_check = await self._verify_wake()
                if wake_check is not None:
                    woke = wake_check.coherent
                else:
                    # Verifier unavailable (cold buffer / bug). Honor a
                    # SAD hint — fail OPEN on the hint path, never lose
                    # a real wake; with no hint, stay closed exactly as
                    # the pre-consolidation no-hint path did.
                    woke = bool(sad_motion)
                if woke:
                    log.info(
                        "%s closed-state WAKE (sad=%s block_max=%.1f blocks=%d)",
                        self._log_tag, bool(sad_motion),
                        wake_check.max_block_delta if wake_check else -1.0,
                        wake_check.active_blocks if wake_check else -1,
                    )
                    # Stash the wake on whatever row this tick writes —
                    # otherwise a woken tick whose regional floors then
                    # read low lands as motion_src="none" with the wake
                    # visible only in logs. Unit note: since the block-
                    # motion consolidation this carries the max BLOCK
                    # DELTA (luma levels), not the old env fraction.
                    backend_wake_env = (
                        wake_check.max_block_delta if wake_check else 0.0
                    )
                else:
                    state_after = await apply_tick_update(
                        session,
                        camera_id=self.camera_id,
                        people_present=False,
                        motion_in_clip=False,
                    )
                    synthetic_fields = {
                        "people_present": False,
                        "skipped_reason": "camera_closed",
                        "motion_src": "none",
                    }
                    if wake_check is not None:
                        synthetic_fields["wake_block_max"] = (
                            wake_check.max_block_delta
                        )
                        if sad_motion:
                            # The hint said wake and the authority said
                            # grain — make the veto auditable per row.
                            synthetic_fields["wake_vetoed"] = True
                    # Closed nights are exactly when delivery health
                    # must stay visible — stamp ingress on skip rows too.
                    self._stamp_ingress(synthetic_fields)
                    await add_analysis(
                        session,
                        camera_id=self.camera_id,
                        provider="skipped",
                        description="(idle — camera closed, no people detected; LLM skipped)",
                        detector_fields=synthetic_fields,
                        people_present=False,
                        people_count=0,
                        motion_in_clip=False,
                        state_after=state_after.state,
                        captured_at=captured_at,
                    )
                    await session.commit()
                    log.info(
                        "%s SKIP closed-state, no motion (sad=%s block_max=%s) "
                        "— counter=%s/%s",
                        self._log_tag, bool(sad_motion),
                        f"{wake_check.max_block_delta:.1f}" if wake_check else "n/a",
                        state_after.no_people_count, state_after.close_threshold,
                    )
                    return

            # Build the MP4 from the buffered window (video + audio).
            # at_fps: the buffer may retain a higher native rate for the
            # RAPID path; the main call always sees target_fps.
            window = self.frame_buffer.snapshot_last_seconds(
                self._settings.livekit_worker_window_seconds,
                now=tick_mono,
                at_fps=self._settings.livekit_worker_target_fps,
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
                "%s reid reid_ran=%s attribution_ran=%s patient_in_clip=%s "
                "confidence=%.3f face_count=%d",
                self._log_tag,
                identity.reid_ran,
                identity.attribution_ran,
                identity.patient_in_clip,
                identity.confidence,
                identity.face_count,
            )
            # Identity evidence frames (soak observability): captured at
            # the KEY MOMENTS of tracking consistency, not every tick —
            # any change in the identity signature (person appears/
            # leaves, UNKNOWN→PATIENT flip, face lock gained), a LOSS
            # frame when tracks drop to zero (blip evidence: captioned,
            # boxless), and a sparse keepalive during long-stable
            # stretches. Flag-gated; never allowed to touch the tick.
            if self._settings.soak_id_frames_enabled:
                try:
                    trks = identity.tracks or ()
                    sig = (
                        len(trks),
                        tuple(sorted(t.label for t in trks)),
                        sum(1 for t in trks
                            if getattr(t, "face_locked", False)),
                    )
                    prev, self._last_id_sig = self._last_id_sig, sig
                    desc = (
                        f"{sig[0]}p " + "/".join(l[:3] for l in sig[1])
                        + (f" {sig[2]}face" if sig[2] else "")
                    ).strip()
                    note = None
                    if prev is not None and sig != prev:
                        note = (
                            f"LOST was {prev[0]}p" if sig[0] == 0
                            else f"id-change {desc}"
                        )
                    elif prev is None and sig[0]:
                        note = f"first {desc}"
                    elif sig[0] and self._tick_counter % 12 == 0:
                        note = f"steady {desc}"
                    if note is not None:
                        await dump_id_evidence(
                            identity, camera_id=self.camera_id,
                            settings=self._settings, note=note,
                        )
                except Exception:  # noqa: BLE001
                    log.debug(
                        "%s id-evidence dump failed", self._log_tag,
                        exc_info=True,
                    )

            # Layer-1 signals (v2 P2): one diff pass over the SAME
            # window the main call sees, scoped by the tick's tracks —
            # per-track motion, env_motion outside person boxes,
            # presence, face_valid. Observability-first (stashed on
            # every analyses row, including suppression skips); the
            # only decisions riding on it in P2 are the unified
            # `motion_in_clip` below and the closed-state wake above.
            # Always-on path for EVERY camera — a raise here would brick
            # every tick, so it gets the same armor as the reid call
            # (PR #20 review M1): failure degrades to zeroed signals and
            # the SAD hint carries motion alone.
            try:
                signals = await asyncio.to_thread(
                    compute_tick_signals,
                    window,
                    tracks=identity.tracks,
                    attribution_ran=identity.attribution_ran,
                    sample_width=self._settings.reid_inference_sample_width,
                )
            except Exception:  # noqa: BLE001
                log.exception("%s tick_signals crashed — zeroed", self._log_tag)
                signals = TickSignals(
                    person_present=None, people_count=None,
                    patient_present=None, track_motion=(),
                    env_motion=0.0, face_valid=None, frames_used=0,
                )
            motion_floor = self._settings.signals_motion_min_active_fraction
            backend_motion = signals.env_motion >= motion_floor or any(
                t.amount >= motion_floor for t in signals.track_motion
            )
            # motion_in_clip = SAD hint OR backend reading. This also
            # un-breaks gateway (Tapo) cameras, which never publish the
            # packet and therefore could never satisfy motion-gated
            # paths (data collection) before. A closed-state backend
            # wake counts too: its whole-frame reading and the regional
            # readings here use different denominators, and a woken
            # tick must never record itself as motionless.
            motion_in_clip = sad_motion or backend_motion or (
                backend_wake_env is not None
            )
            motion_src = (
                "sad+backend" if sad_motion and backend_motion
                else "sad" if sad_motion
                else "backend" if backend_motion or backend_wake_env is not None
                else "none"
            )
            wake_fields: dict = (
                {} if backend_wake_env is None
                else {
                    "woke_by_backend": True,
                    "wake_env_motion": round(backend_wake_env, 4),
                }
            )
            log.info(
                "%s signals motion_src=%s env=%.4f tracks=%d",
                self._log_tag, motion_src, signals.env_motion,
                len(signals.track_motion),
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
            #   reid_ran=False (timeout/crash) -> bypass
            #   attribution_ran=False (reid_enabled off / no enrollment
            #     — tracking ran but identity labels were impossible,
            #     v2 always-on tracking) -> bypass
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
            # `has_ready_enrollment` guard remains the day-zero clause,
            # without an enrollment we can't tell "non-patient" from
            # anything else.
            if (
                identity.reid_ran
                and identity.attribution_ran
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
                        "identity_attribution_ran": identity.attribution_ran,
                        "identity_patient_in_clip": False,
                        "identity_confidence": round(identity.confidence, 4),
                        "identity_face_count": identity.face_count,
                        "identity_has_only_verified_other_tracks":
                            identity.has_only_verified_other_tracks,
                        "motion_src": motion_src,
                        **wake_fields,
                        **signals.stash_fields(),
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

            # Per-account VLM routing + fallback chain. Which model this
            # family runs on comes from `caregivers.vlm_provider` (cached
            # ~60s, so flipping it takes effect within a minute and adds no
            # per-tick query). `run_vlm` runs that provider and falls back
            # through the rest of the chain so a single provider's outage /
            # rate-limit can't blind the family. Gemini gets native video;
            # openai / openrouter share the frame-sampling path. See
            # services/detection/vlm_routing.py.
            vlm_choice = await resolve_vlm_provider(
                session, patient.caregiver_id, self._settings,
            )

            # Optional motion overlay: tint moved regions red so the VLM can see
            # small/sustained motion (tremor/shaking). Only the native-video
            # (Gemini) path consumes this mp4 (the frame-sampled providers use
            # Re-ID's clean frames), so only build it when the choice is gemini.
            # Re-ID above ran on the CLEAN clip. Runs off the event loop; any
            # failure falls back to the clean clip.
            vlm_mp4 = mp4_bytes
            if self._settings.motion_overlay_enabled and vlm_choice == "gemini":
                try:
                    tinted = await asyncio.to_thread(overlay_motion, window)
                    vlm_mp4 = await frames_to_mp4(
                        tinted,
                        fps=self._settings.livekit_worker_target_fps,
                        pcm=pcm if pcm else None,
                        pcm_sample_rate=self.audio_buffer.sample_rate,
                        pcm_num_channels=self.audio_buffer.num_channels,
                    )
                except Exception:  # noqa: BLE001
                    log.exception(
                        "%s motion_overlay failed — using clean clip",
                        self._log_tag,
                    )
                    vlm_mp4 = mp4_bytes

            if self._settings.detection_v2_scheduler_enabled:
                # ── v2 path (P3): L1 due-set → L2 call plans → executor.
                # The ONLY sanctioned coupling point with the new
                # architecture (D1); everything downstream (speech
                # sanitizer, presence gate, analyses, promotion) consumes
                # the merged result unchanged.
                entries = [
                    ScheduledDetector(
                        spec=s, dims=resolve_dims(row),
                        when_scope=getattr(row, "when_scope", None),
                    )
                    for s, row in loadout
                ]
                # RAPID retention (Ledger row 5): hold native `rapid_fps`
                # only while the loadout carries a RAPID-bucket detector.
                # Idempotent; other consumers decimate via at_fps anyway.
                target_fps = self._settings.livekit_worker_target_fps
                self.frame_buffer.retarget(
                    max(self._settings.rapid_fps, target_fps)
                    if any(e.dims.bucket == "rapid" for e in entries)
                    else target_fps
                )
                due_set = resolve_due(
                    entries,
                    signals=signals,
                    identity=identity,
                    settings=self._settings,
                    tick_index=self._tick_counter,
                    local_now=datetime.now(
                        await self._resolve_caregiver_tz(session, patient)
                    ),
                )
                plans = pack_calls(due_set, settings=self._settings)
                log.info(
                    "%s sched due=%d skipped=%d calls=%d crowd=%s",
                    self._log_tag, len(due_set.due), len(due_set.skipped),
                    len(plans), due_set.crowd_fallback,
                )
                # Overlay (RAPID) plans diff + encode at the buffer's
                # native rate; the fps handed over is MEASURED from the
                # snapshot's real spacing so a just-retargeted buffer
                # (one window of 4fps history) remuxes honestly instead
                # of time-compressing motion — the executor marks the
                # shortfall sched_degraded_fps.
                native_window = None
                native_fps = None
                if any(p.channel == "overlay" for p in plans):
                    native_window = self.frame_buffer.snapshot_last_seconds(
                        self._settings.livekit_worker_window_seconds,
                        now=tick_mono,
                    )
                    if len(native_window) >= 2:
                        span = (
                            native_window[-1].captured_at
                            - native_window[0].captured_at
                        )
                        if span > 0:
                            native_fps = max(
                                1, round((len(native_window) - 1) / span)
                            )
                try:
                    sched = await run_scheduled_calls(
                        plans,
                        scene_mp4=vlm_mp4,
                        window=window,
                        native_window=native_window,
                        native_fps=native_fps,
                        pcm=pcm if pcm else None,
                        pcm_sample_rate=self.audio_buffer.sample_rate,
                        pcm_num_channels=self.audio_buffer.num_channels,
                        identity=identity,
                        settings=self._settings,
                        vlm_choice=vlm_choice,
                        run_main_vlm=run_vlm,
                        flash_provider=gemini_provider,
                        log_tag=self._log_tag,
                    )
                except Exception:  # noqa: BLE001
                    log.error(
                        "%s scheduler tick failed (choice=%s)",
                        self._log_tag, vlm_choice,
                    )
                    # Re-raise: _run_tick_safely writes the visible
                    # `tick_failed` row (2026-07-17 — silent returns
                    # here made dying ticks invisible on the dashboard).
                    raise
                used_provider = sched.provider
                result = ProviderResult(
                    description=sched.description,
                    detector_fields=sched.detector_fields,
                    people_present=sched.people_present,
                    people_count=sched.people_count,
                    usage=None,  # ledger keys come pre-summed in sched_fields
                )
                result.detector_fields.update(sched.sched_fields)
                if due_set.skipped:
                    result.detector_fields["sched_skipped"] = [
                        f"{ident}:{reason}" for ident, reason in due_set.skipped
                    ]
                if due_set.crowd_fallback:
                    result.detector_fields["sched_crowd_fallback"] = True
            else:
                try:
                    used_provider, result = await run_vlm(
                        choice=vlm_choice,
                        mp4=vlm_mp4,
                        detectors=detectors,
                        identity=identity,
                        log_tag=self._log_tag,
                        settings=self._settings,
                    )
                except Exception:  # noqa: BLE001
                    log.error(
                        "%s all VLM providers failed (choice=%s)",
                        self._log_tag, vlm_choice,
                    )
                    # Re-raise for the same visibility row as the v2 path.
                    raise

            # Deterministic guard on the model's speech fields: if the
            # actual PCM has no voiced speech (VAD), or the transcript is a
            # degenerate loop, blank speech_heard/speech_transcript. Audio
            # LLMs fabricate transcripts from keyboard/room noise; this
            # stops that reaching the analyses. Detection's native-audio
            # reasoning (agitation tone) is untouched.
            sanitize_speech_fields(
                result.detector_fields,
                pcm=pcm,
                sample_rate=self.audio_buffer.sample_rate,
                num_channels=self.audio_buffer.num_channels,
                log_tag=self._log_tag,
            )

            # Stash identity in detector_fields so the analyses row
            # carries it for the dashboard / event_promotion path.
            # The `identity_reid_ran` flag is the gate signal that
            # tells event_promotion whether to apply identity
            # suppression at all (False = bypass, identity-agnostic).
            result.detector_fields["identity_reid_ran"] = identity.reid_ran
            # v2 P2: reid_ran alone no longer distinguishes "attribution
            # ran, no match" from a tracking-only tick (flag off / no
            # enrollment) — persist both so analyses rows stay debuggable.
            result.detector_fields["identity_attribution_ran"] = identity.attribution_ran
            result.detector_fields["identity_patient_in_clip"] = identity.patient_in_clip
            result.detector_fields["identity_confidence"] = round(identity.confidence, 4)
            result.detector_fields["identity_face_count"] = identity.face_count
            # Layer-1 signals + unified motion verdict (v2 P2) — the
            # sig_* block is the soak's raw material for the P3 floors.
            result.detector_fields["motion_src"] = motion_src
            result.detector_fields.update(wake_fields)
            result.detector_fields.update(signals.stash_fields())

            # Token ledger (v2): per-call spend onto the analyses row when
            # the provider reports it. Raw data for the cost governor.
            if result.usage:
                result.detector_fields["vlm_input_tokens"] = result.usage.get("input", 0)
                result.detector_fields["vlm_output_tokens"] = result.usage.get("output", 0)

            # Gateway-threshold tuning stats (see motion.py #1/#4):
            # observation only, queryable per row. Supersedes the
            # 2026-07-16 hotfix's inline copy.
            try:
                result.detector_fields.update(
                    await asyncio.to_thread(sample_stats, window)
                )
            except Exception:  # noqa: BLE001 — stats must never break a tick
                log.debug("%s sample_stats failed", self._log_tag, exc_info=True)
            # SDK-delivery health for this tick (soak #1 finding 3).
            self._stamp_ingress(result.detector_fields)

            # Presence gate (v2 P2: state machine fed by backend presence).
            # UNION semantics — a person is present if the LLM says so OR
            # the tracker holds any body track OR the face matcher locked
            # the patient (POI). `people_present` comes from the LLM's
            # always-on prompt block (PEOPLE_PRESENT_SPEC in prompt.py).
            #
            # Why union, per source:
            #   • Tracker ADDS presence it can see (face turned away,
            #     occluded, motion-blurred — the LLM misses some of
            #     these). With always-on tracking this now covers
            #     unenrolled patients too — the old faces_enrolled
            #     branch is gone. Deliberate reversal of the earlier
            #     "YOLO body count doesn't gate presence" rule, per the
            #     v2 spec (§1.3 person_present: tracker + LLM as
            #     confirmation). Known cost: a TV screen YOLO tracks as
            #     a person can hold a room open — soak watchlist item;
            #     P3's motion gating makes an open idle room cheap.
            #   • Union means the tracker can only ever ADD presence,
            #     never deny it — the empty-room lesson stands: when
            #     re-ID didn't run (timeout/crash), the LLM alone
            #     decides, and nothing hard-forces "person" (that bug
            #     once kept empty rooms occupied forever).
            if identity.reid_ran:
                tracker_present = identity.people_present_reid
                gate_people_present = bool(
                    result.people_present
                    or identity.patient_in_clip
                    or tracker_present
                )
                gate_people_count = max(
                    result.people_count, identity.people_count_reid
                )
                if identity.patient_in_clip:
                    present_src = "reid_poi"
                elif tracker_present and result.people_present:
                    present_src = "llm+tracker"
                elif tracker_present:
                    present_src = "tracker_only"
                elif result.people_present:
                    present_src = "llm_only"
                else:
                    present_src = "none"
            else:
                gate_people_present = bool(result.people_present)
                gate_people_count = result.people_count
                present_src = "reid_unavailable_llm"
            result.detector_fields["people_present_src"] = present_src
            result.detector_fields["people_count_reid"] = identity.people_count_reid

            # Data-collection recorder decision — READ-ONLY over the
            # signals detection just computed (opt-in flag + motion +
            # presence). Reuses the analysis window/pcm already snapshotted
            # above as the session lead-in. Never alters detection; the
            # actual recording happens in the RoomWorker-owned recorder.
            collect = should_collect(
                privacy_mode_enabled=patient.privacy_mode_enabled,
                data_collection_enabled=patient.data_collection_enabled,
                motion_in_clip=motion_in_clip,
                gate_people_present=gate_people_present,
            )
            if collect:
                decision.collect = True
                decision.ctx = RecorderCtx(
                    caregiver_id=patient.caregiver_id,
                    patient_id=self.patient_id,
                    room=cam.room,
                    gate="poi" if identity.patient_in_clip else "person",
                    confidence=identity.confidence,
                )
                # No lead-in. The 4fps analysis window can't be replayed
                # through the CFR encoder (feed() keeps only the latest
                # frame, so a batch collapses to one), but the matching
                # audio window WAS written to the side-file — which desynced
                # A/V by ~window seconds. Record live-forward only; a paced
                # lead-in replay is a future improvement.

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
                settings=self._settings,
            )
            await session.commit()

            # High-signal confirmation that a real analysis landed — grep
            # "analysis SAVED" to watch live ticks; its absence (with no
            # FAILED / skipped line) means the pipeline stalled somewhere.
            log.info(
                "%s ✅ analysis SAVED — provider=%s people_present=%s count=%s state=%s",
                self._log_tag, used_provider, gate_people_present,
                gate_people_count, state_after.state,
            )

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
                #
                # PRIVACY: the text event + push/email above still fire
                # (the insight reaches the caregiver), but under privacy
                # mode we DON'T archive or surface the video evidence —
                # skip spawning the post-roll/upload work entirely.
                # `archive_event` also re-checks privacy before upload to
                # cover a flip during the post-roll window.
                if not patient.privacy_mode_enabled:
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
