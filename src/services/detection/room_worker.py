"""One subscriber-bot worker per LiveKit room (= per patient).

`RoomWorker` joins a `patient_<id>` room as a viewer-role participant,
listens for cameras to publish tracks, and spins up one `CameraIngest`
per camera. It also routes inbound `motion_state` data packets from
each wall phone into the shared `MotionCache`.

Audio is consumed alongside video: `rtc.AudioStream` is asked for
16 kHz mono int16 PCM, and each chunk is copied into the per-camera
`AudioBuffer`. `mp4_remux.py` then layers it into the MP4 alongside
the video frames so the Gemini prompt's `speech_heard` /
`speech_transcript` fields actually fire.

Camera identity convention (set at pairing time in
`src/api/cameras.py`): `cam_<8 hex chars of camera.id.hex>`. We
resolve identity → full `cameras` row via a DB query when a track is
first subscribed.
"""

from __future__ import annotations

import array
import asyncio
import functools
import json
import logging
import math
import secrets
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from sqlalchemy import select

from src.db.camera import Camera
from src.db.patient import Patient
from src.services.data_collection.manager import RecorderManager
from src.services.data_collection.manager import temp_root as recorder_temp_root
from src.services.detection import frame_codec
from src.services.detection.camera_ingest import CameraIngest
from src.services.detection.patient_presence import PresenceRegistry
from src.services.detection.room_reconciler import LiveKitRoomReconciler

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.motion_cache import MotionCache
    from src.services.identity.identity_client import IdentityProvider
    from src.settings import Settings

log = logging.getLogger(__name__)

# Worker identity prefix in the LiveKit room. Visible in LiveKit Cloud
# logs alongside `cam_<...>` (publisher) and `caregiver_<...>` (talk)
# identities — useful for triaging "is the bot actually joined?".
_WORKER_IDENTITY_PREFIX = "detector_worker_"

# Token TTL for the bot's own viewer JWT. Active rotation (see
# `_TOKEN_RECYCLE_INTERVAL_S` below) refreshes the token well inside
# this window — TTL is the safety margin, not the rotation period.
_WORKER_TOKEN_TTL = timedelta(hours=12)

# Active JWT rotation. Every `_TOKEN_RECYCLE_INTERVAL_S`, force a
# clean disconnect so the supervisor's reconnect cycle mints a fresh
# token. Required because the LiveKit Python SDK reuses the
# connect-time token forever (no refresh callback) — without active
# rotation, a backend that stays up past TTL eventually gets stuck
# in a 401 resume loop the first time a transient network blip
# forces the SDK to re-establish its WebSocket with the stale token
# (observed 2026-05-29: backend up 36h, JWT expired at +12h, network
# blip at +52h triggered a 9-attempt 18-minute resume loop, all 401).
# Recycle at TTL/3 leaves ~8h headroom: if the recycle's reconnect
# fails, the supervisor's backoff loop retries until success well
# before the token actually expires.
_TOKEN_RECYCLE_INTERVAL_S: float = 4 * 60 * 60  # 4 hours

# Reconnect-state watchdog. If the SDK stays in its resume codepath
# longer than this (`reconnecting` event fired, no matching
# `reconnected` since), force a clean disconnect so the supervisor
# can mint a fresh token instead of letting the SDK retry forever
# with the existing one. Belt-and-braces for the same 401-loop bug
# the periodic recycle prevents proactively: this catches the case
# where the network blip happens between recycles. The SDK's normal
# resume completes in well under 10s; a 90s stuck-resume window is
# unambiguously broken.
_RECONNECT_WATCHDOG_TIMEOUT_S: float = 90.0
_RECONNECT_WATCHDOG_TICK_S: float = 5.0

# Topic the wall phone publishes its motion heartbeats on. Mirrored
# in `nemara-camera-app/NemaraCam/LiveKit/LiveKitPublisher.swift`.
_MOTION_STATE_TOPIC = "motion_state"

# Reconnect backoff for the supervisor loop. Starts at 2s and
# doubles to a 60s cap on consecutive connect failures; resets on a
# clean connect.
_RECONNECT_BACKOFF_INITIAL = 2.0
_RECONNECT_BACKOFF_MAX = 60.0

# Storm detection. A healthy LiveKit session lives for many seconds (up
# to the token-recycle interval). A connect that drops in under this many
# seconds — especially on DuplicateIdentity — means another worker is
# contending for our identity. We do NOT reset backoff in that case (so we
# don't hammer LiveKit Cloud in a tight evict/reconnect loop) and, after a
# few rapid cycles, log at ERROR so the condition is *visible* instead of
# buried in INFO/WARNING noise. With per-instance unique identities this
# should never trip; it's defense + observability.
_STORM_SESSION_THRESHOLD_S = 8.0
_STORM_ALERT_CYCLES = 3

# Per-track consumer (audio / video) backoff. The room may stay
# connected while a single track's stream throws — usually a brief
# decode hiccup. Retry in-place a few times before giving up; the
# next track_subscribed event (or a room reconnect) will spawn a
# fresh consumer either way.
_CONSUMER_RETRY_BACKOFF_INITIAL = 1.0
_CONSUMER_RETRY_BACKOFF_MAX = 16.0
_CONSUMER_MAX_RETRIES = 5

# Capacity for the SDK-side frame queues. LiveKit's VideoStream /
# AudioStream default to capacity=0 == UNBOUNDED (RingQueue only
# drops when capacity > 0), and each queued video event pins a full
# native-resolution frame copy (~5.5 MB at 2K). Any sustained
# delivery-minus-drain deficit therefore grows RSS without limit: a
# 0.1 fps shortfall at 2K is ~2 GB/hour — the exact shape of the
# 2026-08-14 production OOMs (linear climb while streaming, flat when
# idle). With a bound, the ring drops the OLDEST frame, which is the
# semantic the pipeline already has: FrameBuffer.push re-applies its
# min-gap rule and discards most delivered frames anyway.
#
# Video: 8 frames ≈ half a second at the 15 fps publish rate — enough
# to absorb a compress hop or a brief event-loop stall, small enough
# that a stalled consumer costs ~44 MB/camera, not gigabytes.
# Audio: 200 chunks (10 ms each at 16 kHz) ≈ 2 s ≈ 64 kB — audio
# gaps degrade clips, so it gets a deeper buffer; it is cheap.
_VIDEO_STREAM_QUEUE_CAPACITY = 8
_AUDIO_STREAM_QUEUE_CAPACITY = 200

# Dedicated pool for per-frame JPEG compression. `asyncio.to_thread`
# uses the loop's DEFAULT executor, which the tick path saturates with
# multi-hundred-ms numpy/ffmpeg jobs (compute_tick_signals,
# sample_stats, mp4 materialize). Frame compression queued behind
# those stalls the video consumer, which is what lets the SDK-side
# stream queue above grow in the first place. Ingress work gets its
# own lane so tick load can never throttle the drain. Two workers
# cover ~12 cameras at 4 fps × ~20 ms/frame with headroom.
_FRAME_COMPRESS_POOL = ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="frame-compress",
)

# Idle-room auto-teardown window. After this many seconds connected
# to a room with zero `cam_*` ingests, the RoomWorker asks its
# parent DetectorWorker to remove it. Prevents the
# startup-reconciliation overshoot case: P3 joins every active
# patient_* room, but a room with no actual camera publishing has
# nothing for us to subscribe to, so `track_subscribed` never fires,
# `_ingests_by_identity` stays empty, and (without this) we'd sit
# connected to LiveKit forever burning an identity slot.
#
# Window choice: 120s is comfortably longer than the wall phone's
# pairing flow + initial publish handshake (~30s), so we don't
# accidentally evict a room whose camera is briefly between
# reconnects. Short enough that a truly orphaned room is cleaned up
# within a couple of minutes — well within human-noticeable
# observability windows.
_IDLE_TEARDOWN_WINDOW_S: float = 120.0

# How often the idle monitor checks. 30s is granular enough that
# the worst-case eviction lag is window + 30s = 150s, but coarse
# enough that the monitor itself costs nearly nothing.
_IDLE_MONITOR_TICK_S: float = 30.0


class RoomWorker:
    """One worker, one LiveKit room, N cameras."""

    def __init__(
        self,
        *,
        room_name: str,
        settings: "Settings",
        sessionmaker: "async_sessionmaker",
        motion_cache: "MotionCache",
        reid: "IdentityProvider",
        lk_api: "object | None" = None,
        on_idle_timeout: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.room_name = room_name
        # Per-INSTANCE random suffix on the LiveKit identity. Stable for
        # this worker's lifetime (so reconnects keep the same identity),
        # but unique across instances. This is the keystone that prevents
        # the DuplicateIdentity reconnect storm: a leaked/overlapping
        # worker (slow teardown, deploy overlap, or a 2nd backend on the
        # same LiveKit project) now joins with a DIFFERENT identity, so
        # LiveKit never evicts anyone — worst case is one redundant
        # analysis that self-heals, not an infinite evict/reconnect loop.
        # Stale sessions are cleared by `_reap_zombie_workers` (REST
        # remove_participant) on startup reconcile, NOT by relying on
        # DuplicateIdentity eviction.
        self._instance_token = secrets.token_hex(3)
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._reid = reid
        # Keyed by PATIENT, not by room. "Patient confirmed on camera A
        # ⇒ their patient-scoped detectors skip on B/C" (2026-07-18) was
        # one instance per room while a room was a patient; an account
        # room holds every resident, so one registry would have let one
        # person's presence suppress another's detectors.
        self._presence = PresenceRegistry()
        # Optional callback used by the idle monitor to ask the
        # parent DetectorWorker to remove this room. See
        # _IDLE_TEARDOWN_WINDOW_S for the trigger condition.
        # Unidirectional (parent to child capability injection); no
        # circular references introduced.
        self._on_idle_timeout = on_idle_timeout
        # Idempotency flag for the idle teardown signal so we fire
        # the callback at most once even if the monitor races with
        # an eviction in progress.
        self._idle_signalled: bool = False
        # Tracks whether this room has ever seen a `cam_*` ingest
        # spawn. If yes, not eligible for idle teardown (the room
        # is in normal use; a transient empty window is just a
        # camera reconnecting). If never, it's eligible.
        self._has_ever_seen_ingest: bool = False
        self._room: rtc.Room | None = None
        # Active ingest pipelines, keyed by the camera's livekit_identity
        # (= the LiveKit participant identity). Allows fast lookup from
        # data-packet handlers without a DB roundtrip per packet.
        self._ingests_by_identity: dict[str, CameraIngest] = {}
        # Serializes the get-then-spawn in `_handle_track_subscribed`.
        # Video and audio arrive as TWO separate track_subscribed
        # events, both routed through `asyncio.create_task`, so they
        # run concurrently. Without this lock, both see an empty
        # `_ingests_by_identity` for the same identity, both spawn a
        # CameraIngest, both bind their track to their own ingest, and
        # both tick independently → two `public.analyses` rows per
        # 10-second window (one with only video buffered, one with
        # only audio buffered). The lock turns the get-then-spawn into
        # a single atomic critical section per identity.
        self._spawn_lock = asyncio.Lock()
        # Per-identity in-flight video consumer tasks. Stopped on
        # participant_disconnected so we don't leak.
        self._video_tasks_by_identity: dict[str, asyncio.Task[None]] = {}
        # Audio consumer tasks mirror the video map. Audio and video
        # arrive as separate `track_subscribed` events; either can
        # arrive first and the other is wired up when it shows.
        self._audio_tasks_by_identity: dict[str, asyncio.Task[None]] = {}
        # The video publication per publisher, kept so the subscribed
        # simulcast layer can be re-requested AFTER subscribe — a closed
        # camera steps down to save egress and back up on wake, and
        # `set_video_quality` needs the publication to say it on.
        self._video_pubs_by_identity: dict[str, rtc.RemoteTrackPublication] = {}
        # Last layer asked for, per publisher, so the 10s tick only
        # signals on a transition.
        self._layer_by_identity: dict[str, str] = {}
        self._stopped = False
        # Supervisor task that owns the connect → wait-for-disconnect →
        # cleanup → reconnect lifecycle. Spawned by start(), cancelled
        # by stop(). All room state lives behind this single task; the
        # outer worker just sees it spin until told otherwise.
        self._supervisor_task: asyncio.Task[None] | None = None
        # Periodic REST reconciler. Reads the authoritative participant
        # list from LiveKit Cloud and writes `cameras.online` based on
        # the two-signal rule (REST-presence AND fresh frame arrival).
        # Sole writer of the column post-refactor.
        self._reconciler = LiveKitRoomReconciler(
            room_name=room_name,
            room_worker=self,
            lk_api=lk_api,  # type: ignore[arg-type]
            sessionmaker=sessionmaker,
        )
        # Per-patient video data-collection recorder. One per worker,
        # shared across this room's cameras (so its concurrent-session
        # cap bounds the whole room). Reuses the frames detection already
        # decodes — no extra LiveKit subscription. Inert unless a patient
        # has opted in; idle cameras pay an O(1) no-op feed.
        # One shared directory, owned by the data_collection package, so
        # the startup sweep in main.py knows where to look.
        self._recorder = RecorderManager(
            settings=settings,
            temp_dir=recorder_temp_root(),
            region_lookup=self._patient_region,
            sidecar_builder=self._build_sidecar,
        )

    async def _build_sidecar(self, fs, clip_key: str, clip_metadata: dict):
        """The note that travels with a clip: what the pipeline already
        recorded inside this clip's window, re-expressed against its
        timeline.

        Reads only. Nothing new is computed for it, and a failure here
        never costs the clip — the caller treats it as best-effort.
        """
        from src.services.annotations.fetch import (
            build_for_clip,
            clip_facts_from_upload,
        )

        clip = clip_facts_from_upload(
            key=clip_key,
            started_at=fs.session_start_utc,
            duration_seconds=float(fs.video_duration_seconds or 0.0),
            caregiver_id=fs.ctx.caregiver_id,
            patient_id=fs.ctx.patient_id,
            camera_id=fs.ctx.camera_id,
            room=fs.ctx.room,
            # Span-level redaction (2026-08-12): "masked" now means "this
            # clip carries pixelated spans" — the exact ranges ride the
            # object metadata as `sensitive_spans`.
            masked=clip_metadata.get("redacted") == "true",
            timeline_exact=clip_metadata.get("timeline_exact") != "false",
        )
        async with self._sessionmaker() as session:
            return await build_for_clip(
                session, clip, generated_at=datetime.now(timezone.utc),
            )

    async def _patient_region(self, patient_id) -> str:
        """Data-residency lookup for the recorder: resolve the patient's
        region so its research clip routes to the region's B2 bucket. A
        vanished row defaults to US (fail-open on routing — the privacy
        check above is what fails closed on whether to upload at all)."""
        async with self._sessionmaker() as session:
            p = await session.get(Patient, patient_id)
        return p.region if p is not None else "US"

    # ── lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the connection supervisor. Returns immediately; the
        actual LiveKit handshake happens asynchronously.

        Connection failures are NOT raised here — the supervisor's
        backoff loop handles them so the parent DetectorWorker doesn't
        have to retry. Surfaces only as logs.
        """
        if self._supervisor_task is not None:
            return
        self._created_at = time.monotonic()
        self._supervisor_task = asyncio.create_task(
            self._maintain_connection(),
            name=f"room-supervisor-{self.room_name}",
        )
        # Spawn the idle monitor only if the parent registered a
        # callback for it (None = the legacy / test path that doesn't
        # want auto-teardown).
        self._idle_monitor_task: asyncio.Task[None] | None = None
        if self._on_idle_timeout is not None:
            self._idle_monitor_task = asyncio.create_task(
                self._monitor_idle(),
                name=f"room-idle-monitor-{self.room_name}",
            )
        # Periodic REST-based participant reconciler. Replaces the
        # SDK-dict-walking reconciler we used to run inline here; see
        # `room_reconciler.py` for the design rationale.
        await self._reconciler.start()

    def live_camera_identities(self) -> set[str]:
        """The `cam_*` identities this worker is currently consuming.

        Read by `ensure_room_left_if_empty` to answer the question its
        name has always implied: is anyone still publishing here? An
        account room routinely holds several sensors, so one leaving
        says nothing about the others.

        `_ingests_by_identity` rather than the SDK's participant list:
        an ingest exists only once a track is actually subscribed and
        being consumed, which is the thing worth keeping the worker
        alive for. A participant that has joined but published nothing
        would otherwise hold the room open indefinitely.
        """
        return set(self._ingests_by_identity)

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Order matters:
        #   1. stop the reconciler so it cannot spawn or evict during
        #      teardown
        #   2. cancel the idle monitor (it would fire a callback we are
        #      about to render moot)
        #   3. cancel the supervisor (owns the LiveKit connection;
        #      `_teardown_active_session` is its post-cancel cleanup)
        await self._reconciler.stop()
        if getattr(self, "_idle_monitor_task", None) is not None:
            self._idle_monitor_task.cancel()
            try:
                await self._idle_monitor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._idle_monitor_task = None
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except (asyncio.CancelledError, Exception):
                pass
            self._supervisor_task = None
        await self._teardown_active_session()
        # Finalize + upload any recordings still in flight (best-effort).
        await self._recorder.aclose()
        log.info("room_worker.stop room=%s", self.room_name)

    async def _monitor_idle(self) -> None:
        """Periodic check: if this RoomWorker has been alive for
        longer than `_IDLE_TEARDOWN_WINDOW_S` AND has never seen a
        `cam_*` ingest spawn, signal the parent to remove us via
        `_on_idle_timeout`.

        Why "never seen" instead of "currently empty": a room with
        an active camera that briefly reconnects can go through
        empty dict windows without being orphaned. The reconciliation
        overshoot case (P3 joined a patient_* room with no cam
        publishing) NEVER sees a single ingest, which is the
        unambiguous signal.

        Failures (callback raises, callback hangs) are caught and
        logged. The signal is fire-once; we never retry.
        """
        try:
            while not self._stopped:
                await asyncio.sleep(_IDLE_MONITOR_TICK_S)
                if self._stopped:
                    return
                if self._idle_signalled:
                    return
                if self._has_ever_seen_ingest:
                    # Active room — monitor's job is done permanently.
                    return
                age = time.monotonic() - self._created_at
                if age < _IDLE_TEARDOWN_WINDOW_S:
                    continue
                # We've been around long enough with zero ingests —
                # this room is orphaned. Fire the eviction callback.
                self._idle_signalled = True
                log.info(
                    "room_worker.idle_timeout room=%s age=%.1fs — "
                    "signalling parent to remove",
                    self.room_name, age,
                )
                if self._on_idle_timeout is None:
                    return
                try:
                    await asyncio.wait_for(
                        self._on_idle_timeout(self.room_name),
                        timeout=15.0,
                    )
                except asyncio.TimeoutError:
                    log.error(
                        "room_worker.idle_timeout callback timed out "
                        "after 15s room=%s — ghost worker remains until "
                        "next backend restart or LiveKit session timeout",
                        self.room_name,
                    )
                except Exception:
                    log.exception(
                        "room_worker.idle_timeout callback raised "
                        "room=%s — ghost worker remains",
                        self.room_name,
                    )
                return
        except asyncio.CancelledError:
            raise

    # ── connection supervisor ──────────────────────────────────────

    async def _maintain_connection(self) -> None:
        """Outer reconnect loop. Each pass:

        1. Mints a fresh viewer JWT (cached creds the SDK had locally
           may already be expired after a long disconnect).
        2. Creates a fresh `rtc.Room`, wires handlers, calls `connect`.
        3. Waits for a `disconnected` event from the SDK.
        4. Tears down any ingests + consumer tasks the previous
           connection's `track_subscribed` handlers had spawned.
        5. Sleeps a backoff interval, then loops.

        Backoff doubles from 2s up to 60s on consecutive failures so
        we don't hammer LiveKit Cloud during an outage; on a clean
        connect the backoff resets.
        """
        backoff = _RECONNECT_BACKOFF_INITIAL
        rapid_cycles = 0
        while not self._stopped:
            cycle_started = time.monotonic()
            try:
                disconnect_reason = await self._connect_and_wait()
                session_secs = time.monotonic() - cycle_started
                is_duplicate = "duplicate" in disconnect_reason.lower()
                if is_duplicate or session_secs < _STORM_SESSION_THRESHOLD_S:
                    # Session died almost immediately — another worker is
                    # contending for this identity (should be impossible
                    # with per-instance identities). Escalate backoff
                    # instead of fast-reconnecting, and surface it loudly.
                    rapid_cycles += 1
                    backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)
                    if rapid_cycles >= _STORM_ALERT_CYCLES:
                        log.error(
                            "room_worker.reconnect_storm room=%s reason=%s "
                            "session=%.1fs rapid_cycles=%d identity=%s — another "
                            "worker is contending for this identity; backing off "
                            "%.1fs",
                            self.room_name, disconnect_reason, session_secs,
                            rapid_cycles, self._identity(), backoff,
                        )
                    else:
                        log.warning(
                            "room_worker.disconnected room=%s reason=%s "
                            "session=%.1fs — reconnecting in %.1fs",
                            self.room_name, disconnect_reason, session_secs, backoff,
                        )
                else:
                    # A healthy session that dropped after steady state —
                    # an ordinary network blip. Reset backoff.
                    log.warning(
                        "room_worker.disconnected room=%s reason=%s — reconnecting",
                        self.room_name, disconnect_reason,
                    )
                    rapid_cycles = 0
                    backoff = _RECONNECT_BACKOFF_INITIAL
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "room_worker.connect_failed room=%s — retry in %.1fs",
                    self.room_name, backoff,
                )
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)
            finally:
                await self._teardown_active_session()
            if self._stopped:
                break
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise

    async def _connect_and_wait(self) -> str:
        """One connect cycle. Returns the disconnect reason the SDK
        gives us when the room drops; raises on connect-time failures
        so the supervisor backs off.

        Spawns two background tasks for the lifetime of this connect
        cycle: a periodic recycle (forces a clean reconnect every
        `_TOKEN_RECYCLE_INTERVAL_S` to rotate the JWT) and a reconnect
        watchdog (forces a disconnect if the SDK gets stuck in
        `CONN_RECONNECTING` past `_RECONNECT_WATCHDOG_TIMEOUT_S`).
        Both fire `room.disconnect()`, which is the same path a
        natural network drop takes, so the supervisor's reconnect
        machinery handles all three cases uniformly.
        """
        token = self._mint_viewer_token()
        room = rtc.Room()
        disconnected = asyncio.Event()
        disconnect_reason: dict[str, str] = {"reason": "unknown"}
        # Tracks when the SDK last entered the resume path. Set by the
        # `reconnecting` event, cleared by `reconnected` (or by a
        # `disconnected` that ends the cycle anyway). Read by the
        # watchdog. We listen on the purpose-built `reconnecting` /
        # `reconnected` events (room.py:399-402) rather than parsing
        # `connection_state_changed` enum transitions — the dedicated
        # events are emitted directly by the SDK's resume codepath and
        # don't depend on us mapping enum values correctly.
        resume_state: dict[str, float | None] = {"reconnecting_since": None}

        @room.on("disconnected")
        def _on_disconnected(reason: object = None) -> None:
            disconnect_reason["reason"] = str(reason) if reason is not None else "unknown"
            disconnected.set()

        @room.on("reconnecting")
        def _on_reconnecting() -> None:
            # Idempotent: if the SDK fires `reconnecting` twice (resume
            # retry attempt), keep the FIRST timestamp so the watchdog
            # threshold measures total stuck time, not just the current
            # attempt's age.
            if resume_state["reconnecting_since"] is None:
                resume_state["reconnecting_since"] = time.monotonic()
                log.warning(
                    "room_worker.reconnecting_started room=%s",
                    self.room_name,
                )

        @room.on("reconnected")
        def _on_reconnected() -> None:
            if resume_state["reconnecting_since"] is not None:
                elapsed = time.monotonic() - resume_state["reconnecting_since"]
                resume_state["reconnecting_since"] = None
                log.info(
                    "room_worker.reconnecting_recovered room=%s elapsed_s=%.1f",
                    self.room_name, elapsed,
                )

        self._wire_room_handlers(room)
        await room.connect(self._settings.livekit_url, token)
        self._room = room
        log.info(
            "room_worker.connected room=%s url=%s identity=%s",
            self.room_name, self._settings.livekit_url, self._identity(),
        )

        # Backfill `track_subscribed` for participants that joined BEFORE
        # we connected. The LiveKit Python SDK only fires that event
        # for tracks published AFTER our join; tracks already in the
        # room at connect time silently land in
        # `room.remote_participants` without the event. Without this
        # walk, a backend restart or late join leaves the wall phone
        # publishing into a room whose detector worker is technically
        # connected but never spawns a CameraIngest, the idle teardown
        # then evicts the worker after 120s, and the caregiver UI
        # sits on "Offline" over a live LiveKit stream. We manually
        # invoke the same handler the event would have called.
        for participant in room.remote_participants.values():
            for publication in participant.track_publications.values():
                track = publication.track
                if track is None:
                    # Not yet subscribed; the natural `track_subscribed`
                    # event will land in a moment. The handler is
                    # idempotent (per-identity dict + spawn lock) so a
                    # later event arriving after this backfill is a
                    # no-op.
                    continue
                log.info(
                    "room_worker.backfill_existing_track room=%s identity=%s kind=%s",
                    self.room_name,
                    self._participant_identity(participant),
                    track.kind,
                )
                asyncio.create_task(
                    self._handle_track_subscribed(track, publication, participant),
                )

        recycle_task = asyncio.create_task(
            self._recycle_loop(room),
            name=f"room-recycle-{self.room_name}",
        )
        watchdog_task = asyncio.create_task(
            self._reconnect_watchdog_loop(room, resume_state),
            name=f"room-watchdog-{self.room_name}",
        )
        try:
            await disconnected.wait()
            return disconnect_reason["reason"]
        finally:
            for task in (recycle_task, watchdog_task):
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    async def _recycle_loop(self, room: rtc.Room) -> None:
        """Sleep for `_TOKEN_RECYCLE_INTERVAL_S`, then force a clean
        disconnect so the supervisor mints a fresh JWT on its next
        cycle. One-shot per connect cycle — if the disconnect
        succeeds, the surrounding `_connect_and_wait` exits and
        `_maintain_connection` spins us back up with a new token.
        """
        try:
            await asyncio.sleep(_TOKEN_RECYCLE_INTERVAL_S)
        except asyncio.CancelledError:
            raise
        log.info(
            "room_worker.token_recycle room=%s — forcing clean reconnect "
            "to rotate JWT",
            self.room_name,
        )
        try:
            await room.disconnect()
        except Exception:
            log.exception(
                "room_worker.token_recycle_disconnect_failed room=%s",
                self.room_name,
            )

    async def _reconnect_watchdog_loop(
        self,
        room: rtc.Room,
        resume_state: dict[str, float | None],
    ) -> None:
        """If the SDK stays in `CONN_RECONNECTING` longer than
        `_RECONNECT_WATCHDOG_TIMEOUT_S`, force a disconnect so the
        supervisor can rebuild the connection with a fresh token. The
        SDK's internal resume retries reuse the original (possibly
        expired) JWT — without this watchdog, a stuck resume burns
        ~18 minutes before the SDK gives up on its own.
        """
        try:
            while True:
                await asyncio.sleep(_RECONNECT_WATCHDOG_TICK_S)
                since = resume_state["reconnecting_since"]
                if since is None:
                    continue
                elapsed = time.monotonic() - since
                if elapsed < _RECONNECT_WATCHDOG_TIMEOUT_S:
                    continue
                log.warning(
                    "room_worker.reconnect_watchdog_fired room=%s "
                    "stuck_reconnecting_for=%.1fs — forcing disconnect so "
                    "supervisor can mint fresh JWT",
                    self.room_name, elapsed,
                )
                try:
                    await room.disconnect()
                except Exception:
                    log.exception(
                        "room_worker.watchdog_disconnect_failed room=%s",
                        self.room_name,
                    )
                return
        except asyncio.CancelledError:
            raise

    async def _teardown_active_session(self) -> None:
        """Cancel consumer tasks, stop ingests, disconnect the room.
        Safe to call multiple times — every step is no-op when there's
        nothing to clean up."""
        consumer_tasks = (
            list(self._video_tasks_by_identity.values())
            + list(self._audio_tasks_by_identity.values())
        )
        for task in consumer_tasks:
            task.cancel()
        for task in consumer_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._video_tasks_by_identity.clear()
        self._audio_tasks_by_identity.clear()
        for ingest in list(self._ingests_by_identity.values()):
            try:
                await ingest.stop()
            except Exception:
                log.exception(
                    "room_worker.ingest_stop_failed room=%s camera=%s",
                    self.room_name, ingest.camera_id,
                )
        self._ingests_by_identity.clear()
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:
                log.exception("room_worker.room_disconnect_failed room=%s", self.room_name)
            self._room = None

    # ── LiveKit event handlers ─────────────────────────────────────

    def _wire_room_handlers(self, room: rtc.Room) -> None:
        """Register every event we care about. LiveKit's Python SDK
        delivers callbacks synchronously on the SDK's loop; long work
        is pushed onto `asyncio.create_task`.
        """

        @room.on("track_subscribed")
        def _on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant,
        ) -> None:
            asyncio.create_task(
                self._handle_track_subscribed(track, publication, participant),
            )

        @room.on("participant_connected")
        def _on_participant_connected(participant: rtc.RemoteParticipant) -> None:
            # Symmetric to the connect-time backfill in `_connect_and_wait`:
            # the SDK's `track_subscribed` event is supposed to fire for
            # every track on a newly-joined participant, but it doesn't
            # always (observed 2026-05-28 with cam_8e3a9bf9 — Pi published
            # tracks, SDK internally logged the publication, but
            # `track_subscribed` never reached our handler, so no
            # CameraIngest, no ticks, no `online` flip). Walking the new
            # participant's publications ourselves makes the spawn path
            # event-loss-resilient. `_handle_track_subscribed` is
            # idempotent under the per-identity spawn lock, so a
            # native `track_subscribed` arriving later is a safe no-op.
            asyncio.create_task(
                self._backfill_participant_tracks(participant),
            )

        @room.on("participant_disconnected")
        def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            asyncio.create_task(self._handle_participant_left(participant))

        @room.on("data_received")
        def _on_data_received(data: rtc.DataPacket) -> None:
            self._handle_data_packet(data)

    async def _backfill_participant_tracks(
        self, participant: rtc.RemoteParticipant,
    ) -> None:
        """Walk every track publication on `participant` and feed each
        subscribed one through `_handle_track_subscribed`, as if the
        SDK had fired the event itself. Called from the
        `participant_connected` event so a missed `track_subscribed`
        doesn't strand the participant with no CameraIngest.

        Idempotent: the handler's per-identity spawn lock + dict
        membership check make a later native `track_subscribed`
        arrival a no-op for the video/audio task tracking too.
        """
        identity = self._participant_identity(participant)
        log.info(
            "room_worker.participant_connected room=%s identity=%s "
            "publications=%d (walking for missed track_subscribed)",
            self.room_name, identity, len(participant.track_publications),
        )
        for publication in participant.track_publications.values():
            track = publication.track
            if track is None:
                # Not yet subscribed; the SDK should still fire
                # track_subscribed once subscription completes. Our
                # handler is idempotent so the later natural event
                # won't double-spawn.
                continue
            await self._handle_track_subscribed(track, publication, participant)

    def request_idle_layer(self, identity: str, idle: bool) -> None:
        """Step this camera's subscribed layer down while it is closed.

        WHY (2026-08-24, measured): the subscription is what LiveKit
        bills, and it does not care whether we analyze the frames. A
        camera in closed-state with nobody in the room still delivered
        its top layer to every subscriber all night — the Warner at
        2304x1296 is 5 Mbps, and with the backend and its identity
        service both subscribed that is ~4.5 GB/hr to watch an empty
        room. The backend container measured 3.96 GB/hr inbound with one
        occupied camera and one empty one.

        MEDIUM (640x360, 450 kbps) and not LOW (320x180, 160 kbps). LOW
        saves 31x against MEDIUM's 11x, but the tick that WAKES us
        analyzes the window already in the buffer, so whatever layer was
        arriving during the closed period is what a fall is judged on.
        640x360 is the documented rollback quality for all analysis;
        320x180 is not something to show a VLM a fall in. Block motion —
        the wake authority — runs on a 32x18 grid and cannot tell the
        difference either way, so the wake itself is never at risk.
        `livekit_idle_subscribe_quality` is the knob if that trade wants
        moving.

        Only fires on a TRANSITION: a layer request is a signalling
        round-trip and this is consulted every 10s tick.

        No-op where the publisher does not simulcast. The gateway sets
        `.simulcast-off` (the Tapo publishes one 1920x1080 layer and
        nothing else), so today this saves on bridge cameras only —
        turning that back on is the other half of the lever.
        """
        quality = (
            self._settings.livekit_idle_subscribe_quality if idle
            else self._settings.livekit_subscribe_quality
        )
        if self._layer_by_identity.get(identity) == quality:
            return
        publication = self._video_pubs_by_identity.get(identity)
        if publication is None:
            return
        ingest = self._ingests_by_identity.get(identity)
        self._layer_by_identity[identity] = quality
        self._request_layer(
            publication,
            ingest.camera_id if ingest else identity,
            quality_name=quality,
        )

    def _request_layer(
        self, publication: rtc.RemoteTrackPublication, camera_id,
        quality_name: str | None = None,
    ) -> None:
        """Pin the analytics ('think lane') simulcast layer for this
        subscriber. HIGH (native) is the steady state. MEDIUM (640x360)
        is a rollback lever for CPU starvation and buffer pressure only
        — it is NOT quality-neutral, so do not reach for it to save
        money:

          - Identity needs native pixels. gpu/identity/loop.py measured
            640px shipping (2026-07-26) and rejected it: ~9x off
            wire/encode but only ~10% off GPU, "not worth the
            small-distant-person detection risk". A face 20ft out has
            too few pixels to embed at 360p.
          - Gemini's token count IS resolution-independent (~12.4k input
            tokens/call whether the SFU delivers 512x288 or 2304x1296),
            but that is a COST fact, not an accuracy one — the same
            price buys a worse picture.
          - Block motion genuinely does not care (32x18 grid).

        The cost lever is frame COUNT (TARGET_FPS), not resolution.

        History: with NO request, a width-0 UpdateTrackSettings default
        lands on VIDEO_QUALITY_LOW → the 320x180 rung at 2-9fps with
        multi-second stalls (soak finding 3, root-caused 2026-07-17).
        Uses the official set_video_quality (SDK >= 1.1.9; we pin
        1.1.13) — replaces the raw-FFI dimension hack. Re-asserted on
        every (re)subscribe by our caller. Best-effort: on failure we
        stay on whatever layer the SFU chose (= old behavior)."""
        # Settings-driven since 2026-07-19: "high" = the top simulcast
        # layer (the camera's full/native resolution) — JPEG-in-buffer
        # storage made native retainable, so the MEDIUM crash-night
        # triage is no longer the ceiling. "medium" stays selectable
        # for instant rollback if 2K decode starves this rig's CPU
        # (watch ingress_recv_fps after any flip).
        quality_name = (
            quality_name or self._settings.livekit_subscribe_quality or "high"
        ).lower()
        quality = {
            "low": rtc.VideoQuality.VIDEO_QUALITY_LOW,
            "medium": rtc.VideoQuality.VIDEO_QUALITY_MEDIUM,
            "high": rtc.VideoQuality.VIDEO_QUALITY_HIGH,
        }.get(quality_name, rtc.VideoQuality.VIDEO_QUALITY_HIGH)
        try:
            publication.set_video_quality(quality)
            log.info(
                "room_worker.layer_request camera=%s quality=%s",
                camera_id, quality_name.upper(),
            )
        except Exception:  # noqa: BLE001 — never let a hint break subscribe
            log.warning(
                "room_worker.layer_request FAILED camera=%s "
                "(staying on SFU-chosen layer)",
                camera_id, exc_info=True,
            )

    async def _handle_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        identity = self._participant_identity(participant)
        if not identity.startswith("cam_"):
            # Caregiver-talk or another worker — ignore.
            return

        # Resolve the LiveKit identity to a real cameras row exactly
        # once per camera. The result populates `_ingests_by_identity`,
        # and `_spawn_lock` prevents the video+audio race from creating
        # two CameraIngest objects (one per track kind) for the same
        # publisher. The lock is per-RoomWorker, not per-identity, so a
        # camera-A audio event briefly blocks a camera-B video event in
        # the same room — fine in practice (the await inside is one DB
        # roundtrip).
        async with self._spawn_lock:
            ingest = self._ingests_by_identity.get(identity)
            if ingest is None:
                ingest = await self._spawn_ingest_for_identity(identity)
                if ingest is None:
                    # Identity isn't a paired camera in our DB — log
                    # and walk away. This shouldn't happen if pairing
                    # is working, but it's the right defensive
                    # posture.
                    log.warning(
                        "room_worker.unknown_identity room=%s identity=%s",
                        self.room_name, identity,
                    )
                    return

        if track.kind == rtc.TrackKind.KIND_VIDEO:
            # Start consuming if we don't already have a task for this
            # publisher. Re-publishes (publisher dropped + rejoined)
            # cancel the old task and start a fresh one.
            prev = self._video_tasks_by_identity.get(identity)
            if prev is not None and not prev.done():
                prev.cancel()
            self._video_pubs_by_identity[identity] = publication
            # A (re)subscribe starts at the full layer whatever the
            # camera's state was: the step-down is re-asserted by the
            # next closed tick, and starting low would silently outlive
            # the reason for it if the ingest never came back.
            self._layer_by_identity[identity] = (
                self._settings.livekit_subscribe_quality
            )
            self._request_layer(publication, ingest.camera_id)
            consumer = asyncio.create_task(
                self._consume_video(track, ingest),
                name=f"video-consumer-{identity}",
            )
            self._video_tasks_by_identity[identity] = consumer
            log.info(
                "room_worker.video_subscribed room=%s camera=%s identity=%s",
                self.room_name, ingest.camera_id, identity,
            )
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            prev = self._audio_tasks_by_identity.get(identity)
            if prev is not None and not prev.done():
                prev.cancel()
            consumer = asyncio.create_task(
                self._consume_audio(track, ingest),
                name=f"audio-consumer-{identity}",
            )
            self._audio_tasks_by_identity[identity] = consumer
            log.info(
                "room_worker.audio_subscribed room=%s camera=%s identity=%s",
                self.room_name, ingest.camera_id, identity,
            )

    async def _handle_participant_left(self, participant: rtc.RemoteParticipant) -> None:
        identity = self._participant_identity(participant)
        ingest = self._ingests_by_identity.pop(identity, None)
        if ingest is None:
            return
        self._video_pubs_by_identity.pop(identity, None)
        self._layer_by_identity.pop(identity, None)
        for task_map in (self._video_tasks_by_identity, self._audio_tasks_by_identity):
            task = task_map.pop(identity, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await ingest.stop()
        # Flush any in-flight recording for this camera — no more ticks
        # will arrive to linger it out, so finalize + upload now.
        await self._recorder.finalize_camera(ingest.camera_id)
        # `cameras.online = false` is NOT written here. The reconciler
        # observes the same `participant_left` via its REST poll and
        # writes the column once `miss_threshold` polls have agreed.
        # Writing here too would race the reconciler and could flap
        # `online` on a sub-second reconnect (publisher dropped + re-
        # published within one reconciler interval).
        log.info(
            "room_worker.participant_left room=%s camera=%s identity=%s",
            self.room_name, ingest.camera_id, identity,
        )

    def _handle_data_packet(self, data: rtc.DataPacket) -> None:
        """Decode `{type: "motion_state", in_motion: bool}` packets
        and update the shared MotionCache. Anything else on a
        different topic, anything that fails to decode, or anything
        from a non-camera identity is dropped silently — packets from
        the caregiver's Talk-mode broadcaster also reach here (LiveKit
        delivers data packets to every subscriber, regardless of the
        sender's intent).
        """
        if data.topic != _MOTION_STATE_TOPIC:
            return
        participant = data.participant
        identity = self._participant_identity(participant) if participant is not None else ""
        if not identity.startswith("cam_"):
            return
        ingest = self._ingests_by_identity.get(identity)
        if ingest is None:
            return
        try:
            payload = json.loads(bytes(data.data).decode("utf-8"))
            in_motion = bool(payload["in_motion"])
            payload_type = payload.get("type")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning(
                "room_worker.motion_decode_failed room=%s identity=%s err=%s",
                self.room_name, identity, exc,
            )
            return
        if payload_type != "motion_state":
            return
        self._motion_cache.put(
            ingest.camera_id, in_motion=in_motion, received_at=time.monotonic(),
        )
        # Diagnostic: confirms motion_state packets are actually reaching
        # the detector worker, and their value. If this never logs while a
        # publisher is up, the data-channel path is broken (room/subscribe);
        # if it logs in_motion=False while someone is moving, the SAD
        # detector isn't latching (see pi_relay SAD telemetry). Dial to
        # debug once the Pi motion path is confirmed healthy.
        log.info(
            "room_worker.motion_received room=%s camera=%s in_motion=%s",
            self.room_name, ingest.camera_id, in_motion,
        )

    # ── audio consumer ─────────────────────────────────────────────

    async def _consume_audio(self, track: rtc.Track, ingest: CameraIngest) -> None:
        """Pull PCM chunks off the LiveKit AudioStream and push into
        the per-camera buffer. Wrapped in a bounded retry loop so a
        transient decode error doesn't permanently silence audio for
        the rest of the camera's session — we re-open the stream up
        to `_CONSUMER_MAX_RETRIES` times with exponential backoff.

        16 kHz mono int16 is the shape Gemini's audio path is happiest
        with and the cheapest to encode into the clip MP4.
        """

        # DIAGNOSTIC (2026-07-28, silent-clip hunt): periodic level meter
        # on the RAW inbound LiveKit PCM, before the buffer. Field clips
        # carried an all-zero audio track (-91 dB) while live listen had
        # real signal — this log point tells us whether zeros arrive from
        # LiveKit or appear later in buffer/encode. peak_db ≈ -inf/-90s
        # sustained during daytime noise = zeros enter HERE.
        meter = {"peak": 0, "chunks": 0, "last_log": time.monotonic()}

        def _push_audio(event: object) -> None:
            pcm = bytes(event.frame.data)  # type: ignore[attr-defined]
            captured_at = time.monotonic()
            ingest.audio_buffer.push(data=pcm, captured_at=captured_at)
            samples = array.array("h")
            samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
            if samples:
                meter["peak"] = max(meter["peak"], max(abs(s) for s in samples))
            meter["chunks"] += 1
            now = time.monotonic()
            if now - meter["last_log"] >= 30.0:
                peak = meter["peak"]
                peak_db = 20 * math.log10(peak / 32768.0) if peak else float("-inf")
                log.info(
                    "audio_inbound_levels camera=%s chunks=%d peak_db=%.1f",
                    ingest.camera_id, meter["chunks"], peak_db,
                )
                meter.update(peak=0, chunks=0, last_log=now)
            # Fork to the data-collection recorder (sync, non-blocking;
            # no-op unless this camera has an active session). The same
            # capture time the buffer gets goes with it, so the recorder
            # can line audio up against video and spot a stream stall.
            self._recorder.feed_audio(ingest.camera_id, pcm, captured_at)

        await self._consume_with_retry(
            label="audio",
            ingest=ingest,
            open_stream=lambda: rtc.AudioStream(
                track,
                capacity=_AUDIO_STREAM_QUEUE_CAPACITY,
                sample_rate=ingest.audio_buffer.sample_rate,
                num_channels=ingest.audio_buffer.num_channels,
            ),
            consume_one=_push_audio,
        )

    # ── video consumer ─────────────────────────────────────────────

    async def _consume_video(self, track: rtc.Track, ingest: CameraIngest) -> None:
        """Pull frames off the LiveKit VideoStream, convert to I420,
        copy into bytes, and push into the per-camera buffer. Same
        bounded retry pattern as `_consume_audio`."""

        # One-shot per (re)subscribe: kick the reconciler the instant
        # frames actually start flowing, so `cameras.online` flips true
        # within ~1s instead of trailing up to one poll interval (~10s).
        # The `participant_joined` webhook fires an immediate reconcile,
        # but that poll lands BEFORE frames arrive (Pi connect+publish+
        # subscribe takes a few seconds) and sees "no fresh frames", so
        # without this nudge online only flips on the next periodic poll
        # — the visible reconnect lag. A fresh `_consume_video` runs on
        # every subscribe, so this re-arms naturally on reconnect.
        online_signaled = False

        async def _push_video(event: rtc.VideoEvent) -> None:
            nonlocal online_signaled
            frame = event.frame
            now_mono = time.monotonic()
            # JPEG-in-buffer receive rework (2026-07-19): decide FIRST
            # who needs this frame. Frames neither the buffer's sampler
            # nor an active recording session wants skip the native-res
            # convert entirely — at 2K that eliminates roughly half the
            # receive-path CPU (delivered ~15fps vs the 8fps kept), and
            # ALL of it for idle/paused cameras.
            will_buffer = ingest.frame_buffer.accepts(now_mono)
            recording = self._recorder.wants_video(ingest.camera_id)
            convert_ms = 0.0
            if will_buffer or recording:
                # Timed: convert+copy at native res on the event loop —
                # the ingress meter reports the worst case per window so
                # loop starvation is measurable, not guessed (soak #1
                # finding 3). Compression hops to a worker thread.
                t0 = time.perf_counter()
                if frame.type != rtc.VideoBufferType.I420:
                    frame = frame.convert(rtc.VideoBufferType.I420)
                frame_bytes = bytes(frame.data)
                convert_ms = (time.perf_counter() - t0) * 1000.0
                if recording:
                    # Full-rate raw feed to the data-collection recorder
                    # — its encoder wants raw frames, unchanged.
                    self._recorder.feed_video(
                        ingest.camera_id, frame.width, frame.height,
                        frame_bytes,
                    )
                if will_buffer:
                    try:
                        jpeg, luma = await asyncio.get_running_loop().run_in_executor(
                            _FRAME_COMPRESS_POOL,
                            functools.partial(
                                frame_codec.compress_for_buffer,
                                frame.width, frame.height, frame_bytes,
                                quality=self._settings.frame_jpeg_quality,
                            ),
                        )
                    except Exception:  # noqa: BLE001 — never kill the reader
                        log.exception(
                            "room_worker.frame_compress_failed camera=%s",
                            ingest.camera_id,
                        )
                        jpeg = None
                        luma = b""
                    if jpeg is not None:
                        # push() re-applies the min-gap rule, so the
                        # rare frame that raced past accepts() during
                        # the compress await is still dropped correctly.
                        ingest.frame_buffer.push(
                            width=frame.width,
                            height=frame.height,
                            data=jpeg,
                            captured_at=now_mono,
                            luma=luma,
                        )
            ingest.ingress_meter.record(
                now_mono,
                accepted=will_buffer,
                convert_ms=convert_ms,
                width=frame.width,
                height=frame.height,
            )
            # Stamp the liveness timestamp. `LiveKitRoomReconciler`
            # reads this every poll (10s) and treats frames within
            # `liveness_grace_s` (default 30s) as fresh. When this
            # stamp stops advancing the reconciler flips `online=false`
            # after the absent-streak debounce. Decoupled from the
            # detector tick: even an `ai_paused` or `disabled` camera
            # gets its `_last_frame_at` advanced here, so `online`
            # tracks actual stream presence, not detector activity.
            ingest.record_frame_arrival(now_mono)
            if not online_signaled:
                online_signaled = True
                asyncio.create_task(self._reconciler.signal_tick())

        await self._consume_with_retry(
            label="video",
            ingest=ingest,
            open_stream=lambda: rtc.VideoStream(
                track, capacity=_VIDEO_STREAM_QUEUE_CAPACITY,
            ),
            consume_one=_push_video,
        )

    async def _consume_with_retry(
        self,
        *,
        label: str,
        ingest: CameraIngest,
        open_stream,
        consume_one,
    ) -> None:
        """Shared retry harness for `_consume_audio` / `_consume_video`.

        Each retry opens a fresh stream — the SDK invalidates an
        AudioStream / VideoStream after a consume-side error, and
        re-using it surfaces the same exception immediately.
        Cancellation propagates cleanly (participant_left + worker
        teardown both flow through here).
        """
        retries = 0
        backoff = _CONSUMER_RETRY_BACKOFF_INITIAL
        while True:
            stream = open_stream()
            try:
                async for event in stream:
                    # Video's consumer is async (thread-hopped JPEG
                    # compression); audio's stays sync. Awaiting here
                    # also serializes pushes per camera, preserving
                    # frame order through the compress hop.
                    result = consume_one(event)
                    if asyncio.iscoroutine(result):
                        await result
                # Stream ended cleanly (LiveKit closed the track) —
                # exit the retry loop; track_subscribed will fire
                # again if the publisher re-publishes.
                return
            except asyncio.CancelledError:
                raise
            except Exception:
                retries += 1
                log.exception(
                    "room_worker.%s_consumer_error room=%s camera=%s retry=%d",
                    label, self.room_name, ingest.camera_id, retries,
                )
                if retries >= _CONSUMER_MAX_RETRIES:
                    log.error(
                        "room_worker.%s_consumer_giving_up room=%s camera=%s "
                        "after %d retries",
                        label, self.room_name, ingest.camera_id, retries,
                    )
                    return
            finally:
                try:
                    await stream.aclose()
                except Exception:
                    pass
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, _CONSUMER_RETRY_BACKOFF_MAX)

    # ── helpers ────────────────────────────────────────────────────

    async def _spawn_ingest_for_identity(self, identity: str) -> CameraIngest | None:
        """Resolve LiveKit identity → `cameras` row → fresh
        `CameraIngest`. Returns None if no row matches the identity
        (the publisher is publishing to a room we own but isn't
        registered as a paired camera in our DB).
        """
        async with self._sessionmaker() as session:
            row = (await session.execute(
                select(Camera).where(Camera.livekit_identity == identity),
            )).scalar_one_or_none()
        if row is None:
            return None
        ingest = CameraIngest(
            camera_id=row.id,
            patient_id=row.patient_id,
            sessionmaker=self._sessionmaker,
            motion_cache=self._motion_cache,
            patient_presence=self._presence.for_patient(row.patient_id),
            settings=self._settings,
            reid=self._reid,
            livekit_identity=identity,
            recorder=self._recorder,
            on_idle_change=lambda idle, ident=identity: (
                self.request_idle_layer(ident, idle)
            ),
        )
        await ingest.start()
        self._ingests_by_identity[identity] = ingest
        # Latch the "this room has been productive" flag so the idle
        # monitor (if registered) treats us as in-use forever after,
        # not just while the ingest dict is currently non-empty. A
        # camera that briefly drops + rejoins should NOT trigger idle
        # teardown.
        self._has_ever_seen_ingest = True
        return ingest

    async def remove_ingest(self, identity: str) -> None:
        """Idempotent removal of a CameraIngest by its LiveKit identity.
        Called by:
          - `CameraIngest._trigger_self_destruct` after N consecutive
            vanished-row ticks (the camera was deleted from the DB).
          - `DetectorWorker.stop_worker_for_camera` from the camera
            DELETE API handler (explicit teardown on user action).

        Symmetric with `_handle_participant_left` (the LiveKit
        participant-leave path) except we don't get a `participant`
        object — only the identity string. Pops from
        `_ingests_by_identity`, cancels the video + audio consumer
        tasks, awaits `ingest.stop()` with a 5s bound so a hung
        consumer can't park this call forever.
        """
        ingest = self._ingests_by_identity.pop(identity, None)
        if ingest is None:
            # Idempotent: already removed (e.g. participant_left arrived
            # between self-destruct trigger and this call). Common during
            # an admin DELETE + Pi power-off race.
            log.debug(
                "room_worker.remove_ingest room=%s identity=%s — no-op (not present)",
                self.room_name, identity,
            )
            return
        # Cancel the video + audio consumer tasks that are pushing frames
        # into this ingest's buffers. Without cancelling them we'd leak
        # the asyncio tasks and they'd keep trying to write to the now-
        # stopped ingest's buffers (harmless but noisy).
        self._video_pubs_by_identity.pop(identity, None)
        self._layer_by_identity.pop(identity, None)
        for task_map in (self._video_tasks_by_identity, self._audio_tasks_by_identity):
            task = task_map.pop(identity, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        # Bound the ingest stop. ingest.stop() has its own 2s bound on
        # in-flight tick bodies, so total upper bound here is ~7s;
        # 5s + 2s ≈ generous slack to a hung Gemini call before we
        # force-cancel. If even this times out, we log loudly but don't
        # raise — caller (self-destruct or DELETE handler) treats us as
        # best-effort.
        try:
            await asyncio.wait_for(ingest.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning(
                "room_worker.remove_ingest room=%s camera=%s identity=%s — "
                "ingest.stop() timed out after 5s; abandoning",
                self.room_name, ingest.camera_id, identity,
            )
        except Exception:
            log.exception(
                "room_worker.remove_ingest room=%s camera=%s identity=%s — "
                "ingest.stop() raised; abandoning",
                self.room_name, ingest.camera_id, identity,
            )
        # Flush any in-flight recording — the tick loop is gone, so the
        # session would otherwise never linger out.
        await self._recorder.finalize_camera(ingest.camera_id)
        log.info(
            "room_worker.ingest_self_destructed room=%s camera=%s identity=%s",
            self.room_name, ingest.camera_id, identity,
        )

    def _mint_viewer_token(self) -> str:
        """Subscriber-only token for this worker. canPublish=false,
        canSubscribe=true, scoped to one room.
        """
        grants = VideoGrants(
            room=self.room_name,
            room_join=True,
            can_publish=False,
            can_subscribe=True,
            can_publish_data=False,
        )
        return (
            AccessToken(self._settings.livekit_api_key, self._settings.livekit_api_secret)
            .with_identity(self._identity())
            .with_grants(grants)
            .with_ttl(_WORKER_TOKEN_TTL)
            .to_jwt()
        )

    def _identity(self) -> str:
        # `detector_worker_<last 12 hex of patient UUID>_<instance>`.
        # The patient segment groups a worker to its room (used by the
        # zombie reap's prefix match); the per-instance suffix makes the
        # full identity UNIQUE per worker instance.
        #
        # WHY the suffix is back: a previous version dropped it and let
        # the identity be STABLE per patient, deliberately relying on
        # LiveKit's DuplicateIdentity eviction to kick a dying pod's
        # leftover session. The assumption was "REST-driven reconciliation
        # means the reconnect loop no longer occurs." That assumption is
        # FALSE in practice — when a RoomWorker.stop() is slow (e.g. CPU
        # under re-ID load) the old session is stranded and the new worker
        # joins with the SAME identity, so the two evict each other every
        # ~3s forever and no ticks complete. Making the identity unique
        # turns that catastrophic storm into, at worst, one redundant
        # analysis until the stale session is reaped/times out.
        #
        # 12 hex chars = the last UUID group (birthday-safe to ~17M
        # patients); the 6-hex instance suffix disambiguates instances.
        short = self.room_name.removeprefix("patient_")[-12:]
        return f"{_WORKER_IDENTITY_PREFIX}{short}_{self._instance_token}"

    @staticmethod
    def _participant_identity(participant: rtc.RemoteParticipant | None) -> str:
        if participant is None:
            return ""
        ident = getattr(participant, "identity", None)
        # The SDK's `identity` is a plain str on 1.x.
        return str(ident) if ident is not None else ""
