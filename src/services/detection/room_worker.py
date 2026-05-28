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

import asyncio
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING

from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from sqlalchemy import select

from src.db.camera import Camera
from src.services.detection.camera_ingest import CameraIngest

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.motion_cache import MotionCache
    from src.services.reid.inference import ReIDInference
    from src.settings import Settings

log = logging.getLogger(__name__)

# Worker identity prefix in the LiveKit room. Visible in LiveKit Cloud
# logs alongside `cam_<...>` (publisher) and `caregiver_<...>` (talk)
# identities — useful for triaging "is the bot actually joined?".
_WORKER_IDENTITY_PREFIX = "detector_worker_"

# Token TTL for the bot's own viewer JWT. Long enough that we don't
# need refresh logic in v0; if we ever do, refresh-at-TTL/2 is the
# pattern.
_WORKER_TOKEN_TTL = timedelta(hours=12)

# Topic the wall phone publishes its motion heartbeats on. Mirrored
# in `nemara-camera-app/NemaraCam/LiveKit/LiveKitPublisher.swift`.
_MOTION_STATE_TOPIC = "motion_state"

# Reconnect backoff for the supervisor loop. Starts at 2s and
# doubles to a 60s cap on consecutive connect failures; resets on a
# clean connect.
_RECONNECT_BACKOFF_INITIAL = 2.0
_RECONNECT_BACKOFF_MAX = 60.0

# Per-track consumer (audio / video) backoff. The room may stay
# connected while a single track's stream throws — usually a brief
# decode hiccup. Retry in-place a few times before giving up; the
# next track_subscribed event (or a room reconnect) will spawn a
# fresh consumer either way.
_CONSUMER_RETRY_BACKOFF_INITIAL = 1.0
_CONSUMER_RETRY_BACKOFF_MAX = 16.0
_CONSUMER_MAX_RETRIES = 5

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
        reid: "ReIDInference",
        on_idle_timeout: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.room_name = room_name
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._reid = reid
        # Optional callback used by the idle monitor to ask the
        # parent DetectorWorker to remove this room. See
        # _IDLE_TEARDOWN_WINDOW_S for the trigger condition.
        # Unidirectional (parent → child capability injection); no
        # circular references introduced.
        self._on_idle_timeout = on_idle_timeout
        # Idempotency flag for the idle teardown signal so we fire
        # the callback at most once even if the monitor races with
        # an eviction in progress.
        self._idle_signalled: bool = False
        # Tracks whether this room has ever seen a `cam_*` ingest
        # spawn. If yes → not eligible for idle teardown (the room
        # is in normal use; a transient empty window is just a
        # camera reconnecting). If never → it's eligible.
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
        self._stopped = False
        # Supervisor task that owns the connect → wait-for-disconnect →
        # cleanup → reconnect lifecycle. Spawned by start(), cancelled
        # by stop(). All room state lives behind this single task; the
        # outer worker just sees it spin until told otherwise.
        self._supervisor_task: asyncio.Task[None] | None = None
        # Per-instance random suffix on the LiveKit identity. Prevents
        # the DuplicateIdentity reconnect loop that happens when a
        # previous backend process was killed ungracefully — LiveKit
        # Cloud holds the old worker's identity slot for ~20s after the
        # TCP drops, and a new backend trying to join with the SAME
        # deterministic identity gets kicked, reconnects, kicks the old
        # ghost, ad infinitum. With a unique suffix per worker instance
        # there's no collision: the new worker connects under a fresh
        # name, the ghost times out in the background, no loop.
        self._identity_suffix = secrets.token_hex(3)

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

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Cancel both background tasks before tearing down LiveKit
        # state. Order: idle monitor first (it might be about to fire
        # a callback that we're already overriding by stopping
        # ourselves), then supervisor (it owns the LiveKit
        # connection). Either task being None is fine.
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
        while not self._stopped:
            try:
                disconnect_reason = await self._connect_and_wait()
                log.warning(
                    "room_worker.disconnected room=%s reason=%s — reconnecting",
                    self.room_name, disconnect_reason,
                )
                # A clean session reached steady state, so reset
                # backoff: this disconnect was a network blip, not a
                # repeated mint/connect failure.
                backoff = _RECONNECT_BACKOFF_INITIAL
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception(
                    "room_worker.connect_failed room=%s — retry in %.1fs",
                    self.room_name, backoff,
                )
            finally:
                await self._teardown_active_session()
            if self._stopped:
                break
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    async def _connect_and_wait(self) -> str:
        """One connect cycle. Returns the disconnect reason the SDK
        gives us when the room drops; raises on connect-time failures
        so the supervisor backs off."""
        token = self._mint_viewer_token()
        room = rtc.Room()
        disconnected = asyncio.Event()
        disconnect_reason: dict[str, str] = {"reason": "unknown"}

        @room.on("disconnected")
        def _on_disconnected(reason: object = None) -> None:
            disconnect_reason["reason"] = str(reason) if reason is not None else "unknown"
            disconnected.set()

        self._wire_room_handlers(room)
        await room.connect(self._settings.livekit_url, token)
        self._room = room
        log.info(
            "room_worker.connected room=%s url=%s identity=%s",
            self.room_name, self._settings.livekit_url, self._identity(),
        )

        await disconnected.wait()
        return disconnect_reason["reason"]

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

        @room.on("participant_disconnected")
        def _on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
            asyncio.create_task(self._handle_participant_left(participant))

        @room.on("data_received")
        def _on_data_received(data: rtc.DataPacket) -> None:
            self._handle_data_packet(data)

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
        for task_map in (self._video_tasks_by_identity, self._audio_tasks_by_identity):
            task = task_map.pop(identity, None)
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await ingest.stop()
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
        await self._consume_with_retry(
            label="audio",
            ingest=ingest,
            open_stream=lambda: rtc.AudioStream(
                track,
                sample_rate=ingest.audio_buffer.sample_rate,
                num_channels=ingest.audio_buffer.num_channels,
            ),
            consume_one=lambda event: ingest.audio_buffer.push(
                data=bytes(event.frame.data),
                captured_at=time.monotonic(),
            ),
        )

    # ── video consumer ─────────────────────────────────────────────

    async def _consume_video(self, track: rtc.Track, ingest: CameraIngest) -> None:
        """Pull frames off the LiveKit VideoStream, convert to I420,
        copy into bytes, and push into the per-camera buffer. Same
        bounded retry pattern as `_consume_audio`."""

        def _push_video(event: rtc.VideoEvent) -> None:
            frame = event.frame
            if frame.type != rtc.VideoBufferType.I420:
                frame = frame.convert(rtc.VideoBufferType.I420)
            ingest.frame_buffer.push(
                width=frame.width,
                height=frame.height,
                data=bytes(frame.data),
                captured_at=time.monotonic(),
            )

        await self._consume_with_retry(
            label="video",
            ingest=ingest,
            open_stream=lambda: rtc.VideoStream(track),
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
                    consume_one(event)
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
            settings=self._settings,
            reid=self._reid,
            # Hand the ingest a back-channel so it can ask to be removed
            # if its `cameras` row vanishes (admin DELETE without a
            # paired participant_left, partition-then-replay races).
            # remove_ingest is idempotent and lock-free of any RoomWorker
            # mutex — see remove_ingest docstring.
            livekit_identity=identity,
            on_self_destruct=self.remove_ingest,
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
        # `detector_worker_<8 hex of room>_<6 hex random per worker instance>`.
        # The room-derived prefix keeps LiveKit logs readable ("this
        # bot belongs to which room"). The random suffix makes the
        # identity unique per backend process / RoomWorker instance —
        # see the comment on `self._identity_suffix` in __init__.
        short = self.room_name.removeprefix("patient_")[:8]
        return f"{_WORKER_IDENTITY_PREFIX}{short}_{self._identity_suffix}"

    @staticmethod
    def _participant_identity(participant: rtc.RemoteParticipant | None) -> str:
        if participant is None:
            return ""
        ident = getattr(participant, "identity", None)
        # The SDK's `identity` is a plain str on 1.x.
        return str(ident) if ident is not None else ""
