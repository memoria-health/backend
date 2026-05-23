"""One subscriber-bot worker per LiveKit room (= per patient).

`RoomWorker` joins a `patient_<id>` room as a viewer-role participant,
listens for cameras to publish tracks, and spins up one `CameraIngest`
per camera. It also routes inbound `motion_state` data packets from
each wall phone into the shared `MotionCache`.

We do NOT subscribe to audio in v0 — the Gemini path's `speech_heard`
/ `speech_transcript` fields will go unfilled until a follow-up adds
PCM-into-ffmpeg in `mp4_remux.py`. The visual detectors (fall,
wandering, agitation, medication) are unaffected.

Camera identity convention (set at pairing time in
`src/api/cameras.py`): `cam_<8 hex chars of camera.id.hex>`. We
resolve identity → full `cameras` row via a DB query when a track is
first subscribed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
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


class RoomWorker:
    """One worker, one LiveKit room, N cameras."""

    def __init__(
        self,
        *,
        room_name: str,
        settings: "Settings",
        sessionmaker: "async_sessionmaker",
        motion_cache: "MotionCache",
    ) -> None:
        self.room_name = room_name
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._motion_cache = motion_cache
        self._room: rtc.Room | None = None
        # Active ingest pipelines, keyed by the camera's livekit_identity
        # (= the LiveKit participant identity). Allows fast lookup from
        # data-packet handlers without a DB roundtrip per packet.
        self._ingests_by_identity: dict[str, CameraIngest] = {}
        # Per-identity in-flight video consumer tasks. Stopped on
        # participant_disconnected so we don't leak.
        self._video_tasks_by_identity: dict[str, asyncio.Task[None]] = {}
        self._stopped = False

    # ── lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        if self._room is not None:
            return
        token = self._mint_viewer_token()
        room = rtc.Room()
        self._wire_room_handlers(room)
        await room.connect(self._settings.livekit_url, token)
        self._room = room
        log.info(
            "room_worker.start room=%s url=%s identity=%s",
            self.room_name, self._settings.livekit_url, self._identity(),
        )

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        # Cancel video consumers first so they don't try to push into
        # buffers as we tear ingests down.
        for task in self._video_tasks_by_identity.values():
            task.cancel()
        for task in self._video_tasks_by_identity.values():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                # Cancellation + tear-down errors are uninteresting at
                # this stage; the worker is shutting down anyway.
                pass
        self._video_tasks_by_identity.clear()
        for ingest in self._ingests_by_identity.values():
            await ingest.stop()
        self._ingests_by_identity.clear()
        if self._room is not None:
            await self._room.disconnect()
            self._room = None
        log.info("room_worker.stop room=%s", self.room_name)

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
        # and the lock prevents a video+audio race from creating two.
        ingest = self._ingests_by_identity.get(identity)
        if ingest is None:
            ingest = await self._spawn_ingest_for_identity(identity)
            if ingest is None:
                # Identity isn't a paired camera in our DB — log and
                # walk away. This shouldn't happen if pairing is
                # working, but it's the right defensive posture.
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
            # v0: don't buffer audio. See module docstring.
            log.info(
                "room_worker.audio_subscribed (not buffered) "
                "room=%s identity=%s",
                self.room_name, identity,
            )

    async def _handle_participant_left(self, participant: rtc.RemoteParticipant) -> None:
        identity = self._participant_identity(participant)
        ingest = self._ingests_by_identity.pop(identity, None)
        if ingest is None:
            return
        task = self._video_tasks_by_identity.pop(identity, None)
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

    # ── video consumer ─────────────────────────────────────────────

    async def _consume_video(self, track: rtc.Track, ingest: CameraIngest) -> None:
        """Pull frames off the LiveKit VideoStream, convert to I420,
        copy into bytes, and push into the per-camera buffer. Runs
        until cancelled (participant left, worker shutting down).
        """
        stream = rtc.VideoStream(track)
        try:
            async for event in stream:
                frame = event.frame
                # Convert to I420 if needed. `frame.convert` allocates
                # a fresh frame; we then copy `.data` into a bytes
                # object so the buffer can outlive the SDK frame.
                if frame.type != rtc.VideoBufferType.I420:
                    frame = frame.convert(rtc.VideoBufferType.I420)
                ingest.frame_buffer.push(
                    width=frame.width,
                    height=frame.height,
                    data=bytes(frame.data),
                    captured_at=time.monotonic(),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception(
                "room_worker.video_consumer_died room=%s camera=%s",
                self.room_name, ingest.camera_id,
            )
        finally:
            await stream.aclose()

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
        )
        await ingest.start()
        self._ingests_by_identity[identity] = ingest
        return ingest

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
        # `detector_worker_<8 hex of room>`. The room name already
        # looks like `patient_<32 hex>` so this is a stable
        # short-string of provenance for LiveKit logs.
        short = self.room_name.removeprefix("patient_")[:8]
        return f"{_WORKER_IDENTITY_PREFIX}{short}"

    @staticmethod
    def _participant_identity(participant: rtc.RemoteParticipant | None) -> str:
        if participant is None:
            return ""
        ident = getattr(participant, "identity", None)
        # The SDK's `identity` is a plain str on 1.x.
        return str(ident) if ident is not None else ""
