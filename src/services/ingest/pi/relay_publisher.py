"""Per-Pi LiveKit publisher coroutine.

Each Pi camera row in the `cameras` table that has `source_type='pi'`
spawns one `RelayPublisher.run()` task inside the pi_ingest worker
process. The task:

  1. Opens the Pi's MJPEG-over-HTTP stream (`mjpeg_reader.read_mjpeg`).
  2. Decodes each JPEG to a Y plane (motion detector input) plus an
     I420 buffer (LiveKit video input).
  3. Runs the SAD-tile motion detector and publishes a `motion_state`
     LiveKit data packet on a fixed cadence so the existing detection
     pipeline's `MotionCache` populates without changes.
  4. Publishes each frame to LiveKit as a `cam_*` participant
     identical in shape to what the iPhone NemaraCam app publishes
     today. From the room subscriber's perspective (RoomWorker,
     CameraIngest, etc.) a Pi camera is indistinguishable from an
     iPhone camera.

Failure model:

  - Per-frame failures (one bad JPEG, one publish hiccup) are caught
    and logged; the loop continues.
  - Stream-level failures (TCP reset, read timeout) propagate as
    `MjpegStreamError` from the reader and tear down the LiveKit
    connection. The `PiIngestManager` reconnects with backoff.
  - Connect-time failures raise so the manager can back off.

Threading: a single asyncio task per relay. The MJPEG reader does
its I/O; the LiveKit SDK does its I/O; the motion detector and
JPEG decoder are CPU-bound but cheap enough at 15 fps that we do
not need to offload to a thread.

The LiveKit dependencies are injected via factory callables so the
relay can be tested with in-process fakes that record calls without
needing a real LiveKit server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Protocol
from uuid import UUID

from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from src.services.ingest.pi.audio_reader import CHANNELS as AUDIO_CHANNELS
from src.services.ingest.pi.audio_reader import FRAME_SAMPLES as AUDIO_FRAME_SAMPLES
from src.services.ingest.pi.audio_reader import SAMPLE_RATE_HZ as AUDIO_SAMPLE_RATE_HZ
from src.services.ingest.pi.audio_reader import (
    AudioStreamError,
    read_pcm_audio,
)
from src.services.ingest.pi.jpeg_decoder import (
    DecodedFrame,
    JpegDecodeError,
    decode_jpeg,
)
from src.services.ingest.pi.mjpeg_reader import (
    MjpegStreamError,
    read_mjpeg,
)
from src.services.ingest.pi.motion_detector import (
    MotionConfig,
    MotionDetector,
)
from src.settings import Settings

log = logging.getLogger(__name__)

# Match the iPhone NemaraCam app's heartbeat cadence (5 s) so the
# backend's MotionCache TTL works the same for Pi and iPhone cameras.
DEFAULT_MOTION_HEARTBEAT_SECONDS: float = 5.0

# SAD-tile motion threshold for the Pi relay, OVERRIDING the iOS-tuned
# MotionConfig default (1.5). The iPhone's clean sensor sits below 1.5
# when still, so its consecutive-frame counter resets at rest and a
# rising edge fires on each real movement. The Pi's re-compressed MJPEG
# has a far higher per-tile noise floor — MEASURED ~2.1-2.6 max_tile_sad
# on a *static, empty* scene — which is ABOVE 1.5. At threshold=1.5 the
# `max_tile_sad >= threshold` test is permanently true, so the
# consecutive counter never resets, climbs monotonically, and the
# rising-edge condition (count == required) fires exactly ONCE at startup
# and never again: a closed Pi camera goes permanently blind to new
# motion (the symptom: it only re-opened via an AI pause/resume, which
# just force-sets state=open). Real frantic motion measured 17-124, so
# 8.0 sits comfortably above the noise floor (further guarded by the
# 2-consecutive-frame requirement) and well below real motion. Re-measure
# / tune via the `pi_relay.fps … max_tile_sad=` telemetry if a camera's
# environment (lighting, sensor) pushes the static floor higher.
_PI_MOTION_THRESHOLD: float = 8.0

# LiveKit data packet topic the iPhone publishes motion state on.
# Mirrored exactly so the existing RoomWorker._handle_data_packet
# logic recognizes Pi packets without changes.
MOTION_STATE_TOPIC: str = "motion_state"

# LiveKit re-publish (H.264) bitrate ceiling, set on the track publish
# options. The iPhone publishes 720p at 2.5 Mbps; Pi cameras now stream
# full-FOV 1080p, so the cap is lifted to 4 Mbps so the extra detail
# survives the re-encode to BOTH the caregiver live view and the
# detection pipeline (both consume this H.264 track, not the raw MJPEG).
# It is a ceiling, not a target — a still room uses far less and only
# high-motion 1080p scenes approach it. simulcast off.
DEFAULT_MAX_VIDEO_BITRATE_BPS: int = 4_000_000

# Required-field cap on the publish-side framerate. The LiveKit
# Python SDK's protobuf for `publish_track.options.video_encoding`
# requires `max_framerate` to be non-zero or it refuses to serialize
# (caught the first push end-to-end on a Pi). 30 fps is the iPhone
# convention and is well above the Pi Zero W's 15 fps source, so it
# is non-binding in practice.
DEFAULT_MAX_VIDEO_FRAMERATE: float = 30.0

# Token TTL for the publisher. Long enough that we never refresh
# mid-connection, short enough that a stolen token is not useful for
# long. The supervisor mints a fresh token on every reconnect.
DEFAULT_TOKEN_TTL_SECONDS: int = 60 * 60

# Consecutive JPEG-decode failures the relay tolerates before it
# bails out and lets the manager reconnect. Without this guard, a
# Pi that goes glitchy mid-stream (sending malformed JPEGs but no
# TCP error) would silently freeze the LiveKit video while motion
# heartbeats keep firing, leaving the detection pipeline ticking on
# stale frames. 50 frames at 15 fps is ~3.3 s of bad video, long
# enough to ride out a brief WiFi glitch but short enough that a
# real stream failure surfaces quickly.
DEFAULT_MAX_CONSECUTIVE_DECODE_ERRORS: int = 50


@dataclass(frozen=True)
class RelayConfig:
    """One Pi camera's relay configuration. Built from the cameras
    table row by `PiIngestManager`."""

    camera_id: UUID
    camera_identity: str  # cam_<8 hex>, matches cameras.livekit_identity
    room_name: str  # patient_<patient_id>, matches cameras.livekit_room
    # Pull-mode only: the URL the backend used to pull MJPEG from.
    # Empty string for push-mode connections; the request body of
    # POST /v1/cameras/me/frames provides the frames directly. Kept on
    # the dataclass so the pull entrypoint (`run()`) still type-checks.
    pi_stream_url: str = ""
    # Optional audio endpoint. When None, the relay publishes only a
    # video track. When set, an audio LiveKit track is published in
    # parallel; audio-stream failures degrade to video-only without
    # taking down the video relay.
    pi_audio_url: str | None = None
    # Pi-tuned threshold (see _PI_MOTION_THRESHOLD) — NOT the iOS default
    # 1.5, which is below the Pi MJPEG noise floor and silently disables
    # motion detection on a closed camera.
    motion_config: MotionConfig = field(
        default_factory=lambda: MotionConfig(threshold=_PI_MOTION_THRESHOLD)
    )
    motion_heartbeat_seconds: float = DEFAULT_MOTION_HEARTBEAT_SECONDS
    token_ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS
    max_video_bitrate_bps: int = DEFAULT_MAX_VIDEO_BITRATE_BPS
    max_video_framerate: float = DEFAULT_MAX_VIDEO_FRAMERATE
    max_consecutive_decode_errors: int = DEFAULT_MAX_CONSECUTIVE_DECODE_ERRORS
    # Bounds on the MJPEG reader, surfaced here so the manager can
    # tune them per-camera (e.g. tighter timeouts on local LAN).
    mjpeg_connect_timeout_s: float = 5.0
    mjpeg_read_timeout_s: float = 10.0


class _RoomFactory(Protocol):
    """Hook for tests: returns a fresh `rtc.Room`-like object. The
    real factory is `livekit.rtc.Room`; tests pass an in-process
    fake that records connect/publish/disconnect calls.
    """

    def __call__(self) -> rtc.Room: ...


class _VideoSourceFactory(Protocol):
    """Hook for tests: returns a fresh `rtc.VideoSource`-like object
    bound to the given width/height. Real factory is the SDK's
    constructor.
    """

    def __call__(self, width: int, height: int) -> rtc.VideoSource: ...


class RelayPublisher:
    """One-shot relay for a single Pi camera.

    `run()` does the entire connect → publish → motion-heartbeat
    lifecycle, returning normally on cancellation and raising on
    connect-time failures so the manager can apply backoff. The
    manager re-instantiates a fresh `RelayPublisher` on each
    reconnect cycle; this class does not own its own reconnect loop.
    """

    def __init__(
        self,
        config: RelayConfig,
        settings: Settings,
        *,
        room_factory: _RoomFactory | None = None,
        video_source_factory: _VideoSourceFactory | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._settings = settings
        self._room_factory = room_factory or rtc.Room
        self._video_source_factory = video_source_factory or (
            lambda w, h: rtc.VideoSource(w, h)
        )
        self._clock = clock
        # Latest motion state, written by the per-frame loop and read
        # by the heartbeat loop. asyncio Lock would be overkill; the
        # write is a single attribute assignment of an immutable bool.
        self._in_motion: bool = False
        self._motion_detector = MotionDetector(config.motion_config)
        # Throughput meter: count frames published, log fps every
        # `_fps_log_interval_s` seconds. Diagnostic only; cost is one
        # increment + one wall-clock check per frame.
        self._frames_since_log: int = 0
        self._last_fps_log_t: float = self._clock()
        self._fps_log_interval_s: float = 5.0

    async def run(self) -> None:
        """Pull-mode entrypoint: opens the Pi's MJPEG URL and runs
        the publish loop on the resulting frame iterator. Kept for
        backwards compatibility with the (now-disabled) reconciliation
        worker. The push path uses `run_from_iterator` directly,
        feeding it the request body of POST /v1/cameras/me/frames.
        """
        log.info(
            "pi_relay.start.pull camera=%s url=%s",
            self._config.camera_identity, self._config.pi_stream_url,
        )
        async with read_mjpeg(
            self._config.pi_stream_url,
            connect_timeout=self._config.mjpeg_connect_timeout_s,
            read_timeout=self._config.mjpeg_read_timeout_s,
        ) as frames:
            await self.run_from_iterator(frames)

    async def run_from_iterator(
        self, frames: AsyncIterator[bytes],
    ) -> None:
        """Run the relay against an arbitrary async iterator of JPEG
        bytes. The push endpoint passes the multipart-parsed request
        body here; the pull path passes the read_mjpeg() iterator.

        Returns on clean iterator exhaustion (Pi disconnected cleanly).
        Raises MjpegStreamError on persistent decode failure so the
        caller can tear down the connection.
        """
        # Pull the first frame BEFORE connecting to LiveKit so we
        # know the resolution to size the VideoSource with.
        first_frame_bytes = await self._read_first_frame(frames)
        first = decode_jpeg(first_frame_bytes)

        token = self._mint_publisher_token()
        room = self._room_factory()
        disconnected = asyncio.Event()

        @room.on("disconnected")
        def _on_disconnected(reason: object = None) -> None:
            log.warning(
                "pi_relay.disconnected camera=%s reason=%s",
                self._config.camera_identity, reason,
            )
            disconnected.set()

        await room.connect(self._settings.livekit_url, token)
        log.info(
            "pi_relay.connected camera=%s room=%s",
            self._config.camera_identity, self._config.room_name,
        )

        source = self._video_source_factory(first.width, first.height)
        track = rtc.LocalVideoTrack.create_video_track("video", source)
        publish_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_encoding=rtc.VideoEncoding(
                max_bitrate=self._config.max_video_bitrate_bps,
                max_framerate=self._config.max_video_framerate,
            ),
            simulcast=False,
        )
        await room.local_participant.publish_track(track, publish_options)

        # Push the first frame we already decoded.
        self._on_decoded_frame(source, first)

        heartbeat_task = asyncio.create_task(
            self._motion_heartbeat_loop(room, disconnected),
            name=f"pi_relay_heartbeat_{self._config.camera_identity}",
        )

        # Optional audio: only meaningful in the pull-from-URL design
        # (the URL is in RelayConfig). In push mode this stays None;
        # audio support over push lands in a follow-up endpoint
        # (POST /v1/cameras/me/audio) when needed.
        audio_task: asyncio.Task[None] | None = None
        if self._config.pi_audio_url is not None:
            audio_task = asyncio.create_task(
                self._audio_publish_loop(room, disconnected),
                name=f"pi_relay_audio_{self._config.camera_identity}",
            )

        consecutive_decode_errors: int = 0
        try:
            async for jpeg_bytes in frames:
                if disconnected.is_set():
                    break
                try:
                    decoded = decode_jpeg(jpeg_bytes)
                except JpegDecodeError as exc:
                    # One bad frame is recoverable; the next frame
                    # will arrive in ~67 ms. Logging at DEBUG so a
                    # noisy WiFi link does not spam INFO logs.
                    log.debug(
                        "pi_relay.bad_frame camera=%s err=%s",
                        self._config.camera_identity, exc,
                    )
                    consecutive_decode_errors += 1
                    if (
                        consecutive_decode_errors
                        > self._config.max_consecutive_decode_errors
                    ):
                        # Tear down so the caller (push handler /
                        # pull manager) reconnects. Without this, a Pi
                        # sending only bad JPEGs would freeze the
                        # LiveKit video while heartbeats kept the room
                        # alive and the detection pipeline ticked on
                        # stale frames.
                        raise MjpegStreamError(
                            f"{consecutive_decode_errors} consecutive "
                            f"JPEG decode failures on camera "
                            f"{self._config.camera_identity}"
                        ) from exc
                    continue
                consecutive_decode_errors = 0
                self._on_decoded_frame(source, decoded)
        finally:
            heartbeat_task.cancel()
            with _suppress(asyncio.CancelledError):
                await heartbeat_task
            if audio_task is not None:
                audio_task.cancel()
                with _suppress(asyncio.CancelledError):
                    await audio_task
            await room.disconnect()

        log.info("pi_relay.stop camera=%s", self._config.camera_identity)

    # ── internals ────────────────────────────────────────────────

    async def _read_first_frame(
        self, frames: object  # AsyncIterator[bytes]; loose for the Protocol
    ) -> bytes:
        """Pull one JPEG to size the VideoSource. Raises if the
        stream ends before producing a frame.
        """
        async for jpeg_bytes in frames:  # type: ignore[attr-defined]
            return jpeg_bytes
        raise MjpegStreamError(
            f"Pi stream {self._config.pi_stream_url} produced no frames"
        )

    def _on_decoded_frame(
        self, source: rtc.VideoSource, decoded: DecodedFrame
    ) -> None:
        """Common per-frame action: push to LiveKit, run motion
        detector, update latest in_motion flag.
        """
        frame = rtc.VideoFrame(
            decoded.width,
            decoded.height,
            rtc.VideoBufferType.I420,
            decoded.i420,
        )
        source.capture_frame(frame)

        # Run the motion detector on every frame BEFORE the telemetry log
        # so the fps line can also report the live SAD scores — that one
        # line answers "is the detector even seeing pixel change?" (mean/
        # max SAD near zero ⇒ decode/plane problem) vs "is the threshold
        # too high?" (SAD high but never latches).
        result = self._motion_detector.process(decoded.y_plane)
        if result.in_motion:
            # Rising edge: latch in_motion=True. The heartbeat loop
            # will publish this as soon as its cadence permits.
            self._in_motion = True

        # Throughput + motion telemetry: log every _fps_log_interval_s so
        # we can diagnose encode rate, drops, backpressure AND motion
        # detection from a single line.
        self._frames_since_log += 1
        now = self._clock()
        elapsed = now - self._last_fps_log_t
        if elapsed >= self._fps_log_interval_s:
            log.info(
                "pi_relay.fps camera=%s frames=%d elapsed_s=%.2f fps=%.2f "
                "mean_sad=%.1f max_tile_sad=%.1f in_motion_latched=%s",
                self._config.camera_identity,
                self._frames_since_log,
                elapsed,
                self._frames_since_log / elapsed,
                result.mean_sad,
                result.max_tile_sad,
                self._in_motion,
            )
            self._frames_since_log = 0
            self._last_fps_log_t = now

    async def _audio_publish_loop(
        self, room: rtc.Room, disconnected: asyncio.Event
    ) -> None:
        """Pull PCM frames from the Pi's audio URL and publish them
        as a LiveKit audio track.

        Lifecycle is independent of the video loop: an audio stream
        failure (TCP reset, read timeout, wrong content type) logs a
        warning and returns silently. The video relay keeps running
        with the camera as video-only. This degradation matches the
        contract that pi_audio_url is optional.

        The audio track is published lazily on the first successful
        connect so a Pi that is up for video but down for audio does
        not advertise a stale audio track to subscribers.
        """
        assert self._config.pi_audio_url is not None
        try:
            async with read_pcm_audio(self._config.pi_audio_url) as frames:
                source = rtc.AudioSource(AUDIO_SAMPLE_RATE_HZ, AUDIO_CHANNELS)
                track = rtc.LocalAudioTrack.create_audio_track("audio", source)
                publish_options = rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_MICROPHONE,
                )
                await room.local_participant.publish_track(track, publish_options)
                log.info(
                    "pi_relay.audio_connected camera=%s",
                    self._config.camera_identity,
                )

                async for pcm_frame in frames:
                    if disconnected.is_set():
                        return
                    audio_frame = rtc.AudioFrame.create(
                        sample_rate=AUDIO_SAMPLE_RATE_HZ,
                        num_channels=AUDIO_CHANNELS,
                        samples_per_channel=AUDIO_FRAME_SAMPLES,
                    )
                    # Copy PCM bytes into the LiveKit frame buffer.
                    # The frame's `data` is a writable bytearray
                    # exposed by the SDK; slicing keeps the assignment
                    # to exactly `samples_per_channel * 2` bytes.
                    audio_frame.data[: len(pcm_frame)] = pcm_frame
                    await source.capture_frame(audio_frame)
        except AudioStreamError as exc:
            log.warning(
                "pi_relay.audio_stream_failed camera=%s err=%r, "
                "video continues without audio",
                self._config.camera_identity, exc,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning(
                "pi_relay.audio_unexpected_failure camera=%s",
                self._config.camera_identity, exc_info=True,
            )

    async def _motion_heartbeat_loop(
        self, room: rtc.Room, disconnected: asyncio.Event
    ) -> None:
        """Publish a `motion_state` data packet every
        `motion_heartbeat_seconds` until the room disconnects.

        Matches the iPhone publisher's heartbeat schema:
            {"type": "motion_state", "in_motion": bool}
        on topic "motion_state". The MotionCache TTL on the backend
        treats a missing or stale packet as in_motion=false, so a
        crash here degrades gracefully.

        Rising-edge semantics: the per-frame loop sets self._in_motion
        when a streak crosses the threshold. We publish the latch and
        immediately clear it so the NEXT heartbeat reports false
        unless new motion was observed in the interim. Matches how
        iOS interprets the `in_motion` field as "since the last
        heartbeat" rather than instantaneous.
        """
        while not disconnected.is_set():
            try:
                await asyncio.wait_for(
                    disconnected.wait(),
                    timeout=self._config.motion_heartbeat_seconds,
                )
                # Got the disconnect; exit the loop.
                return
            except TimeoutError:
                # Normal cadence tick.
                pass

            # Atomic test-and-clear: a single tuple assignment is one
            # bytecode op under CPython's GIL, so we cannot lose a
            # rising-edge set between the read and the clear. Without
            # this, the per-frame loop could write True between the
            # read and the clear and have its event swallowed.
            in_motion, self._in_motion = self._in_motion, False
            payload = json.dumps({
                "type": "motion_state",
                "in_motion": in_motion,
            }).encode("utf-8")
            try:
                await room.local_participant.publish_data(
                    payload, topic=MOTION_STATE_TOPIC
                )
            except Exception:
                # Heartbeat failure is non-fatal; the per-frame loop
                # is the load-bearing publish path and will detect a
                # real disconnect via the room "disconnected" event.
                log.warning(
                    "pi_relay.heartbeat_failed camera=%s",
                    self._config.camera_identity,
                    exc_info=True,
                )

    def _mint_publisher_token(self) -> str:
        """Publisher token for the camera. Grants:
        - room_join=True (must join to publish)
        - can_publish=True (video + audio + data)
        - can_subscribe=False (publisher does not need to consume
          its own room)
        """
        grants = VideoGrants(
            room=self._config.room_name,
            room_join=True,
            can_publish=True,
            can_publish_data=True,
            can_subscribe=False,
        )
        return (
            AccessToken(
                self._settings.livekit_api_key,
                self._settings.livekit_api_secret,
            )
            .with_identity(self._config.camera_identity)
            .with_grants(grants)
            .with_ttl(timedelta(seconds=self._config.token_ttl_seconds))
            .to_jwt()
        )


class _suppress:
    """Lightweight contextlib.suppress for catching a known exception.

    The stdlib version exists but importing contextlib for a single
    one-line use is heavier than this two-line class. Used in
    `run()`'s heartbeat cancel path.
    """

    def __init__(self, *exc_types: type[BaseException]) -> None:
        self._exc_types = exc_types

    def __enter__(self) -> None:
        return None

    def __exit__(
        self, exc_type: type[BaseException] | None, exc, tb,
    ) -> bool:
        return exc_type is not None and issubclass(exc_type, self._exc_types)


# Type alias used by callers for the run-coroutine factory.
RelayCoroutineFactory = Callable[
    [RelayConfig, Settings], Awaitable[None],
]


def default_relay_coroutine(
    config: RelayConfig, settings: Settings
) -> Awaitable[None]:
    """The default factory the manager calls to get a relay coroutine
    for a camera. Indirected so tests can swap in a fake.
    """
    return RelayPublisher(config, settings).run()
