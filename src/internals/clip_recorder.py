"""Internal dev tool: on-demand LiveKit clip recorder (data collection).

NON-PRODUCTION. On a record-clip request this connects a LiveKit
SUBSCRIBER to the patient's room, captures the target camera's
video+audio for N seconds (timed from the first frame), muxes ONE MP4
with the real LiveKit audio, and writes it to the dump dir.

Its OWN subscriber + capture buffers, fully independent of the detector
tick — so it produces a true N-second clip WITHOUT touching the
production analysis window (which stays at `livekit_worker_window_seconds`
for the model). This means it does NOT change what the model sees, so the
clips you collect here stay consistent with the analyses already on disk.

Reuses services helpers (allowed direction: internals -> services):
`FrameSnapshot` + `frames_to_mp4`, and the rtc / AccessToken LiveKit
pattern from `room_worker`. Joins subscribe-only with a distinct,
non-`cam_` identity so presence logic (which keys on `cam_*`) ignores it.

One recording at a time (a module lock); a transient room connection per
press (connect -> capture -> disconnect). Nothing under `src/` outside
`src/internals/` imports this module.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.mp4_remux import frames_to_mp4
from src.settings import Settings

log = logging.getLogger(__name__)

# The backend runs on the same laptop during collection, so it writes
# the clip straight to disk — same folder the other clips land in.
_OUT_DIR = Path.home() / "Desktop" / "agitation-clips"
_AUDIO_SAMPLE_RATE = 16_000
_AUDIO_CHANNELS = 1
# Give up if no frame arrives — the camera is closed / not publishing.
_FIRST_FRAME_TIMEOUT = 8.0
_TOKEN_TTL = timedelta(seconds=300)

# One recording at a time per process.
_lock = asyncio.Lock()


def output_dir() -> str:
    return str(_OUT_DIR)


async def record_clip(
    *,
    settings: Settings,
    room_name: str,
    camera_identity: str,
    duration: float,
) -> str | None:
    """Capture one `duration`-second clip of `camera_identity` in
    `room_name`. Returns the saved path, or None on failure / if a
    recording is already running. Best-effort — never raises."""
    if _lock.locked():
        log.warning("clip_recorder.busy — a recording is already in progress; ignoring")
        return None
    async with _lock:
        try:
            return await _capture(settings, room_name, camera_identity, float(duration))
        except Exception:
            log.exception("clip_recorder.record_clip failed")
            return None


async def _capture(
    settings: Settings, room_name: str, camera_identity: str, duration: float,
) -> str | None:
    frames: list[FrameSnapshot] = []
    pcm: list[bytes] = []
    capturing = {"on": False}          # dict so closures see updates unambiguously
    first_frame = asyncio.Event()
    consumers: list[asyncio.Task] = []
    room = rtc.Room()

    def _spawn_video(track: rtc.Track) -> None:
        async def _run() -> None:
            stream = rtc.VideoStream(track)
            try:
                async for ev in stream:
                    if not capturing["on"]:
                        continue
                    f = ev.frame
                    if f.type != rtc.VideoBufferType.I420:
                        f = f.convert(rtc.VideoBufferType.I420)
                    frames.append(FrameSnapshot(
                        width=f.width, height=f.height,
                        data=bytes(f.data), captured_at=time.monotonic(),
                    ))
                    if not first_frame.is_set():
                        first_frame.set()
            finally:
                await stream.aclose()
        consumers.append(asyncio.create_task(_run()))

    def _spawn_audio(track: rtc.Track) -> None:
        async def _run() -> None:
            stream = rtc.AudioStream(
                track, sample_rate=_AUDIO_SAMPLE_RATE, num_channels=_AUDIO_CHANNELS,
            )
            try:
                async for ev in stream:
                    if capturing["on"]:
                        pcm.append(bytes(ev.frame.data))
            finally:
                await stream.aclose()
        consumers.append(asyncio.create_task(_run()))

    @room.on("track_subscribed")
    def _on_track(track, publication, participant) -> None:  # type: ignore[no-untyped-def]
        # Only the target camera's tracks; ignore other participants
        # (the detector worker, other subscribers).
        if participant.identity != camera_identity:
            return
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            _spawn_video(track)
        elif track.kind == rtc.TrackKind.KIND_AUDIO:
            _spawn_audio(track)

    token = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(f"clip_rec_{datetime.now():%H%M%S}")
        .with_grants(VideoGrants(
            room=room_name, room_join=True,
            can_publish=False, can_subscribe=True, can_publish_data=False,
        ))
        .with_ttl(_TOKEN_TTL)
        .to_jwt()
    )

    try:
        await room.connect(settings.livekit_url, token)
        log.info("clip_recorder.connected room=%s target=%s", room_name, camera_identity)

        capturing["on"] = True
        try:
            await asyncio.wait_for(first_frame.wait(), timeout=_FIRST_FRAME_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning(
                "clip_recorder: no frames within %.0fs — camera not publishing? "
                "(room=%s target=%s)", _FIRST_FRAME_TIMEOUT, room_name, camera_identity,
            )
            return None

        # Capture `duration` seconds starting from the first frame.
        await asyncio.sleep(duration)
        capturing["on"] = False

        if not frames:
            log.warning("clip_recorder: 0 frames captured")
            return None

        # Real-time playback: derive fps from how fast frames actually
        # arrived rather than assuming the camera's nominal rate.
        elapsed = frames[-1].captured_at - frames[0].captured_at
        fps = max(1, round(len(frames) / elapsed)) if elapsed > 0.5 else 15

        mp4 = await frames_to_mp4(
            frames, fps=fps,
            pcm=b"".join(pcm) if pcm else None,
            pcm_sample_rate=_AUDIO_SAMPLE_RATE, pcm_num_channels=_AUDIO_CHANNELS,
        )

        await asyncio.to_thread(_OUT_DIR.mkdir, parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = _OUT_DIR / f"rec_{stamp}.mp4"
        await asyncio.to_thread(path.write_bytes, mp4)
        log.info(
            "clip_recorder.saved %s (%d frames @ %dfps, %.1fs span, %d audio chunks, %d bytes)",
            path, len(frames), fps, elapsed, len(pcm), len(mp4),
        )
        return str(path)
    finally:
        for t in consumers:
            t.cancel()
        if consumers:
            await asyncio.gather(*consumers, return_exceptions=True)
        try:
            await room.disconnect()
        except Exception:
            pass
