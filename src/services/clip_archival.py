"""±15s clip archival for detector events.

When a detector fires, the caregiver app needs to see *what happened*
— a list of words isn't enough evidence. This module assembles the
±15s clip the EventDetail player plays:

  pre.mp4  — buffered video+audio from T-15s to T. We already built
             this MP4 for the LLM call at tick T; reuse those bytes
             rather than re-encoding.
  post.mp4 — buffered video+audio from T to T+15s. Sleeps
             `settings.clip_post_event_wait_seconds` and then
             snapshots the FrameBuffer + AudioBuffer (the subscriber-
             bot keeps writing into them continuously between ticks).
  final    — ffmpeg concat-demuxer pass, stitched into one MP4 and
             uploaded as `events/{event_id}.mp4` in the
             `incident-clips` Supabase Storage bucket.

After upload completes, `events.clip_storage_key` is set and
`events.has_clip=true`; Supabase Realtime broadcasts the UPDATE so
the caregiver app's EventDetailView swaps the "Clip processing…"
placeholder for the inline `VideoPlayer`.

LiveKit Track Composite Egress is the alternative architecture the
plan considered. We use the buffer route instead: the subscriber-bot
already has the frames+PCM in memory at tick time, so we avoid the
LiveKit Cloud + Supabase S3-compat-endpoint dependency, and the
post-event capture is deterministic — no webhook needed.

Caveat: if the worker dies inside the post-event sleep window, the
clip is lost. The events row is already in the DB so the caregiver
still sees the alert text — just no playback. Acceptable for v0;
LiveKit Egress is a future option if reliability of clips becomes a
shipping concern.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import TYPE_CHECKING
from uuid import UUID

from src.db.event import Event
from src.services import storage
from src.services.clip_annotate import annotate_clip_for_display
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.audio_buffer import AudioBuffer
    from src.services.detection.frame_buffer import FrameBuffer
    from src.settings import Settings

log = logging.getLogger(__name__)


class ConcatError(RuntimeError):
    """ffmpeg's concat-demuxer pass failed. Captured stderr is in the
    message so the worker can log the underlying cause."""


async def archive_event(
    *,
    event_id: UUID,
    patient_id: UUID,
    pre_event_mp4: bytes,
    frame_buffer: "FrameBuffer",
    audio_buffer: "AudioBuffer",
    sessionmaker: "async_sessionmaker",
    settings: "Settings",
) -> None:
    """Capture the post-event window, concat with the pre-event MP4
    handed in, burn a green bbox tracking the patient through the
    ±15s window, upload the result, and flip `has_clip=true` on the
    events row.

    Errors are caught and logged. The events row stays without a clip
    rather than the whole detector pipeline crashing — caregivers
    still see the alert text either way.

    `pre_event_mp4` is what `_run_tick` already built and handed to
    the LLM. Reusing the bytes saves an ffmpeg encode and guarantees
    pre/post have identical codec params for `-c copy` concat.

    `patient_id` drives the Re-ID gallery lookup at annotation time.
    Annotation is best-effort: if it fails, the un-annotated clip is
    uploaded so caregivers never lose a clip to display-layer issues.
    """
    try:
        await _archive_one(
            event_id=event_id,
            patient_id=patient_id,
            pre_event_mp4=pre_event_mp4,
            frame_buffer=frame_buffer,
            audio_buffer=audio_buffer,
            sessionmaker=sessionmaker,
            settings=settings,
        )
    except Exception:  # noqa: BLE001
        log.exception("clip_archival.failed event_id=%s", event_id)


async def _archive_one(
    *,
    event_id: UUID,
    patient_id: UUID,
    pre_event_mp4: bytes,
    frame_buffer: "FrameBuffer",
    audio_buffer: "AudioBuffer",
    sessionmaker: "async_sessionmaker",
    settings: "Settings",
) -> None:
    # Sleep first so the post-event buffer fills out. The frame and
    # audio consumers in room_worker.py keep pushing into the buffers
    # independently of the tick loop, so by the time we wake the
    # buffers contain a fresh window covering T..T+wait.
    await asyncio.sleep(settings.clip_post_event_wait_seconds)

    post_frames = frame_buffer.snapshot_last_seconds(
        settings.clip_post_event_wait_seconds,
    )
    if not post_frames:
        log.warning(
            "clip_archival.no_post_frames event_id=%s — skipping clip",
            event_id,
        )
        return
    post_pcm = audio_buffer.snapshot_last_seconds(
        settings.clip_post_event_wait_seconds,
    )

    try:
        post_event_mp4 = await frames_to_mp4(
            post_frames,
            fps=settings.livekit_worker_target_fps,
            pcm=post_pcm if post_pcm else None,
            pcm_sample_rate=audio_buffer.sample_rate,
            pcm_num_channels=audio_buffer.num_channels,
        )
    except (Mp4RemuxError, ValueError) as exc:
        log.warning("clip_archival.post_remux_failed event_id=%s err=%s", event_id, exc)
        return

    combined_mp4 = await concat_mp4s(pre_event_mp4, post_event_mp4)

    # Display-only post-processing: burn a green bbox tracking the
    # patient across the clip. Best-effort — returns the input bytes
    # unchanged if the patient has no enrollment, if annotation is
    # disabled, or if any step crashes.
    annotated_mp4 = await annotate_clip_for_display(
        combined_mp4,
        patient_id=patient_id,
        sessionmaker=sessionmaker,
        settings=settings,
    )

    final_key = await storage.upload_clip(
        event_id=event_id, mp4_bytes=annotated_mp4, settings=settings,
    )

    async with sessionmaker() as session:
        event = await session.get(Event, event_id)
        if event is None:
            log.warning(
                "clip_archival.event_vanished event_id=%s "
                "— uploaded clip is orphaned at key=%s",
                event_id, final_key,
            )
            return
        event.clip_storage_key = final_key
        event.has_clip = True
        await session.commit()

    log.info(
        "clip_archival.done event_id=%s key=%s bytes=%d",
        event_id, final_key, len(annotated_mp4),
    )


async def concat_mp4s(pre: bytes, post: bytes) -> bytes:
    """Stitch two MP4s into one via ffmpeg's concat demuxer.

    `-c copy` skips re-encoding because pre and post come from the
    same `frames_to_mp4` invocation (identical codec/params). The
    demuxer needs file paths, not pipes; tempfiles are scoped to this
    function and unlinked in the finally clause.

    Raises `ConcatError` on non-zero exit or missing ffmpeg.
    """
    pre_path: str | None = None
    post_path: str | None = None
    list_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(pre)
            pre_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(post)
            post_path = f.name
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="w", encoding="utf-8",
        ) as f:
            f.write(f"file '{pre_path}'\nfile '{post_path}'\n")
            list_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                "-movflags", "+faststart+empty_moov",
                "-f", "mp4",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise ConcatError("ffmpeg binary not found on PATH") from exc

        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise ConcatError(f"ffmpeg concat exited {proc.returncode}: {err or '(no stderr)'}")
        if not stdout:
            raise ConcatError("ffmpeg concat produced 0 bytes")
        return stdout
    finally:
        for path in (pre_path, post_path, list_path):
            if path is None:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
