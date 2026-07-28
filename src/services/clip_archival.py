"""±15s clip archival for detector events.

When a detector fires, the caregiver app needs to see *what happened*
— a list of words isn't enough evidence. This module assembles the
±15s clip the EventDetail player plays:

  pre  — clip_fps frame refs of the analysis window, snapshotted at
         tick time (shared with the FrameBuffer — cheap to hold).
  post — clip_fps frames from T to T+15s. Sleeps
         `settings.clip_post_event_wait_seconds` and then snapshots
         the FrameBuffer (the subscriber-bot keeps writing into it
         continuously between ticks).
  final — pre+post in ONE ffmpeg encode at clip_fps with a single
         contiguous PCM audio track, uploaded as
         `events/{event_id}.mp4` in the `incident-clips` Supabase
         Storage bucket. Single-encode is deliberate (2026-07-27): the
         old reuse-LLM-MP4 + concat assembly had a visible seam and
         per-part audio that never aligned cleanly. Audio restored
         2026-07-28 (one clean mux); CENSORED clips still strip it.

Before ANY upload, the assembled clip goes through the sensitivity
screen (`detection/sensitivity.py` — every clip, every camera, private
rooms included; company baseline, 2026-07-27). A sensitive verdict —
or any screening/censoring failure — means only a CENSORED rendition
(`detection/censor.py`: person-mask blur, audio stripped) can be
uploaded; if even censoring fails the clip is dropped and
`events.clip_suppressed` stamped. Raw footage never reaches storage on
any error path.

After upload completes, `events.clip_storage_key` is set and
`events.has_clip=true` (+ `clip_censored`/`sensitivity_categories`
when censored); Supabase Realtime broadcasts the UPDATE so the
caregiver app's EventDetailView swaps the "Clip processing…"
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
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

from src.db.event import Event
from src.db.patient import Patient
from src.services import storage
from src.services.detection import censor, sensitivity
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.audio_buffer import AudioBuffer
    from src.services.detection.frame_buffer import FrameBuffer
    from src.settings import Settings

log = logging.getLogger(__name__)


async def archive_event(
    *,
    event_id: UUID,
    patient_id: UUID,
    camera_id: UUID,
    pre_frames: list,
    pre_pcm: bytes,
    pre_end_captured_at: float,
    frame_buffer: "FrameBuffer",
    audio_buffer: "AudioBuffer",
    sessionmaker: "async_sessionmaker",
    settings: "Settings",
) -> None:
    """Capture the post-event window, encode it together with the
    pre-event frames in ONE ffmpeg pass at `clip_fps` (audio muxed as a
    single contiguous PCM track — 2026-07-28, audio restored; censored
    clips still strip it), upload the result, and flip `has_clip=true`
    on the events row.

    Errors are caught and logged. The events row stays without a clip
    rather than the whole detector pipeline crashing — caregivers
    still see the alert text either way.

    `pre_frames` is a clip_fps snapshot of the analysis window taken
    at tick time (frame refs shared with the buffer — cheap to hold).
    Single-encode (2026-07-27) replaced the old reuse-the-LLM-MP4 +
    concat + muxed-PCM assembly: one encode at clip_fps has no concat
    seam and no audio, which is deliberate — the muxed audio never
    aligned cleanly and made clips look cheap.

    `pre_end_captured_at` is the `time.monotonic()` timestamp of the
    last frame in the pre-event snapshot. The archive task uses it
    to clock-align the post-event snapshot — without it, the LLM's
    5-10s of think time leaves a hole in wall-clock coverage between
    pre's last frame and post's first frame, which surfaces as a
    visible scene jump at the pre/post boundary.

    `patient_id` is used for the region routing of the upload;
    `camera_id` only for log context on the sensitivity screen (the
    screen runs for EVERY camera — privacy_enabled no longer gates
    clips, it gates live view).
    """
    try:
        await _archive_one(
            event_id=event_id,
            patient_id=patient_id,
            camera_id=camera_id,
            pre_frames=pre_frames,
            pre_pcm=pre_pcm,
            pre_end_captured_at=pre_end_captured_at,
            frame_buffer=frame_buffer,
            audio_buffer=audio_buffer,
            sessionmaker=sessionmaker,
            settings=settings,
        )
    except Exception:  # noqa: BLE001
        log.exception("clip_archival.failed event_id=%s", event_id)


async def _stamp_clip_suppressed(
    sessionmaker: "async_sessionmaker", event_id: UUID,
) -> None:
    """Mark an event as never-getting-a-clip so the apps swap the
    "Generating clip…" placeholder for the no-clip state. Every
    terminal skip path must call this (field bug 2026-07-28: the
    stale-window skip didn't, leaving a forever-spinner)."""
    async with sessionmaker() as session:
        event = await session.get(Event, event_id)
        if event is not None and not event.clip_suppressed:
            event.clip_suppressed = True
            await session.commit()


async def _archive_one(
    *,
    event_id: UUID,
    patient_id: UUID,
    camera_id: UUID,
    pre_frames: list,
    pre_pcm: bytes,
    pre_end_captured_at: float,
    frame_buffer: "FrameBuffer",
    audio_buffer: "AudioBuffer",
    sessionmaker: "async_sessionmaker",
    settings: "Settings",
) -> None:
    # Sleep until wall-clock has advanced enough that the buffer
    # contains `clip_post_event_wait_seconds` of frames AFTER the
    # pre snapshot's last frame. This eliminates the LLM-think-time
    # gap between pre and post — both video and audio snapshots
    # below use `pre_end_captured_at` as the cutoff anchor, so the
    # concat is temporally seamless.
    wait_seconds = settings.clip_post_event_wait_seconds
    target_wall_time = pre_end_captured_at + wait_seconds
    remaining = target_wall_time - time.monotonic()
    post_available = True
    if remaining > 0:
        await asyncio.sleep(remaining)
    elif -remaining > settings.frame_buffer_retention_seconds:
        # Stale-window guard (2026-07-18, softened 2026-07-28): the
        # archive task started so late (slow VLM answer) that the buffer
        # no longer holds the event's post-roll — packaging frames from
        # NOW would attach footage of a DIFFERENT moment. But the
        # PRE-roll refs were pinned at tick time and cover the event
        # itself, so ship a pre-only clip rather than nothing (a fall
        # clip without aftermath beats no clip at all).
        log.warning(
            "clip_archival.stale_window event_id=%s lag=%.1fs — "
            "post-roll lost to eviction; shipping pre-roll-only clip",
            event_id, -remaining,
        )
        post_available = False

    # `now=target_wall_time` makes the snapshot cutoff exactly
    # `pre_end_captured_at` (i.e., the next frame after pre ended),
    # regardless of how long the asyncio runtime actually slept for.
    post_frames = (
        frame_buffer.snapshot_last_seconds(
            wait_seconds, now=target_wall_time,
            # Clip cadence — the buffer retains frame_buffer_target_fps
            # natively; RAPID may hold more, the clip decimates to clip_fps.
            at_fps=settings.clip_fps,
        )
        if post_available
        else []
    )
    if not post_frames and not pre_frames:
        log.warning(
            "clip_archival.no_frames event_id=%s — skipping clip",
            event_id,
        )
        await _stamp_clip_suppressed(sessionmaker, event_id)
        return

    # Single-encode assembly (2026-07-27): pre frames captured at tick
    # time + post frames from the buffer, one video-only ffmpeg pass at
    # clip_fps. No concat (no seam), no muxed PCM (deliberate — audio
    # never aligned cleanly and made clips look cheap). Guard against
    # boundary overlap: post's cutoff anchors at pre_end, but a late
    # retarget can duplicate the boundary frame.
    pre_last = pre_frames[-1].captured_at if pre_frames else float("-inf")
    clip_frames = list(pre_frames) + [
        f for f in post_frames if f.captured_at > pre_last
    ]
    post_pcm = (
        audio_buffer.snapshot_last_seconds(wait_seconds, now=target_wall_time)
        if post_available
        else b""
    )
    clip_pcm = pre_pcm + post_pcm
    try:
        combined_mp4 = await frames_to_mp4(
            clip_frames,
            fps=settings.clip_fps,
            pcm=clip_pcm if clip_pcm else None,
            pcm_sample_rate=audio_buffer.sample_rate,
            pcm_num_channels=audio_buffer.num_channels,
        )
    except (Mp4RemuxError, ValueError) as exc:
        log.warning("clip_archival.clip_encode_failed event_id=%s err=%s", event_id, exc)
        await _stamp_clip_suppressed(sessionmaker, event_id)
        return

    async with sessionmaker() as session:
        patient = await session.get(Patient, patient_id)
        # Capture region inside the session (attribute may expire after close).
        # Missing patient defaults to US — the clip still uploads, just to the
        # default region (a vanished patient is an edge case, not India data).
        patient_region = patient.region if patient is not None else "US"

    # Sensitivity screen + censor at the upload boundary — EVERY clip,
    # EVERY camera (2026-07-27; replaces the transitional private-camera
    # suppression — private rooms now GET clips, censored when needed).
    # See docs/plans/2026-07-27-room-privacy-and-clip-censoring.md.
    resolution = await resolve_clip_for_upload(
        combined_mp4, event_id=event_id, camera_id=camera_id,
        settings=settings,
    )
    # Drop our reference to the raw bytes the moment the decision is
    # made — from here only `resolution.upload_bytes` (raw only when the
    # screen positively cleared it) may leave this function.
    del combined_mp4

    if resolution.upload_bytes is None:
        # Fail-closed terminal: the clip was (or must be treated as)
        # sensitive and even the censor fallback failed. Drop it — no
        # upload, has_clip stays False — and stamp clip_suppressed so
        # the apps show a "no clip" placeholder instead of a forever
        # "Generating clip…". The TEXT event still reaches the caregiver.
        log.warning(
            "clip_archival.censor_drop event_id=%s camera_id=%s — clip "
            "discarded (censoring failed; raw never uploaded)",
            event_id, camera_id,
        )
        await _stamp_clip_suppressed(sessionmaker, event_id)
        return

    final_key = await storage.upload_clip(
        event_id=event_id, mp4_bytes=resolution.upload_bytes,
        settings=settings, region=patient_region,
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
        if resolution.censored:
            event.clip_censored = True
            event.sensitivity_categories = resolution.categories or None
        await session.commit()

    log.info(
        "clip_archival.done event_id=%s key=%s bytes=%d censored=%s",
        event_id, final_key, len(resolution.upload_bytes),
        resolution.censored,
    )


def _screening_active(settings: Settings) -> bool:
    """Whether the sensitivity screen runs for this process.

    `sensitivity_screening_enabled=false` is an internal/dev kill-switch
    ONLY: it is honored solely in env=local/test (mirroring how
    internal_service_key fail-closes by env). In staging/prod a false
    value is IGNORED — screening stays on — and we log loudly so the
    misconfiguration is visible instead of silently shipping raw clips.
    """
    if settings.sensitivity_screening_enabled:
        return True
    if settings.env in ("local", "test"):
        return False
    log.warning(
        "sensitivity_screening_enabled=false ignored outside local/test "
        "(env=%s) — screening stays ON",
        settings.env,
    )
    return True


@dataclass(frozen=True)
class ClipResolution:
    """What may leave the box for one clip. ``upload_bytes is None``
    means DROP (stamp clip_suppressed); otherwise upload exactly those
    bytes. ``categories`` is None when censoring happened without a
    trusted verdict (screen failure → full-frame blur)."""

    upload_bytes: bytes | None
    censored: bool
    categories: list[str] | None


async def resolve_clip_for_upload(
    mp4: bytes,
    *,
    event_id: UUID,
    camera_id: UUID,
    settings: "Settings",
) -> ClipResolution:
    """Run the screen-then-censor decision for one assembled clip.

    Fail-closed matrix (docs/plans/2026-07-27-room-privacy-and-clip-
    censoring.md):

      screen → not sensitive   → upload the clip as-is (video + audio)
      screen → sensitive       → censor (person-mask blur, audio
                                 stripped; a no-person frame is blurred
                                 full-frame inside `censor_clip`)
      screen error/timeout/
      ambiguous                → censor FULL-FRAME (no verdict → the
                                 only safe treatment is everything)
      censor error (both
      tiers)                   → DROP the clip (upload_bytes=None)

    The only path that returns the raw bytes is a positive, well-formed
    "not sensitive" verdict AT OR ABOVE the confidence floor
    (`settings.sensitivity_confidence_floor`) — a clearance the model
    itself barely believes is doubt, and doubt censors.
    `sensitivity_screening_enabled=false` (internal/dev kill-switch,
    mirrors auto_privacy_mode) bypasses the whole pipeline, but ONLY in
    env=local/test — see `_screening_active`.
    """
    if not _screening_active(settings):
        return ClipResolution(upload_bytes=mp4, censored=False, categories=None)

    verdict: sensitivity.ScreenResult | None
    try:
        verdict = await sensitivity.screen_clip(mp4, settings=settings)
    except Exception:  # noqa: BLE001 — ANY screen failure fails closed
        log.exception(
            "clip_archival.screen_failed event_id=%s camera_id=%s — "
            "censoring full-frame (fail closed)",
            event_id, camera_id,
        )
        verdict = None

    if verdict is not None and not verdict.sensitive:
        if verdict.confidence >= settings.sensitivity_confidence_floor:
            return ClipResolution(upload_bytes=mp4, censored=False, categories=None)
        # Low-confidence clearance: sensitive=false the model itself
        # barely believes is not a clearance. Doubt → censor (falls
        # through to the person-mask path below with the verdict kept).
        log.info(
            "clip_archival.low_confidence_clear event_id=%s camera_id=%s "
            "confidence=%.2f floor=%.2f — censoring",
            event_id, camera_id, verdict.confidence,
            settings.sensitivity_confidence_floor,
        )

    # No trusted verdict → we don't know WHAT is sensitive, so blur
    # everything. With a verdict, person-mask censoring applies (and
    # falls back to full-frame internally).
    full_frame = verdict is None
    categories = list(verdict.categories) if verdict is not None else None
    log.info(
        "clip_archival.censoring event_id=%s camera_id=%s full_frame=%s "
        "categories=%s",
        event_id, camera_id, full_frame, categories,
    )
    try:
        censored_bytes = await _censor_bytes(mp4, full_frame=full_frame)
    except Exception:  # noqa: BLE001 — censor failure → drop, never raw
        log.exception(
            "clip_archival.censor_failed event_id=%s camera_id=%s — "
            "dropping clip (raw never uploaded)",
            event_id, camera_id,
        )
        return ClipResolution(upload_bytes=None, censored=True, categories=categories)
    return ClipResolution(
        upload_bytes=censored_bytes, censored=True, categories=categories,
    )


def _scratch_dir() -> str:
    """Dedicated temp dir for clip tempfiles (raw + censored + concat).

    Isolating clip bytes in one directory makes the startup sweep below
    safe: nothing else writes here, so everything found at boot is a
    stranded file from a hard-killed process and can be deleted.
    """
    path = os.path.join(tempfile.gettempdir(), "nemara-clip-scratch")
    os.makedirs(path, exist_ok=True)
    return path


def sweep_scratch_dir() -> int:
    """Delete every file in the clip scratch dir. Called once at app
    startup, before any clip work begins: normal/error paths unlink
    their own tempfiles, so anything present at boot is raw or censored
    clip bytes stranded by a SIGKILL/OOM mid-censor. Best-effort;
    returns the number of files removed."""
    removed = 0
    try:
        root = _scratch_dir()
        for name in os.listdir(root):
            try:
                os.unlink(os.path.join(root, name))
                removed += 1
            except OSError:
                continue
    except OSError:
        log.warning("clip_archival.scratch_sweep failed", exc_info=True)
        return removed
    if removed:
        log.warning(
            "clip_archival.scratch_sweep removed %d stranded clip "
            "tempfile(s) from a previous hard-killed process",
            removed,
        )
    return removed


async def _censor_bytes(mp4: bytes, *, full_frame: bool) -> bytes:
    """Round-trip the in-memory clip through `censor.censor_clip` via
    tempfiles. The RAW tempfile (and the censored one) are unlinked in
    the finally — no raw clip bytes persist on disk past this call,
    success or failure."""
    in_path: str | None = None
    out_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_scratch_dir()) as f:
            f.write(mp4)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_scratch_dir()) as f:
            out_path = f.name
        await censor.censor_clip(in_path, out_path, full_frame=full_frame)
        with open(out_path, "rb") as f:
            censored_bytes = f.read()
        if not censored_bytes:
            raise censor.CensorError("censor produced 0 bytes")
        return censored_bytes
    finally:
        for path in (in_path, out_path):
            if path is None:
                continue
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


