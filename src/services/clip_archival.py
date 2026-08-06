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
  final — pre+post in ONE wall-clock-faithful encode
         (`frames_to_vfr_mp4`: each frame keeps its real captured_at
         timing, so the clip plays the true window duration and audio
         aligns 1:1) with a single contiguous PCM track, uploaded as
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

from sqlalchemy import text as sa_text

from src.db.event import Event
from src.db.patient import Patient
from src.services import storage
from src.services.detection import censor, pod_masks, sensitivity
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_vfr_mp4

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.audio_buffer import AudioBuffer
    from src.services.detection.frame_buffer import FrameBuffer
    from src.settings import Settings

log = logging.getLogger(__name__)

# Strong refs for in-flight operator-clip tasks (see archive_event).
_GATE_CLIP_TASKS: set[asyncio.Task] = set()


async def archive_event(
    *,
    event_id: UUID,
    patient_id: UUID,
    camera_id: UUID,
    pre_frames: list,
    pre_pcm: bytes,
    # Measured capture time of the first PCM chunk. None = unknown, in
    # which case audio is muxed without shift rather than guessed at.
    pre_pcm_start: float | None = None,
    pre_end_captured_at: float,
    frame_buffer: "FrameBuffer",
    audio_buffer: "AudioBuffer",
    sessionmaker: "async_sessionmaker",
    settings: "Settings",
) -> None:
    """Capture the post-event window, encode it together with the
    pre-event frames in ONE wall-clock-faithful ffmpeg pass
    (`frames_to_vfr_mp4` — 2026-07-28: real per-frame timing, so the
    clip plays the true window duration and the contiguous PCM track
    aligns 1:1; censored clips still strip audio), upload the result,
    and flip `has_clip=true` on the events row.

    Errors are caught and logged. The events row stays without a clip
    rather than the whole detector pipeline crashing — caregivers
    still see the alert text either way.

    `pre_frames` is a clip_fps snapshot of the analysis window taken
    at tick time (frame refs shared with the buffer — cheap to hold).

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
    # EVENT-GATE FAST PATH — runs as a BACKGROUND TASK, never inline.
    #
    # When the event is held for an operator verdict, encode and upload the
    # PRE-ROLL alone — the footage the model actually judged — so the
    # operator sees video in seconds rather than after the post-roll. They
    # have 60 seconds to decide; a clip that lands at T+20s is too late.
    #
    # Why a task and not an await: masking takes ~14s of single-threaded
    # YOLO, while `_archive_one` below has only
    # `frame_buffer_retention_seconds - clip_post_event_wait_seconds` of
    # slack (2s today) before the post-roll it needs is evicted from the
    # ring buffer. Awaiting the mask here cost the CAREGIVER their
    # post-roll — observed 2026-08-05, event 2bdf15b5 logged
    # `stale_window lag=18.7s` at the exact instant the mask finished and
    # the family got an 11s clip instead of 26s. The operator's clip must
    # never be able to degrade the caregiver's.
    #
    # The pre-roll arrives as FRAMES and is only encoded further down for
    # the combined clip, so this encodes its own. (The original version of
    # this block referenced a `pre_event_mp4` argument that has never
    # existed on this function; it raised NameError the first time it ever
    # ran, on 2026-08-05, because before then no patient had the gate on.)
    #
    # The operator's clip is ALWAYS masked, unconditionally — it does not
    # use `resolve_clip_for_upload`, which asks a model whether this
    # particular clip is sensitive. That is a reasonable question for the
    # family who live in the house and the wrong one for a reviewer who
    # sees every household in the product. No screen means no verdict to
    # get wrong and no path that uploads unmasked footage. It also keeps
    # Gemini off the operator's path entirely, so no 503 can blur, delay
    # or drop the one clip that has 60 seconds to arrive.
    async def _gate_fast_path() -> None:
        try:
            async with sessionmaker() as s:
                held = (await s.execute(
                    sa_text(
                        "select 1 from events where id = :id "
                        "and gate_state = 'pending'"),
                    {"id": str(event_id)})).first()
            if not held:
                return
            async with sessionmaker() as s:
                patient = await s.get(Patient, patient_id)
                patient_region = patient.region if patient is not None else "US"

            # Pre-roll only: end at the last pre-event frame and mux just
            # the pre-roll PCM. Same VFR encode as the combined clip, so
            # timing and audio alignment behave identically. `pre_frames`
            # is a snapshot list holding its own references, so buffer
            # eviction cannot pull frames out from under this task while
            # it runs alongside the post-roll capture.
            pre_mp4 = await frames_to_vfr_mp4(
                pre_frames,
                fps=settings.clip_fps,
                end_at=pre_end_captured_at,
                pcm=pre_pcm if pre_pcm else None,
                pcm_sample_rate=audio_buffer.sample_rate,
                pcm_num_channels=audio_buffer.num_channels,
                pcm_starts_at=pre_pcm_start,
            )
            # Green silhouette (CENSOR_MODE_RESEARCH) rather than
            # pixelation: it reads as a deliberate redaction at a glance
            # on a phone, and posture, position and motion — the things
            # that distinguish a fall from someone sitting down — survive
            # it. Audio is KEPT: for "possible calling for help" the sound
            # is the evidence, and masking the picture does not touch it.
            masked = await _censor_bytes(
                pre_mp4, full_frame=False, keep_audio=True,
                mode=censor.CENSOR_MODE_RESEARCH,
                camera_id=camera_id,
                clip_start=pod_masks.monotonic_to_epoch(
                    pre_frames[0].captured_at,
                ),
            )
            pre_key = await storage.upload_clip(
                event_id=event_id, mp4_bytes=masked,
                settings=settings, sub_key="gate-pre", region=patient_region)
            async with sessionmaker() as s:
                await s.execute(
                    sa_text("update events set gate_clip_key = :k where id = :id"),
                    {"k": pre_key, "id": str(event_id)})
                await s.commit()
            log.info(
                "event_gate.fast_path_clip event=%s key=%s masked=green bytes=%d",
                event_id, pre_key, len(masked),
            )
        except Exception:  # noqa: BLE001 — best-effort, never fatal
            log.exception("event_gate.fast_path_failed event=%s", event_id)

    # Keep a strong reference. asyncio only holds a weak one, so a
    # fire-and-forget task can be collected mid-flight — the operator's
    # clip would then silently never appear, indistinguishable from a slow
    # upload. Same guard the rest of this codebase uses (see
    # camera_ingest._bg_tasks, livekit_webhooks, room_classifier).
    _task = asyncio.create_task(
        _gate_fast_path(), name=f"event-gate-clip-{event_id}")
    _GATE_CLIP_TASKS.add(_task)
    _task.add_done_callback(_GATE_CLIP_TASKS.discard)

    try:
        await _archive_one(
            event_id=event_id,
            patient_id=patient_id,
            camera_id=camera_id,
            pre_frames=pre_frames,
            pre_pcm=pre_pcm,
            pre_pcm_start=pre_pcm_start,
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
    # Measured capture time of the first PCM chunk. None = unknown, in
    # which case audio is muxed without shift rather than guessed at.
    pre_pcm_start: float | None = None,
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

    # Single-encode assembly: pre frames captured at tick time + post
    # frames from the buffer, one ffmpeg pass. Guard against boundary
    # overlap: post's cutoff anchors at pre_end, but a late retarget
    # can duplicate the boundary frame.
    pre_last = pre_frames[-1].captured_at if pre_frames else float("-inf")
    clip_frames = list(pre_frames) + [
        f for f in post_frames if f.captured_at > pre_last
    ]
    post_pcm, post_pcm_start = (
        audio_buffer.snapshot_window(wait_seconds, now=target_wall_time)
        if post_available
        else (b"", None)
    )
    clip_pcm = pre_pcm + post_pcm
    # Where the combined track actually STARTS: the pre-roll's measured
    # start when there is pre-roll audio, else the post-roll's. Never
    # derived from the byte length — a buffer gap would then shift the
    # whole track against the video.
    clip_pcm_start = pre_pcm_start if pre_pcm else post_pcm_start
    # Wall-clock-faithful VFR encode (2026-07-28): each frame keeps its
    # real captured_at as display timing, so the clip plays the true
    # window duration and audio aligns 1:1 — no fixed-fps time
    # compression (the old encode played frame-count/clip_fps seconds:
    # short, sped-up clips that teleported across still stretches).
    clip_ends_at = target_wall_time if post_available else pre_end_captured_at
    try:
        combined_mp4 = await frames_to_vfr_mp4(
            clip_frames,
            fps=settings.clip_fps,
            end_at=clip_ends_at,
            pcm=clip_pcm if clip_pcm else None,
            pcm_sample_rate=audio_buffer.sample_rate,
            pcm_num_channels=audio_buffer.num_channels,
            pcm_starts_at=clip_pcm_start,
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
        # First frame of the ASSEMBLED clip, converted to epoch — the
        # censor indexes the pod's masks by capture time, and the clip's
        # own timeline starts here.
        clip_start=pod_masks.monotonic_to_epoch(clip_frames[0].captured_at),
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
    clip_start: float,
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
    # The intimate-situation carve-out (Aryaman 2026-07-29): audio stays
    # on every other censored clip, because the reason we kept it is
    # that a cry for help must be audible. Here the sound is itself the
    # exposure, so a blurred picture with the audio intact would defeat
    # the censor.
    #
    # A MISSING verdict does NOT silence. The no-verdict path is the
    # screen-failure fallback (timeout, low confidence, API error) and
    # it fires on ordinary clips — a fall whose screen timed out would
    # lose the shout, which is precisely the case the audio rule exists
    # to protect. Silence requires a POSITIVE intimacy finding.
    keep_audio = not (set(categories or ()) & sensitivity.SILENT_CATEGORIES)
    log.info(
        "clip_archival.censoring event_id=%s camera_id=%s full_frame=%s "
        "categories=%s keep_audio=%s",
        event_id, camera_id, full_frame, categories, keep_audio,
    )
    try:
        censored_bytes = await _censor_bytes(
            mp4, full_frame=full_frame, keep_audio=keep_audio,
            camera_id=camera_id, clip_start=clip_start,
        )
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


async def _censor_bytes(
    mp4: bytes, *, full_frame: bool, keep_audio: bool = True,
    mode: str = censor.CENSOR_MODE_PIXELATE,
    camera_id: UUID, clip_start: float,
) -> bytes:
    """Round-trip the in-memory clip through `censor.censor_clip` via
    tempfiles. The RAW tempfile (and the censored one) are unlinked in
    the finally — no raw clip bytes persist on disk past this call,
    success or failure.

    `clip_start` is the EPOCH capture time of the clip's first frame.
    Frame buffers stamp `captured_at` on the MONOTONIC clock and the
    pod's masks are stamped in epoch, so callers convert once here (via
    `pod_masks.monotonic_to_epoch`) and the censor never has to know two
    clocks exist.
    """
    in_path: str | None = None
    out_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_scratch_dir()) as f:
            f.write(mp4)
            in_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=_scratch_dir()) as f:
            out_path = f.name
        await censor.censor_clip(
            in_path, out_path, full_frame=full_frame, keep_audio=keep_audio,
            mode=mode, camera_id=str(camera_id), clip_start=clip_start,
        )
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


