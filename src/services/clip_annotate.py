"""Burn a green bounding box around the patient on archived event clips.

Display-only post-processing. Runs AFTER `clip_archival.concat_mp4s`
has assembled the ±15s clip and BEFORE the bytes are uploaded to
Supabase Storage. The detector pipeline is untouched.

Approach (matches `nemara-reid-experiment`):
  - ffmpeg decodes the input clip to a directory of high-quality JPEGs
    at the source's native fps. No frame skipping, no re-timing.
  - Every JPEG is processed by `BodyReIDPipeline` in temporal order so
    BoT-SORT can carry track IDs across frames. We draw the locked
    PATIENT track's bbox on each frame as it appears.
  - ffmpeg re-encodes the JPEGs into the final MP4 in a SINGLE pass
    with libx264 (`-crf 18 -preset slow`) plus audio passthrough from
    the original input. No double-encode, no cv2.VideoWriter — the
    bytes only round-trip once through an encoder.

Per-frame inference adds noticeable wall-clock to the archive path
(~25–35s on CPU for a 30s clip at 15 fps). Archive is async and
fire-and-forget, so the clip just becomes viewable a few seconds
later than before. The accuracy upgrade matches what the experiment
produces.

Failure-mode: any error returns the original (un-annotated) MP4 so
the caregiver never loses a clip to a display-layer issue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import cv2
import numpy as np

from src.repos.face_embedding_repo import has_ready_enrollment, list_ready_for_patient_unscoped
from src.services.detection.frame_annotate import _BOX_COLOR_BGR, _BOX_THICKNESS
from src.services.reid import embeddings as reid_embeddings
from src.services.reid.body_pipeline import BodyReIDPipeline, PipelineConfig
from src.services.reid.embeddings import embedding_from_bytes

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AnnotateConfig:
    yolo_confidence: float
    face_check_interval: int
    hits_to_lock: int
    face_min_det_score: float
    face_checks_for_other: int
    match_threshold_base: float
    insightface_model: str


async def annotate_clip_for_display(
    mp4_bytes: bytes,
    *,
    patient_id: UUID,
    sessionmaker: "async_sessionmaker",
    settings: "Settings",
) -> bytes:
    """Return a new MP4 with a green rectangle tracking the patient's
    body throughout the clip. Audio is preserved.

    Returns the original bytes unchanged if:
      - annotation is disabled in settings
      - the patient has no ready enrollment (nothing to match against)
      - any step crashes (decode, inference, encode, mux)

    Never raises — the caller (clip_archival) treats this as
    best-effort post-processing.
    """
    if not settings.reid_clip_annotation_enabled:
        return mp4_bytes

    try:
        async with sessionmaker() as session:
            if not await has_ready_enrollment(session, patient_id):
                return mp4_bytes
            gallery = await _load_gallery(session, patient_id)
    except Exception:
        log.exception("clip_annotate: gallery load failed patient=%s", patient_id)
        return mp4_bytes

    if gallery.size == 0:
        return mp4_bytes

    cfg = _AnnotateConfig(
        yolo_confidence=settings.reid_yolo_confidence,
        face_check_interval=settings.reid_face_check_interval,
        hits_to_lock=settings.reid_hits_to_lock,
        face_min_det_score=settings.reid_face_min_det_score,
        face_checks_for_other=settings.reid_face_checks_for_other,
        match_threshold_base=settings.reid_match_threshold_base,
        insightface_model=settings.reid_insightface_model,
    )

    try:
        return await asyncio.to_thread(
            _annotate_clip_sync, mp4_bytes, gallery, cfg,
        )
    except Exception:
        log.exception("clip_annotate: failed patient=%s, returning raw clip", patient_id)
        return mp4_bytes


async def _load_gallery(session, patient_id: UUID) -> np.ndarray:
    """Stacked (N, 512) float32 array of ready embeddings, or an empty
    array if none. Matches the shape `topk_mean_cosine_match` expects.
    """
    rows = await list_ready_for_patient_unscoped(session, patient_id)
    vectors: list[np.ndarray] = []
    for row in rows:
        if not row.embedding:
            continue
        try:
            vectors.append(embedding_from_bytes(row.embedding))
        except ValueError:
            continue
    if not vectors:
        return np.empty((0, reid_embeddings.EMBEDDING_DIM), dtype=np.float32)
    return np.stack(vectors)


# ── Sync pipeline (ffmpeg + cv2 + body Re-ID) ─────────────────────


def _annotate_clip_sync(
    mp4_bytes: bytes,
    gallery: np.ndarray,
    cfg: _AnnotateConfig,
) -> bytes:
    """Decode → per-frame inference + draw → re-encode. Runs in a
    thread so we don't block the event loop."""
    workdir = tempfile.mkdtemp(prefix="nemara-annotate-")
    in_path = os.path.join(workdir, "input.mp4")
    frames_dir = os.path.join(workdir, "frames")
    final_path = os.path.join(workdir, "final.mp4")
    os.makedirs(frames_dir, exist_ok=True)
    try:
        with open(in_path, "wb") as f:
            f.write(mp4_bytes)

        fps = _probe_fps(in_path)
        if fps is None or fps <= 0:
            log.warning("clip_annotate: could not determine input fps")
            return mp4_bytes

        # Step 1: extract every frame as a high-quality JPEG. -q:v 2
        # is the second-best quality level (1=best, 31=worst); we want
        # the inference inputs to be visually close to the source so
        # YOLO + SCRFD aren't hampered by compression noise the source
        # didn't have.
        extract_rc = _ffmpeg_extract_frames(in_path, frames_dir)
        if extract_rc != 0:
            log.warning(
                "clip_annotate: ffmpeg extract exit=%d, returning raw clip",
                extract_rc,
            )
            return mp4_bytes

        jpegs = sorted(
            os.path.join(frames_dir, name)
            for name in os.listdir(frames_dir)
            if name.endswith(".jpg")
        )
        if not jpegs:
            log.warning("clip_annotate: no frames produced by ffmpeg")
            return mp4_bytes

        # Step 2: per-frame inference. The pipeline carries BoT-SORT
        # state across frames so the locked PATIENT track stays bound
        # through brief face occlusions / fast motion.
        pipeline = BodyReIDPipeline(
            gallery=gallery,
            config=PipelineConfig(
                sample_fps=max(1, int(round(fps))),
                yolo_confidence=cfg.yolo_confidence,
                face_check_interval=cfg.face_check_interval,
                hits_to_lock=cfg.hits_to_lock,
                face_min_det_score=cfg.face_min_det_score,
                face_checks_for_other=cfg.face_checks_for_other,
                match_threshold_base=cfg.match_threshold_base,
                insightface_model=cfg.insightface_model,
            ),
        )

        for frame_idx, jpeg_path in enumerate(jpegs):
            img = cv2.imread(jpeg_path)
            if img is None:
                continue
            result = pipeline.process_frame(img, frame_idx)
            if result.patient_bbox is not None:
                x0, y0, x1, y1 = result.patient_bbox
                cv2.rectangle(
                    img,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    _BOX_COLOR_BGR,
                    _BOX_THICKNESS,
                )
                # Save back. JPEG quality 95 keeps the source detail
                # intact for the re-encode step that follows.
                cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            # Frames without a patient lock are left as the source JPEG;
            # nothing to re-save, avoids unnecessary disk I/O.

        # Step 3: re-encode JPEGs + original audio into final MP4 in
        # one libx264 pass. -crf 18 is visually lossless for typical
        # phone-camera content; -preset slow gives the encoder time
        # to find good motion vectors. Audio is copied verbatim from
        # the original input.
        mux_rc = _ffmpeg_assemble(
            frames_dir=frames_dir,
            audio_source_path=in_path,
            out_path=final_path,
            fps=fps,
        )
        if mux_rc != 0:
            log.warning("clip_annotate: ffmpeg assemble exit=%d", mux_rc)
            return mp4_bytes

        with open(final_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ── ffmpeg helpers ────────────────────────────────────────────────


def _probe_fps(path: str) -> float | None:
    """Return the average frame rate of `path`'s first video stream.
    None if ffprobe isn't available or the stream can't be read.
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "json",
                path,
            ],
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        log.exception("clip_annotate: ffprobe failed for %s", path)
        return None
    try:
        info = json.loads(proc.stdout)
        rate = info["streams"][0]["avg_frame_rate"]
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
    # avg_frame_rate is "num/den". Both 0 means unknown.
    try:
        num_s, den_s = rate.split("/")
        num = float(num_s)
        den = float(den_s)
    except (ValueError, AttributeError):
        return None
    if den == 0:
        return None
    return num / den


def _ffmpeg_extract_frames(in_path: str, frames_dir: str) -> int:
    """Extract every frame as a JPEG. -q:v 2 keeps the JPEGs close to
    visually lossless so the inference inputs aren't degraded by
    compression noise the source didn't have.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-q:v", "2",
            os.path.join(frames_dir, "%06d.jpg"),
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        log.warning(
            "clip_annotate: extract stderr=%s",
            proc.stderr.decode("utf-8", errors="replace")[:500],
        )
    return proc.returncode


def _ffmpeg_assemble(
    *,
    frames_dir: str,
    audio_source_path: str,
    out_path: str,
    fps: float,
) -> int:
    """Reassemble the (possibly annotated) JPEG sequence into an MP4,
    muxing the original audio stream alongside. Single libx264
    encode, no intermediate cv2 VideoWriter — preserves the source
    quality much better than the previous mp4v→libx264 round-trip.

    `?` on the audio mapping makes it optional; if the original clip
    had no audio (silent buffer fallback), the output is video-only
    and ffmpeg doesn't error.
    """
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "%06d.jpg"),
            "-i", audio_source_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        log.warning(
            "clip_annotate: assemble stderr=%s",
            proc.stderr.decode("utf-8", errors="replace")[:500],
        )
    return proc.returncode
