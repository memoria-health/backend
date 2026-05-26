"""Burn a green bounding box around the patient on archived event clips.

Display-only post-processing. Runs AFTER `clip_archival.concat_mp4s`
has assembled the ±15s clip and BEFORE the bytes are uploaded to
Supabase Storage. The detector pipeline is untouched — this lives on
the archive path only so it doesn't pay any cost on ticks that don't
fire an event.

Per-frame Re-ID would be ~30s of inference for a typical 30s clip at
native fps. To keep that bounded we sample inference at
`reid_clip_annotate_fps` (default 2 fps), carry the most recent
bbox forward to in-between frames, and decay it back to "no box"
after `reid_clip_annotate_carry_seconds` (1.0s) without a fresh
detection. The result: smooth annotation across the clip at <10s of
extra work per archive.

Failure-mode: any error here returns the original (un-annotated) MP4
so we never lose a clip just because the post-processing crashed.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import cv2
import numpy as np

from src.repos.face_embedding_repo import has_ready_enrollment, list_ready_for_patient_unscoped
from src.services.detection.frame_annotate import _BOX_COLOR_BGR, _BOX_THICKNESS
from src.services.reid import embeddings as reid_embeddings
from src.services.reid.embeddings import embedding_from_bytes, topk_mean_cosine_match

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AnnotateConfig:
    sample_fps: float
    carry_seconds: float
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
    face. Audio is preserved.

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
        sample_fps=settings.reid_clip_annotate_fps,
        carry_seconds=settings.reid_clip_annotate_carry_seconds,
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


def _annotate_clip_sync(
    mp4_bytes: bytes,
    gallery: np.ndarray,
    cfg: _AnnotateConfig,
) -> bytes:
    """The cv2 + ffmpeg heavy lifting. Runs in a thread so we don't
    block the event loop.
    """
    workdir = tempfile.mkdtemp(prefix="nemara-annotate-")
    in_path = os.path.join(workdir, "input.mp4")
    annotated_path = os.path.join(workdir, "annotated.mp4")
    final_path = os.path.join(workdir, "final.mp4")
    try:
        with open(in_path, "wb") as f:
            f.write(mp4_bytes)

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            log.warning("clip_annotate: cv2 cannot open input")
            return mp4_bytes

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width == 0 or height == 0:
            log.warning("clip_annotate: zero-sized input %dx%d", width, height)
            return mp4_bytes

        # Sample inference at most every `sample_interval` frames.
        sample_interval = max(1, int(round(fps / cfg.sample_fps)))
        carry_frames = int(round(fps * cfg.carry_seconds))

        # mp4v keeps quality decent and is universally readable. The
        # final ffmpeg mux re-encodes with libx264 for delivery, so the
        # intermediate codec is just a working format.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            log.warning("clip_annotate: cv2 cannot open writer")
            cap.release()
            return mp4_bytes

        last_bbox: tuple[int, int, int, int] | None = None
        frames_since_bbox = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % sample_interval == 0:
                bbox = _detect_patient_bbox(
                    frame, gallery, cfg.match_threshold_base, cfg.insightface_model,
                )
                if bbox is not None:
                    last_bbox = bbox
                    frames_since_bbox = 0
                else:
                    frames_since_bbox += sample_interval
            else:
                frames_since_bbox += 1

            if last_bbox is not None and frames_since_bbox <= carry_frames:
                x0, y0, x1, y1 = last_bbox
                cv2.rectangle(frame, (x0, y0), (x1, y1), _BOX_COLOR_BGR, _BOX_THICKNESS)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        # Mux: take the annotated (video-only) MP4 plus the original
        # audio stream and produce the final delivery file. -c copy
        # everywhere except where we have to re-encode, and we don't
        # — annotated MP4 is already video-only with no audio to copy.
        # Original mp4 supplies audio via the second input.
        proc = _ffmpeg_mux(annotated_path, in_path, final_path)
        if proc != 0:
            log.warning("clip_annotate: ffmpeg mux exit=%d, returning raw clip", proc)
            return mp4_bytes

        with open(final_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _detect_patient_bbox(
    frame_bgr: np.ndarray,
    gallery: np.ndarray,
    match_threshold_base: float,
    model_name: str,
) -> tuple[int, int, int, int] | None:
    """Run SCRFD + ArcFace on a single BGR frame, return the bbox of
    the highest-scoring face that matches the gallery above the
    adaptive threshold. None if no face matches.
    """
    # InsightFace expects BGR ndarray, same as what cv2 produces.
    app = reid_embeddings.get_face_app(model_name)
    faces = app.get(frame_bgr)
    if not faces:
        return None

    best_score = -1.0
    best_bbox: tuple[int, int, int, int] | None = None
    for face in faces:
        if face.embedding is None:
            continue
        emb = face.embedding.astype(np.float32)
        # ArcFace embeddings are L2-normalized by the model itself, but
        # be defensive in case a version change drops that.
        norm = float(np.linalg.norm(emb))
        if norm > 0:
            emb = emb / norm

        match = topk_mean_cosine_match(emb, gallery)
        x1, y1, x2, y2 = (float(v) for v in face.bbox)
        face_area = max(1.0, (x2 - x1) * (y2 - y1))
        threshold = reid_embeddings.effective_match_threshold(
            match_threshold_base, float(face.det_score), face_area,
        )
        if match.score >= threshold and match.score > best_score:
            best_score = match.score
            best_bbox = (int(x1), int(y1), int(x2), int(y2))
    return best_bbox


def _ffmpeg_mux(video_path: str, audio_source_path: str, out_path: str) -> int:
    """Take an annotated video-only MP4 and the original MP4 (which
    may have audio), produce an output with the annotated video and
    the original audio. Re-encodes video with libx264 ultrafast for a
    universally-playable output; copies audio if present.

    Returns the ffmpeg exit code.
    """
    import subprocess
    proc = subprocess.run(
        [
            "ffmpeg", "-y",
            "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-i", audio_source_path,
            "-map", "0:v:0",
            "-map", "1:a:0?",  # `?` makes the audio mapping optional
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            out_path,
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        log.warning(
            "clip_annotate: ffmpeg mux stderr=%s",
            proc.stderr.decode("utf-8", errors="replace")[:500],
        )
    return proc.returncode
