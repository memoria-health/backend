"""InsightFace SCRFD + ArcFace wrapper and gallery matching.

InsightFace's pipeline does these steps before producing an embedding:

  1. SCRFD detects the face bounding box + 5 facial landmarks
  2. The face is aligned + cropped to a 112×112 region using the
     landmarks. Everything outside the face — shoulders, clothing,
     background — is discarded
  3. ArcFace processes only the aligned crop and produces a 512-dim
     L2-normalized embedding

So embeddings encode face-only structure, not clothing or background.

The model bundle is ~280 MB. Loaded lazily on first use and cached at
module scope. The Dockerfile pre-downloads it at build time so the
first tick after deploy doesn't pay the download cost.
"""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# The matching math + embedding codec moved to the identity package
# (gpu/identity/matching.py, 2026-07-27 consolidation): the format is
# an identity-domain contract shared through the DB. Re-exported here
# so backend enrollment + existing importers keep one import path.
from identity.matching import (  # noqa: E402,F401
    EMBEDDING_DIM,
    FaceHit,
    MatchResult,
    effective_match_threshold,
    embedding_from_bytes,
    embedding_to_bytes,
    topk_mean_cosine_match,
)

# 512-dim ArcFace embedding. Kept as a constant so callers (repo,
# tests, the adaptive learning harvester) don't hard-code the literal.
_init_lock = threading.Lock()
_face_app = None


def get_face_app(model_name: str):
    """Lazily build and cache the InsightFace FaceAnalysis pipeline.

    Called inside `asyncio.to_thread` from `embed_image_bytes` so the
    backend event loop doesn't block on the ~3s init cost.
    """
    global _face_app
    if _face_app is not None:
        return _face_app
    with _init_lock:
        if _face_app is not None:
            return _face_app
        # Local import: insightface drags in a lot at import time and
        # we don't want to pay it during unit tests that don't touch
        # the embedding path.
        from insightface.app import FaceAnalysis  # type: ignore
        log.info("reid.embeddings.init model=%s", model_name)
        app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        _face_app = app
        return _face_app


async def embed_image_bytes(
    image_bytes: bytes,
    *,
    model_name: str = "buffalo_l",
) -> FaceHit | None:
    """Detect + embed the highest-confidence face in `image_bytes`.

    Returns None when SCRFD finds no face. When multiple faces are
    present (rare in selfies, possible in harvested wall-cam crops),
    returns the one with the highest det_score. The caller decides
    what to do with the result; this function makes no judgment about
    quality beyond "there was something detectable."

    Runs the heavy work in a thread so the event loop stays
    responsive — InsightFace itself is sync numpy + onnxruntime.
    """
    return await asyncio.to_thread(_embed_image_bytes_sync, image_bytes, model_name)


def _embed_image_bytes_sync(image_bytes: bytes, model_name: str) -> FaceHit | None:
    """Sync inner of `embed_image_bytes`. Kept separate so tests can
    monkeypatch the thread-bound work without dealing with the async
    wrapper."""
    import cv2  # local: heavy, only needed on the embedding path

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        log.warning("reid.embed.decode_failed bytes=%d", len(image_bytes))
        return None

    app = get_face_app(model_name)
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: float(f.det_score))
    return FaceHit(
        embedding=best.normed_embedding.astype(np.float32),
        det_score=float(best.det_score),
        bbox=tuple(float(v) for v in best.bbox),
    )


def embed_in_box(
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    model_name: str = "buffalo_l",
) -> list[FaceHit]:
    """Detect + embed every face inside a person bounding box.

    Used by the body-Re-ID pipeline: YOLO + BoT-SORT give us a person
    box; we then look for the face only inside that crop. Cheaper than
    scanning the whole frame for faces, and avoids cross-tagging a
    face from one person to another's track when two people stand
    close together.

    Returned bboxes are translated back into the parent frame's
    coordinates so the caller can draw / log against the original
    frame.
    """
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1i = max(0, int(x1))
    y1i = max(0, int(y1))
    x2i = min(w, int(x2))
    y2i = min(h, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return []
    crop = frame_bgr[y1i:y2i, x1i:x2i]
    app = get_face_app(model_name)
    faces = app.get(crop)
    out: list[FaceHit] = []
    for f in faces:
        fx1, fy1, fx2, fy2 = (float(v) for v in f.bbox)
        out.append(FaceHit(
            embedding=f.normed_embedding.astype(np.float32),
            det_score=float(f.det_score),
            bbox=(fx1 + x1i, fy1 + y1i, fx2 + x1i, fy2 + y1i),
        ))
    return out


# ─── Embedding ↔ bytes helpers ──────────────────────────────────


