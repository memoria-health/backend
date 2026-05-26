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

# 512-dim ArcFace embedding. Kept as a constant so callers (repo,
# tests, the adaptive learning harvester) don't hard-code the literal.
EMBEDDING_DIM: int = 512

# ─── Adaptive threshold ─────────────────────────────────────────


def effective_match_threshold(
    base_threshold: float,
    det_score: float,
    face_pixel_area: float,
) -> float:
    """Quality-adaptive cosine threshold for matching a probe face
    against the patient gallery.

    The principle: high-quality faces (clear, frontal, large) hold to
    a tighter bar; low-quality faces (dim, distant, partial) drop the
    bar so we don't false-negative the patient under realistic
    wall-cam conditions.

    Formula: base * clip(0.6 + det_score*0.5 + size_factor, 0.5, 1.2)
    clamped to [0.20, 0.55].

    Calibration points (with base=0.40):

      - Clear close face (det_score=0.9, 200×200=40000 px) → ~0.50
      - Wall-cam normal (det_score=0.7, 100×100=10000 px)  → ~0.40
      - Dim distant face (det_score=0.4, 50×50=2500 px)    → ~0.24

    The 0.20 floor prevents the threshold from collapsing to ~0 on
    catastrophically bad face detections (det_score near zero).
    """
    quality_factor = 0.6 + det_score * 0.5
    if face_pixel_area > 0:
        size_log = math.log10(max(face_pixel_area / 5000.0, 0.01))
        quality_factor += size_log * 0.2
    quality_factor = max(0.5, min(1.2, quality_factor))
    return max(0.20, min(0.55, base_threshold * quality_factor))


# ─── Face embedding extraction ─────────────────────────────────


@dataclass(frozen=True)
class FaceHit:
    """One detected + embedded face. `embedding` is L2-normalized;
    cosine similarity == dot product against gallery vectors.
    """
    embedding: np.ndarray   # shape (EMBEDDING_DIM,) float32
    det_score: float        # SCRFD detection confidence in [0, 1]
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in image coords

    @property
    def face_pixel_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# Lock guards lazy initialization. The FaceAnalysis object is thread-
# safe for inference but not for `prepare()` — two concurrent callers
# both calling prepare() can corrupt the underlying ONNX session.
_init_lock = threading.Lock()
_face_app = None


def _get_face_app(model_name: str):
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

    app = _get_face_app(model_name)
    faces = app.get(img)
    if not faces:
        return None
    best = max(faces, key=lambda f: float(f.det_score))
    return FaceHit(
        embedding=best.normed_embedding.astype(np.float32),
        det_score=float(best.det_score),
        bbox=tuple(float(v) for v in best.bbox),
    )


# ─── Embedding ↔ bytes helpers ──────────────────────────────────


def embedding_to_bytes(vec: np.ndarray) -> bytes:
    """Serialize a (EMBEDDING_DIM,) float32 vector to the bytea
    representation we store in Postgres. The vector must already be
    L2-normalized (InsightFace's `normed_embedding` always is)."""
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"expected shape ({EMBEDDING_DIM},), got {vec.shape}"
        )
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    return vec.tobytes()


def embedding_from_bytes(buf: bytes) -> np.ndarray:
    """Inverse of `embedding_to_bytes`. Validates the byte length so a
    corrupted row surfaces immediately rather than silently producing
    nonsense matches downstream."""
    expected = EMBEDDING_DIM * 4
    if len(buf) != expected:
        raise ValueError(
            f"embedding bytea wrong length: got {len(buf)}, expected {expected}"
        )
    return np.frombuffer(buf, dtype=np.float32)


# ─── Gallery match ───────────────────────────────────────────────


@dataclass(frozen=True)
class MatchResult:
    """Top-K mean cosine similarity vs the patient's gallery.

    `score` is the mean of the top-K best similarities, which is more
    stable than max() (one outlier embedding can't single-handedly
    decide a match) and more discriminating than mean() (an irrelevant
    bulk of bad embeddings can't drag the score down)."""
    score: float
    top_k_used: int


def topk_mean_cosine_match(
    probe: np.ndarray,
    gallery: np.ndarray,
    top_k: int = 5,
) -> MatchResult:
    """Return the top-K mean cosine similarity between `probe` and the
    rows of `gallery`.

    Both `probe` (shape (D,)) and `gallery` (shape (N, D)) must be
    L2-normalized. Returns (0.0, 0) when `gallery` is empty.
    """
    if gallery.size == 0:
        return MatchResult(score=0.0, top_k_used=0)
    sims = gallery @ probe.astype(np.float32)
    k = min(top_k, len(sims))
    topk = np.sort(sims)[-k:]
    return MatchResult(score=float(np.mean(topk)), top_k_used=k)
