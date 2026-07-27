"""Identity matching math — numpy-pure, shared across the seam.

The embedding FORMAT and MATCHING rules are identity-domain truths:
the backend's enrollment writes vectors the GPU-host identity service
must read (via the DB), so the codec + comparison live HERE, in the
identity package, and the backend imports them back. The heavyweight
face-embedding pipeline (insightface) stays backend-side in
src/services/identity/embeddings.py — enrollment is a backend concern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

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
    # SAM-face countersign (remote arm): False = the face concept
    # session refuted this claim (e.g. SCRFD firing on the back of a
    # head). Local SCRFD hits default True (no oracle in that path).
    oracle_face: bool = True

    @property
    def face_pixel_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# Lock guards lazy initialization. The FaceAnalysis object is thread-
# safe for inference but not for `prepare()` — two concurrent callers
# both calling prepare() can corrupt the underlying ONNX session.

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
