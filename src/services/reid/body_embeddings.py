"""OSNet body embedding extraction for the cross-clip Re-ID gallery.

Companion to `cross_clip_body_gallery.py`. The gallery stores body
embeddings; this module provides the inference path to compute them
from a frame + body bbox.

Design: lazy singleton wrapping boxmot's `PyTorchBackend` (the same
ReID model BoT-SORT loads internally). Reusing boxmot's wrapper avoids
shipping our own OSNet weights and means we get one model in memory
for the whole process even when many `BodyReIDPipeline` instances are
created per tick.

The boxmot `PyTorchBackend.get_features(xyxys, img)` already L2-
normalizes the output embeddings, so consumers can compute cosine
similarity as a dot product directly.

Threading: the underlying torch model is not thread-safe, but our
production code path drives this from inside `asyncio.to_thread` calls
(one Re-ID pipeline run = one thread, not concurrent), so we don't
need additional locking. Adding it would be a `threading.Lock`
wrapping `get_features` if that assumption ever breaks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


_BACKEND_SINGLETON: Optional[object] = None


def _select_device():
    """Match tracker.py's device selection so OSNet runs on the same
    device as BoT-SORT's internal copy."""
    import os
    import torch
    override = os.environ.get("NEMARA_REID_DEVICE", "").strip().lower()
    if override == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if override == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _get_backend(weights: Path = Path("osnet_x0_25_msmt17.pt")) -> object:
    """Lazy-load + cache the boxmot PyTorchBackend that wraps OSNet.

    Same weights file as `tracker.py` — boxmot caches the downloaded
    weights file under `~/.boxmot`, so the second load is a no-op disk
    read. We hold a separate backend reference here rather than poking
    at the BoT-SORT instance's `reid_model` field; that field is
    private to the boxmot version we use and could move.
    """
    global _BACKEND_SINGLETON
    if _BACKEND_SINGLETON is not None:
        return _BACKEND_SINGLETON

    # Local imports keep cold start cheap for code paths that never
    # touch ReID (tests, health checks, etc.).
    from boxmot.reid.backends.pytorch_backend import PyTorchBackend

    device = _select_device()
    _BACKEND_SINGLETON = PyTorchBackend(weights=weights, device=device, half=False)
    log.info("body_embeddings.osnet_loaded weights=%s device=%s", weights, device)
    return _BACKEND_SINGLETON


def extract_body_embedding(
    *,
    frame_bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    min_box_area: float = 1500.0,
) -> Optional[np.ndarray]:
    """Compute an L2-normalized 512-dim OSNet body embedding for one
    body box on one frame. Returns None when the body box is too
    small to be informative (defaults to a 30x50 px floor — tiny
    boxes produce unstable embeddings that pollute the gallery).

    `bbox` is `(x1, y1, x2, y2)` in image pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if box_area < min_box_area:
        return None

    backend = _get_backend()
    # boxmot's get_features accepts an (N, 4) xyxy float array and an
    # HxWx3 BGR frame. It returns (N, D) L2-normalized embeddings.
    xyxys = np.array([[x1, y1, x2, y2]], dtype=np.float32)
    embeddings = backend.get_features(xyxys, frame_bgr)
    if embeddings is None or len(embeddings) == 0:
        return None
    return embeddings[0].astype(np.float32)
