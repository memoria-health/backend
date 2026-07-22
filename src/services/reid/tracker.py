"""BoT-SORT multi-object tracker with OSNet body Re-ID appearance
embeddings via boxmot.

Verbatim port of `nemara-reid-experiment/src/reid/pipeline/tracker.py`.
Track IDs are stable WITHIN a session (== one per-tick clip, ~11s). They
are NOT cross-tick or cross-session identifiers — that role belongs
to face recognition (`embeddings.embed_image_bytes`).

A new `PersonTracker` is built per tick today (no cross-tick state).
Adding cross-tick continuity is deferred per the plan's "Out of scope
for v0" section.

OSNet weights are pre-downloaded in the Dockerfile so the first tick
after deploy doesn't pay the cold-download cost.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _select_device():
    """Production wall-cam workload is CPU on Railway. Override via
    NEMARA_REID_DEVICE=cuda for a GPU dev box if you have one."""
    import torch
    override = os.environ.get("NEMARA_REID_DEVICE", "").strip().lower()
    if override == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if override == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# One OSNet backend per (weights, device), shared by every tracker.
# PersonTracker is built PER TICK by design (per-clip track state),
# but the old code rebuilt PyTorchBackend — a full model load — each
# time. Leaked ~1 native thread + tens of MB of arena per analyzed
# tick (2026-07-19 soak: ~90-130MB/min climb, 236 futex-parked
# threads). Weights are read-only at inference; concurrent forward
# from both cameras' pipelines is safe.
_REID_BACKENDS: dict[tuple[str, str], object] = {}


def _shared_reid_backend(weights: Path, device) -> object:
    key = (str(weights), str(device))
    be = _REID_BACKENDS.get(key)
    if be is None:
        from boxmot.reid.backends.pytorch_backend import PyTorchBackend
        be = PyTorchBackend(weights=weights, device=device, half=False)
        _REID_BACKENDS[key] = be
        log.info("reid.osnet_backend_loaded weights=%s device=%s", weights, device)
    return be


class PersonTracker:
    """BoT-SORT + OSNet appearance Re-ID wrapper.

    Build a fresh one per tick; the tracker's internal state is
    intentionally per-session.
    """

    def __init__(
        self,
        *,
        reid_weights: Path = Path("osnet_x0_25_msmt17.pt"),
        frame_rate: int = 5,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
    ) -> None:
        # Local imports keep the cold-start cost of `pytest`-only
        # paths low; boxmot pulls torch + several CV libs.
        from boxmot.trackers.botsort.botsort import BotSort
        device = _select_device()
        reid_model = _shared_reid_backend(reid_weights, device)
        # `frame_rate` controls how aggressively the tracker ages out
        # stale tracks. 5 fps matches our per-tick Re-ID sample rate.
        # The three thresholds expose BoT-SORT's ByteTrack-style
        # two-stage association (identity-persistence W6): detections
        # ≥ high go through the first association pass and may BIRTH
        # tracks (subject to new_track_thresh); detections in
        # [low, high) are second-pass CONTINUATION-only — they keep an
        # existing track alive through seated/lying stillness but can
        # never create one.
        self.tracker = BotSort(
            reid_model=reid_model,
            frame_rate=frame_rate,
            with_reid=True,
            track_high_thresh=track_high_thresh,
            track_low_thresh=track_low_thresh,
            new_track_thresh=new_track_thresh,
        )

    def update(self, dets: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
        """Push one frame of detections to the tracker. Returns Mx8
        plain numpy: [x1, y1, x2, y2, track_id, confidence, class, det_index].

        When `dets` is empty, still call `update` so the tracker can age
        stale tracks. Returns (0, 8) in that case.
        """
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=np.float32)
        result = self.tracker.update(dets, frame_bgr)
        if result is None or len(result) == 0:
            return np.empty((0, 8), dtype=np.float32)
        # boxmot 18 returns a TrackResults ndarray subclass. Stitch from
        # named attributes so a future column-order change in boxmot
        # doesn't silently break us.
        xyxy = np.asarray(result.xyxy, dtype=np.float32)
        ids = np.asarray(result.id, dtype=np.float32).reshape(-1, 1)
        conf = np.asarray(result.conf, dtype=np.float32).reshape(-1, 1)
        cls = np.asarray(result.cls, dtype=np.float32).reshape(-1, 1)
        det_ind = np.asarray(result.det_ind, dtype=np.float32).reshape(-1, 1)
        return np.hstack([xyxy, ids, conf, cls, det_ind])
