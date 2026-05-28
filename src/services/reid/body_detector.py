"""YOLO11n person detector.

Verbatim port of `nemara-reid-experiment/src/reid/pipeline/detector.py`.
Filters to COCO class 0 (person) only and returns an Nx6 array of
[x1, y1, x2, y2, confidence, class] in absolute pixel coords.

Lazily constructs the underlying YOLO model on first use so we don't
pay the ~500ms torch init at every test import. In production the
Dockerfile pre-caches `yolo11n.pt` so the first tick after deploy
doesn't pay the cold-download cost either.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


_lock = threading.Lock()
_model = None


def _get_model(model_name: str):
    """Lazy module-level singleton, same pattern as the InsightFace
    FaceAnalysis init."""
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        from ultralytics import YOLO
        log.info("reid.body_detector.init model=%s", model_name)
        _model = YOLO(model_name)
        return _model


class PersonDetector:
    """Single-class (person) detector wrapper around Ultralytics YOLO."""

    def __init__(self, *, model_name: str = "yolo11n.pt", confidence: float = 0.4) -> None:
        self.model_name = model_name
        self.confidence = confidence

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Run inference on one BGR frame. Returns Nx6 (x1, y1, x2, y2,
        conf, class) with class always 0. Empty (0, 6) when no person.
        """
        model = _get_model(self.model_name)
        results = model(
            frame_bgr,
            classes=[0],            # COCO class 0 = person
            conf=self.confidence,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf, cls]).astype(np.float32)
