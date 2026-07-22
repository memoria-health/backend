"""Unit: attribution_ran is set at its REAL construction site
(`_run_pipeline_sync`), not just in stubs (T1 review finding — every
other test fakes the result upstream, so a regression to
`attribution_ran=True` on gallery-less runs was invisible).

BodyReIDPipeline is stubbed in the inference namespace; the frames are
real tiny JPEGs so the decode loop runs for real.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np

import src.services.reid.inference as inf
from src.services.detection.frame_extraction import ExtractedFrame
from src.services.reid.inference import ReIDInference


def _jpeg_frame() -> ExtractedFrame:
    ok, buf = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
    assert ok
    return ExtractedFrame(data_base64=base64.b64encode(buf.tobytes()).decode())


class _StubPipeline:
    """Just enough surface for _run_pipeline_sync's result assembly."""

    def __init__(self, *, gallery, config, pre_match_callback=None) -> None:
        self._tracks: dict = {}
        self.has_any_patient_track = False
        self.best_face_embedding = None
        self.best_score = 0.0
        self.best_frame_idx = None
        self.face_count = 0
        self.latest_patient_bbox = None
        self.num_tracks = 0
        self.max_concurrent_tracks = 0
        self.has_only_verified_other_tracks = False
        self.yolo_stats = {
            "frames_processed": 0, "frames_with_detections": 0,
            "zero_detection_frame_pct": 0.0, "total_detections": 0,
            "max_confidence": 0.0, "mean_confidence": 0.0,
            "confidence_threshold": 0.4,
        }

    def process_frame(self, img, frame_idx):
        return SimpleNamespace(patient_bbox=None)

    def track_observations(self):
        return []


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        reid_yolo_model="stub.pt", reid_yolo_confidence=0.4,
        reid_face_check_interval=5, reid_hits_to_lock=2,
        reid_face_min_det_score=0.5, reid_face_checks_for_other=5,
        reid_match_threshold_base=0.40, reid_insightface_model="stub",
        reid_body_gallery_enabled=False,
        # Identity-persistence (2026-07-19) fields the pipeline reads.
        reid_yolo_detect_floor=0.10, reid_track_birth_confidence=0.40,
        reid_face_lock_min_px=36.0, reid_face_lock_min_det=0.6,
        reid_face_lock_score_single=0.45, reid_face_lock_score_multi=0.55,
        reid_face_lock_margin=0.15,
        reid_harvest_min_confidence=0.45,
        reid_quality_min_width=400, reid_quality_sharpness_floor=3200.0,
        reid_quality_min_frames=6,
    )


def _run(gallery: np.ndarray):
    svc = ReIDInference(sessionmaker=None, settings=_settings(), body_gallery=None)
    frames = [_jpeg_frame() for _ in range(3)]
    result, harvest = svc._run_pipeline_sync(
        frames=frames, gallery=gallery, patient_id=uuid4(), sample_fps=5,
    )
    return result, harvest


def test_gallery_less_run_reports_no_attribution(monkeypatch) -> None:
    monkeypatch.setattr(inf, "BodyReIDPipeline", _StubPipeline)
    result, harvest = _run(np.empty((0, 512), dtype=np.float32))
    assert result.reid_ran is True
    assert result.attribution_ran is False
    assert result.faces_enrolled is False
    assert result.tracks == ()
    assert harvest is None


def test_gallery_run_reports_attribution(monkeypatch) -> None:
    monkeypatch.setattr(inf, "BodyReIDPipeline", _StubPipeline)
    result, _ = _run(np.ones((2, 512), dtype=np.float32))
    assert result.reid_ran is True
    assert result.attribution_ran is True
    assert result.faces_enrolled is True
