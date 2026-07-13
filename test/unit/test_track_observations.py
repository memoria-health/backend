"""Unit: per-track output from BodyReIDPipeline (detection v2 P2 T2).

`process_frame` must record every (frame_idx, bbox) a track is reported
at, and `track_observations()` must expose immutable, sorted per-track
summaries — the raw material for per-track motion scoping and
env-motion masking in the signals layer.

Driven gallery-less (the v2 always-on mode): face checks are skipped,
so stubbing YOLO/BoT-SORT outputs exercises pure track bookkeeping.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from src.services.reid.body_pipeline import (
    LABEL_PATIENT,
    LABEL_UNKNOWN,
    BodyReIDPipeline,
    PipelineConfig,
    TrackObservation,
)
from src.services.reid.inference import IdentityResult

_FRAME = np.zeros((120, 160, 3), dtype=np.uint8)


def _make_pipeline() -> BodyReIDPipeline:
    with (
        patch("src.services.reid.body_pipeline.body_detector.PersonDetector"),
        patch("src.services.reid.body_pipeline.PersonTracker"),
    ):
        return BodyReIDPipeline(
            gallery=np.empty((0, 512), dtype=np.float32),
            config=PipelineConfig(sample_fps=5),
        )


def _feed(pipeline: BodyReIDPipeline, frame_idx: int, rows: list[list[float]]) -> None:
    """rows: [x1, y1, x2, y2, tid, det_conf] per visible track."""
    tracked = np.array(
        [[x1, y1, x2, y2, tid, conf, 0.0, 0.0] for x1, y1, x2, y2, tid, conf in rows],
        dtype=np.float64,
    ).reshape(len(rows), 8)
    pipeline._detector.detect = lambda frame: np.empty((0, 6))
    pipeline._tracker.update = lambda dets, frame: tracked
    pipeline.process_frame(_FRAME, frame_idx)


def test_bbox_series_records_every_sighting_in_order() -> None:
    p = _make_pipeline()
    _feed(p, 0, [[10, 10, 50, 90, 1, 0.9], [80, 12, 120, 95, 2, 0.8]])
    _feed(p, 1, [[12, 11, 52, 91, 1, 0.9]])          # track 2 lost this frame
    _feed(p, 2, [[14, 12, 54, 92, 1, 0.9], [82, 13, 122, 96, 2, 0.7]])

    obs = p.track_observations()
    assert [o.track_id for o in obs] == [1, 2]
    assert [idx for idx, _ in obs[0].bbox_series] == [0, 1, 2]
    assert [idx for idx, _ in obs[1].bbox_series] == [0, 2]
    assert obs[0].bbox_series[1][1] == (12.0, 11.0, 52.0, 91.0)


def test_observations_are_immutable_and_unlabeled_by_default() -> None:
    p = _make_pipeline()
    _feed(p, 0, [[10, 10, 50, 90, 7, 0.9]])
    (obs,) = p.track_observations()
    assert isinstance(obs, TrackObservation)
    assert isinstance(obs.bbox_series, tuple)
    assert obs.label == LABEL_UNKNOWN
    assert obs.confidence == 0.0
    assert obs.face_locked is False
    assert obs.pre_labeled is False


def test_observations_reflect_lock_state_and_sort_order() -> None:
    p = _make_pipeline()
    _feed(p, 0, [[10, 10, 50, 90, 9, 0.9], [80, 12, 120, 95, 3, 0.8]])
    # Simulate a PATIENT face-lock on track 9 (the state the face
    # matcher writes) and a body-gallery pre-label on track 3.
    p._tracks[9].label = LABEL_PATIENT
    p._tracks[9].confidence = 0.71
    p._tracks[9].face_locked = True
    p._tracks[3].pre_label = "OTHER"

    obs = p.track_observations()
    assert [o.track_id for o in obs] == [3, 9]       # sorted, not insertion order
    assert obs[1].label == LABEL_PATIENT
    assert obs[1].confidence == 0.71
    assert obs[1].face_locked is True
    assert obs[0].pre_labeled is True


def test_no_tracks_yields_empty_list_and_empty_result_has_none() -> None:
    assert _make_pipeline().track_observations() == []
    # IdentityResult distinguishes "ran, saw nobody" (empty tuple) from
    # "didn't run" (None).
    assert IdentityResult.empty().tracks is None
