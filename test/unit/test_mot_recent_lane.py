"""MOT re-bind lane (2026-07-23).

The recent-embedding buffer lets a patient whose track just broke re-acquire
by TEMPORAL LOCALITY — matching their last-seen body — instead of re-climbing
the stale harvest pool for ~6 ticks. Pins:

  - a returning body that grazes BELOW the pool threshold re-acquires once its
    last-seen view is in the lane;
  - an empty lane leaves match() unchanged (purely additive);
  - the lane is TTL-pruned (a stale view stops helping);
  - a DIFFERENT body is NOT lifted by the patient's lane — no new FP path
    (the single edge case: never let someone else become the patient).
"""
from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from identity.gallery import (
    LABEL_PATIENT,
    CrossClipBodyGallery,
)


def _settings(**kw) -> SimpleNamespace:
    base = dict(
        reid_body_gallery_ttl_seconds=14400,
        reid_body_match_patient_threshold=0.72,
        reid_body_match_other_threshold=0.80,
        reid_body_dedup_threshold=0.95,
        reid_body_pool_max=64,
        reid_gallery_harvest_min_interval_s=0.0,
        reid_gallery_harvest_burst=4,
        reid_gallery_golden_count=8,
        reid_other_pool_max=32,
        reid_other_pool_ttl_seconds=7200.0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _unit(idx: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _blend(a: np.ndarray, b: np.ndarray, wa: float) -> np.ndarray:
    v = wa * a + (1.0 - wa) * b
    return (v / np.linalg.norm(v)).astype(np.float32)


def test_recent_lane_lifts_reacquire_below_pool_threshold():
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    # Harvest pool holds a far-pose view; the returning body is a different
    # pose that only reaches ~0.707 cosine against it — under the 0.72 bar.
    g.harvest(patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.9)
    probe = _blend(_unit(0), _unit(1), 0.5)
    cold = g.match(patient_id=pid, probe_embedding=probe)
    assert cold is None or cold.label != LABEL_PATIENT
    # Patient was matched a moment ago -> lane holds this exact view.
    g.remember_recent(patient_id=pid, embedding=probe, confidence=0.72)
    warm = g.match(patient_id=pid, probe_embedding=probe)
    assert warm is not None and warm.label == LABEL_PATIENT and warm.score >= 0.72


def test_empty_recent_lane_is_a_noop():
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    g.harvest(patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.9)
    probe = _blend(_unit(0), _unit(1), 0.5)
    m = g.match(patient_id=pid, probe_embedding=probe)
    assert m is None or m.label != LABEL_PATIENT


def test_recent_lane_ttl_expires():
    g = CrossClipBodyGallery(settings=_settings(reid_reacquire_recent_ttl_s=0.0))
    pid = uuid4()
    g.harvest(patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.9)
    probe = _blend(_unit(0), _unit(1), 0.5)
    g.remember_recent(patient_id=pid, embedding=probe, confidence=0.72)
    m = g.match(patient_id=pid, probe_embedding=probe)  # ttl=0 -> pruned first
    assert m is None or m.label != LABEL_PATIENT


def test_recent_lane_does_not_boost_a_different_body():
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    g.harvest(patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.9)
    g.remember_recent(patient_id=pid, embedding=_unit(0), confidence=0.9)
    other = _unit(5)  # orthogonal to the patient's pool + lane
    m = g.match(patient_id=pid, probe_embedding=other)
    assert m is None or m.label != LABEL_PATIENT
