"""Unit tests for `CrossClipBodyGallery` (POOL design, identity-
persistence W4, 2026-07-19 — replaces the cluster-based tests).

The gallery is the persistent identity layer that lets a face-matched
person be body-matched in future clips when their face is hidden. The
pool design keeps a flat per-label buffer of embeddings and matches a
probe on the mean of its top-3 cosines. These tests pin:

  - Harvest writes valid labels, rejects invalid ones, dedups
    near-duplicates, FIFO-evicts past the pool cap.
  - Match uses top-k(3) mean; PATIENT/OTHER thresholds are asymmetric;
    PATIENT preferred on close ties.
  - THE FRAGMENTATION REGRESSION: dissimilar same-person views
    accumulate in ONE pool (no merge threshold to fail) and a probe
    near one of them matches — the exact case the cluster design lost
    (soak #2: 11 singleton clusters, scores flapping around 0.72).
  - TTL eviction drops stale embeddings on read/write paths.
  - Match returns None on empty gallery / below threshold.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from src.services.reid.cross_clip_body_gallery import (
    LABEL_OTHER,
    LABEL_PATIENT,
    CrossClipBodyGallery,
)


def _settings(
    *,
    ttl: int = 14400,
    patient_threshold: float = 0.72,
    other_threshold: float = 0.80,
    dedup_threshold: float = 0.95,
    pool_max: int = 64,
    harvest_min_interval: float = 0.0,
    golden_count: int = 8,
    other_pool_max: int = 32,
    other_ttl: float = 7200.0,
) -> SimpleNamespace:
    """SimpleNamespace stand-in for the real Settings — keeps the
    gallery's coupling to Settings purely structural. Hygiene rate cap
    defaults to 0 here so multi-harvest tests run un-throttled; the
    rate-cap test opts in explicitly."""
    return SimpleNamespace(
        reid_body_gallery_ttl_seconds=ttl,
        reid_body_match_patient_threshold=patient_threshold,
        reid_body_match_other_threshold=other_threshold,
        reid_body_dedup_threshold=dedup_threshold,
        reid_body_pool_max=pool_max,
        reid_gallery_harvest_min_interval_s=harvest_min_interval,
        reid_gallery_harvest_burst=4,
        reid_gallery_golden_count=golden_count,
        reid_other_pool_max=other_pool_max,
        reid_other_pool_ttl_seconds=other_ttl,
    )


def _unit(idx: int, dim: int = 512) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _blend(a: np.ndarray, b: np.ndarray, wa: float) -> np.ndarray:
    v = wa * a + (1.0 - wa) * b
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture()
def gallery() -> CrossClipBodyGallery:
    return CrossClipBodyGallery(settings=_settings())


def test_harvest_rejects_invalid_label(gallery):
    assert gallery.harvest(
        patient_id=uuid4(), embedding=_unit(0), label="UNKNOWN",
        confidence=0.9,
    ) is False


def test_harvest_and_match_same_view(gallery):
    pid = uuid4()
    assert gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT,
        confidence=0.9,
    )
    m = gallery.match(patient_id=pid, probe_embedding=_unit(0))
    assert m is not None and m.label == LABEL_PATIENT
    assert m.score == pytest.approx(1.0)
    assert m.pool_size == 1


def test_harvest_dedups_near_duplicates(gallery):
    pid = uuid4()
    assert gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT,
        confidence=0.9,
    )
    near = _blend(_unit(0), _unit(1), 0.99)   # cosine ~0.999 to first
    assert gallery.harvest(
        patient_id=pid, embedding=near, label=LABEL_PATIENT,
        confidence=0.9,
    ) is False
    assert gallery.size(pid) == 1


def test_pool_accumulates_dissimilar_views_no_fragmentation(gallery):
    """The cluster-design regression: cross-pose views all land in
    ONE pool, and a probe equal to one of them matches despite the
    others being far — the exact case merge@0.85 lost."""
    pid = uuid4()
    views = [
        _blend(_unit(0), _unit(i), 0.72)     # share a base component
        for i in (1, 2, 3)
    ]
    for v in views:
        assert gallery.harvest(
            patient_id=pid, embedding=v, label=LABEL_PATIENT,
            confidence=0.9,
        )
    assert gallery.label_counts(pid)[LABEL_PATIENT] == 3
    m = gallery.match(patient_id=pid, probe_embedding=views[0])
    assert m is not None and m.label == LABEL_PATIENT
    assert m.score > 0.72


def test_match_below_threshold_returns_none(gallery):
    pid = uuid4()
    gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT,
        confidence=0.9,
    )
    assert gallery.match(patient_id=pid, probe_embedding=_unit(5)) is None


def test_match_empty_gallery_returns_none(gallery):
    assert gallery.match(patient_id=uuid4(), probe_embedding=_unit(0)) is None


def test_other_threshold_is_stricter(gallery):
    pid = uuid4()
    gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_OTHER,
        confidence=0.9,
    )
    # Probe between the two bars: above 0.72, below 0.80 → OTHER does
    # NOT match (conservative direction — never skip the LLM on a
    # maybe-patient). Note _blend normalizes, so cosine ≠ weight:
    # w=0.53 → cos = 0.53/√(0.53²+0.47²) ≈ 0.748.
    probe = _blend(_unit(0), _unit(1), 0.53)
    sim = float(np.dot(probe, _unit(0)))
    assert 0.72 < sim < 0.80
    assert gallery.match(patient_id=pid, probe_embedding=probe) is None


def test_tie_break_prefers_patient(gallery):
    pid = uuid4()
    gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT,
        confidence=0.9,
    )
    gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_OTHER,
        confidence=0.9,
    )
    m = gallery.match(patient_id=pid, probe_embedding=_unit(0))
    assert m is not None and m.label == LABEL_PATIENT


def test_fifo_eviction_past_pool_cap():
    g = CrossClipBodyGallery(
        settings=_settings(pool_max=3, dedup_threshold=0.999),
    )
    pid = uuid4()
    for i in (1, 2, 3, 4):
        g.harvest(
            patient_id=pid,
            embedding=_blend(_unit(0), _unit(i), 0.72),
            label=LABEL_PATIENT,
            confidence=0.9,
        )
    assert g.label_counts(pid)[LABEL_PATIENT] == 3    # oldest evicted


def test_ttl_evicts_stale_embeddings(monkeypatch):
    g = CrossClipBodyGallery(settings=_settings(ttl=100))
    pid = uuid4()
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    g.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT,
        confidence=0.9,
    )
    t[0] += 200.0
    assert g.match(patient_id=pid, probe_embedding=_unit(0)) is None
    assert g.size(pid) == 0


def test_clear_and_introspection(gallery):
    pid = uuid4()
    gallery.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT,
        confidence=0.9,
    )
    assert gallery.size(pid) == 1
    assert gallery.cluster_count(pid) == 1            # non-empty pools
    assert gallery.cluster_label_counts(pid)[LABEL_PATIENT] == 1
    gallery.clear(pid)
    assert gallery.size(pid) == 0


# ─── Harvest hygiene (2026-07-19 late soak: the aunt poisoning) ───


def test_rate_cap_allows_one_burst_then_throttles() -> None:
    # One lock event = up to 4 poses (the multi-pose deposit). The
    # 5th rapid harvest is churn and gets throttled.
    g = CrossClipBodyGallery(settings=_settings(harvest_min_interval=60.0))
    pid = uuid4()
    for i in range(4):
        assert g.harvest(
            patient_id=pid, embedding=_unit(i), label=LABEL_PATIENT,
            confidence=0.6,
        )
    assert not g.harvest(
        patient_id=pid, embedding=_unit(9), label=LABEL_PATIENT, confidence=0.9,
    )
    assert g.size(pid) == 4


def test_rate_cap_is_per_label() -> None:
    g = CrossClipBodyGallery(settings=_settings(harvest_min_interval=60.0))
    pid = uuid4()
    assert g.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.6,
    )
    # OTHER pool has its own clock — a guest can enroll immediately.
    assert g.harvest(
        patient_id=pid, embedding=_unit(1), label=LABEL_OTHER, confidence=0.7,
    )


def test_golden_entries_survive_eviction() -> None:
    # Cap 4, golden 1: the single highest-confidence entry is the
    # OLDEST — pre-hygiene FIFO would evict it first.
    g = CrossClipBodyGallery(settings=_settings(pool_max=4, golden_count=1))
    pid = uuid4()
    assert g.harvest(
        patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.95,
    )
    for i in range(1, 6):        # five mediocre harvests overflow the cap
        g.harvest(
            patient_id=pid, embedding=_unit(i), label=LABEL_PATIENT,
            confidence=0.45,
        )
    pool = g._pools[pid][LABEL_PATIENT]
    # Pre-hygiene FIFO would have evicted the oldest (=golden) entry.
    assert len(pool) == 4
    assert max(e.confidence for e in pool) == pytest.approx(0.95)


def test_other_pool_has_shorter_ttl(monkeypatch) -> None:
    # PATIENT ttl 14400s, OTHER ttl 100s. Advance a fake clock 200s:
    # the guest entry dies, the patient entry survives.
    g = CrossClipBodyGallery(settings=_settings(other_ttl=100.0))
    pid = uuid4()
    t = {"now": 1000.0}
    monkeypatch.setattr(
        "src.services.reid.cross_clip_body_gallery.time",
        SimpleNamespace(monotonic=lambda: t["now"]),
    )
    g.harvest(patient_id=pid, embedding=_unit(0), label=LABEL_PATIENT, confidence=0.6)
    g.harvest(patient_id=pid, embedding=_unit(1), label=LABEL_OTHER, confidence=0.7)
    assert g.size(pid) == 2
    t["now"] += 200.0
    g.gc(pid)
    assert g.size(pid) == 1     # only the PATIENT entry remains


def test_other_pool_cap_is_separate() -> None:
    g = CrossClipBodyGallery(settings=_settings(pool_max=64, other_pool_max=2))
    pid = uuid4()
    for i in range(4):
        g.harvest(
            patient_id=pid, embedding=_unit(i), label=LABEL_OTHER,
            confidence=0.5,
        )
    assert g.size(pid) == 2     # OTHER pool bounded at its own cap
