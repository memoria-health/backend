"""Unit tests for `CrossClipBodyGallery` (POOL design, identity-
persistence W4, 2026-07-19 — replaces the cluster-based tests).

The gallery is the persistent identity layer that lets a face-matched
person be body-matched in future clips when their face is hidden. The
pool design keeps a flat per-patient buffer of embeddings and matches a
probe on the mean of its top-3 cosines. These tests pin:

  - Harvest dedups near-duplicates and evicts past the pool cap while
    protecting the highest-confidence ("golden") entries.
  - Match uses top-k(3) mean and returns None below the threshold.
  - THE FRAGMENTATION REGRESSION: dissimilar same-person views
    accumulate in ONE pool (no merge threshold to fail) and a probe
    near one of them matches — the exact case the cluster design lost
    (soak #2: 11 singleton clusters, scores flapping around 0.72).
  - TTL eviction drops stale embeddings on read/write paths.
  - MULTI-PATIENT (2026-08-20): a camera covering several residents
    asks each pool the same binary question and takes the best score.
    The load-bearing tests here are the ones that pin what must NOT
    happen — no fallback to the first-listed patient, no attribution
    when nobody clears the bar — and the parity guard proving the
    one-patient path is unchanged.

The PATIENT/OTHER label tests were deleted with the labels themselves
(2026-08-20). The identity node never wrote or read an OTHER pool, so
the asymmetric threshold, separate cap, shorter TTL and "prefer PATIENT
within 0.05" tie-break were all unreachable — testing them was pinning
behaviour no caller could ever observe.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from identity.gallery import CrossClipBodyGallery


def _settings(
    *,
    ttl: int = 14400,
    patient_threshold: float = 0.72,
    dedup_threshold: float = 0.95,
    pool_max: int = 64,
    harvest_min_interval: float = 0.0,
    golden_count: int = 8,
) -> SimpleNamespace:
    """SimpleNamespace stand-in for the real Settings — keeps the
    gallery's coupling to Settings purely structural. Hygiene rate cap
    defaults to 0 here so multi-harvest tests run un-throttled; the
    rate-cap test opts in explicitly."""
    return SimpleNamespace(
        reid_body_gallery_ttl_seconds=ttl,
        reid_body_match_patient_threshold=patient_threshold,
        reid_body_dedup_threshold=dedup_threshold,
        reid_body_pool_max=pool_max,
        reid_gallery_harvest_min_interval_s=harvest_min_interval,
        reid_gallery_harvest_burst=4,
        reid_gallery_golden_count=golden_count,
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


def test_harvest_and_match_same_view(gallery):
    pid = uuid4()
    assert gallery.harvest(
        patient_id=pid, embedding=_unit(0), confidence=0.9,
    )
    m = gallery.match(patient_ids=[pid], probe_embedding=_unit(0))
    assert m is not None and m.patient_id == pid
    assert m.score == pytest.approx(1.0)
    assert m.pool_size == 1


def test_harvest_dedups_near_duplicates(gallery):
    pid = uuid4()
    assert gallery.harvest(
        patient_id=pid, embedding=_unit(0), confidence=0.9,
    )
    near = _blend(_unit(0), _unit(1), 0.99)   # cosine ~0.999 to first
    assert gallery.harvest(
        patient_id=pid, embedding=near, confidence=0.9,
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
        assert gallery.harvest(patient_id=pid, embedding=v, confidence=0.9)
    assert gallery.size(pid) == 3
    m = gallery.match(patient_ids=[pid], probe_embedding=views[0])
    assert m is not None and m.patient_id == pid
    assert m.score > 0.72


def test_match_below_threshold_returns_none(gallery):
    pid = uuid4()
    gallery.harvest(patient_id=pid, embedding=_unit(0), confidence=0.9)
    assert gallery.match(patient_ids=[pid], probe_embedding=_unit(5)) is None


def test_match_empty_gallery_returns_none(gallery):
    assert gallery.match(
        patient_ids=[uuid4()], probe_embedding=_unit(0),
    ) is None


def test_ttl_evicts_stale_embeddings(monkeypatch):
    g = CrossClipBodyGallery(settings=_settings(ttl=100))
    pid = uuid4()
    t = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: t[0])
    g.harvest(patient_id=pid, embedding=_unit(0), confidence=0.9)
    t[0] += 200.0
    assert g.match(patient_ids=[pid], probe_embedding=_unit(0)) is None
    assert g.size(pid) == 0


def test_clear_and_introspection(gallery):
    pid = uuid4()
    gallery.harvest(patient_id=pid, embedding=_unit(0), confidence=0.9)
    assert gallery.size(pid) == 1
    gallery.clear(pid)
    assert gallery.size(pid) == 0


# ─── Harvest hygiene (2026-07-19 late soak: the aunt poisoning) ───


def test_rate_cap_allows_one_burst_then_throttles() -> None:
    # One lock event = up to 4 poses (the multi-pose deposit). The
    # 5th rapid harvest is churn and gets throttled.
    g = CrossClipBodyGallery(settings=_settings(harvest_min_interval=60.0))
    pid = uuid4()
    for i in range(4):
        assert g.harvest(patient_id=pid, embedding=_unit(i), confidence=0.6)
    assert not g.harvest(
        patient_id=pid, embedding=_unit(9), confidence=0.9,
    )
    assert g.size(pid) == 4


def test_rate_cap_is_per_patient() -> None:
    """Two residents on one camera keep separate harvest clocks —
    otherwise the first one to lock would throttle the second out of
    ever building a pool."""
    g = CrossClipBodyGallery(settings=_settings(harvest_min_interval=60.0))
    a, b = uuid4(), uuid4()
    for i in range(4):
        assert g.harvest(patient_id=a, embedding=_unit(i), confidence=0.6)
    assert not g.harvest(patient_id=a, embedding=_unit(9), confidence=0.9)
    assert g.harvest(patient_id=b, embedding=_unit(20), confidence=0.6)


def test_golden_entries_survive_eviction() -> None:
    # Cap 4, golden 1: the single highest-confidence entry is the
    # OLDEST — pre-hygiene FIFO would evict it first.
    g = CrossClipBodyGallery(settings=_settings(pool_max=4, golden_count=1))
    pid = uuid4()
    assert g.harvest(patient_id=pid, embedding=_unit(0), confidence=0.95)
    for i in range(1, 6):        # five mediocre harvests overflow the cap
        g.harvest(patient_id=pid, embedding=_unit(i), confidence=0.45)
    pool = g._pools[pid]
    # Pre-hygiene FIFO would have evicted the oldest (=golden) entry.
    assert len(pool) == 4
    assert max(e.confidence for e in pool) == pytest.approx(0.95)


# ─── Multi-patient: a camera that watches more than one resident ───


def test_single_patient_path_is_unchanged(gallery):
    """PARITY GUARD (2026-08-20). The overwhelming majority of cameras
    watch exactly one person, and that path must behave exactly as it
    did before multi-patient landed: the one pool is scored, the
    threshold is the only bar, and the verdict names that patient."""
    pid = uuid4()
    gallery.harvest(patient_id=pid, embedding=_unit(0), confidence=0.9)

    hit = gallery.match(patient_ids=[pid], probe_embedding=_unit(0))
    assert hit is not None
    assert hit.patient_id == pid and hit.score == pytest.approx(1.0)

    # Just under the bar is still a miss — no second pool exists to
    # rescue it, exactly as before.
    near = _blend(_unit(0), _unit(1), 0.5)      # cosine ≈ 0.707 < 0.72
    assert float(np.dot(near, _unit(0))) < 0.72
    assert gallery.match(patient_ids=[pid], probe_embedding=near) is None


def test_picks_the_highest_scoring_patient(gallery):
    """The lounge case: two residents enrolled, the body is one of
    them, and the stronger pool wins."""
    a, b = uuid4(), uuid4()
    gallery.harvest(patient_id=a, embedding=_unit(0), confidence=0.9)
    gallery.harvest(patient_id=b, embedding=_unit(1), confidence=0.9)

    m = gallery.match(patient_ids=[a, b], probe_embedding=_unit(1))
    assert m is not None and m.patient_id == b
    assert m.score == pytest.approx(1.0)


def test_never_falls_back_to_a_listed_patient(gallery):
    """THE SHADY-BUG GUARD. A stranger walks through a shared lounge.
    Neither resident's pool clears the bar, so the answer must be
    nobody — never "the first patient we were handed", which is the
    shape a careless implementation takes and which would silently
    file a stranger's activity on a resident's record."""
    a, b = uuid4(), uuid4()
    gallery.harvest(patient_id=a, embedding=_unit(0), confidence=0.9)
    gallery.harvest(patient_id=b, embedding=_unit(1), confidence=0.9)

    assert gallery.match(
        patient_ids=[a, b], probe_embedding=_unit(7),
    ) is None
    # ...including when only one of them has any pool at all, which is
    # the real state of a camera newly shared with an un-enrolled
    # resident.
    fresh = CrossClipBodyGallery(settings=_settings())
    fresh.harvest(patient_id=a, embedding=_unit(0), confidence=0.9)
    assert fresh.match(
        patient_ids=[a, b], probe_embedding=_unit(7),
    ) is None


def test_verdict_does_not_depend_on_the_order_patients_are_passed(gallery):
    """An exact tie must resolve the same way every probe. Ordering
    comes from a coverage query with no guaranteed ORDER BY, so a
    max()-takes-first rule would make attribution flap between two
    residents from one tick to the next."""
    a, b = uuid4(), uuid4()
    gallery.harvest(patient_id=a, embedding=_unit(0), confidence=0.9)
    gallery.harvest(patient_id=b, embedding=_unit(0), confidence=0.9)

    forward = gallery.match(patient_ids=[a, b], probe_embedding=_unit(0))
    reverse = gallery.match(patient_ids=[b, a], probe_embedding=_unit(0))
    assert forward is not None and reverse is not None
    assert forward.patient_id == reverse.patient_id


def test_one_resident_enrolled_does_not_shadow_the_other(gallery):
    """A shared camera where only one resident has walked past enough
    to build a pool must still recognize the other one the moment their
    pool exists — the covered set is read per probe, not cached from
    whoever locked first."""
    a, b = uuid4(), uuid4()
    gallery.harvest(patient_id=a, embedding=_unit(0), confidence=0.9)
    assert gallery.match(
        patient_ids=[a, b], probe_embedding=_unit(1),
    ) is None

    gallery.harvest(patient_id=b, embedding=_unit(1), confidence=0.9)
    m = gallery.match(patient_ids=[a, b], probe_embedding=_unit(1))
    assert m is not None and m.patient_id == b
