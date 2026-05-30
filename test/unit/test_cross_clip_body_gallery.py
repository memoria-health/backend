"""Unit tests for `CrossClipBodyGallery` (cluster-based design).

The gallery is the persistent identity layer that lets a face-matched
person be body-matched in future clips when their face is hidden.
It groups embeddings into per-person clusters and matches probes
against each cluster's mean cosine. These tests pin:

  - Harvest writes valid labels, rejects invalid ones, dedups
    near-duplicates within a cluster.
  - Match returns the best cluster above the per-label cosine
    threshold, with PATIENT preferred on close ties.
  - TTL eviction drops stale clusters on read/write paths.
  - Per-label max-clusters cap evicts least-recently-updated first.
  - Cluster merge: similar same-label embeddings fuse into one
    cluster; dissimilar ones seed new clusters.
  - Multi-embedding match: probe matches against cluster mean cosine,
    not single best.
  - Match returns None when gallery is empty / no cluster passes
    threshold / patient has no clusters.
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
    GalleryMatch,
    _BodyCluster,
)


def _settings(
    *,
    ttl: int = 14400,
    patient_threshold: float = 0.72,
    other_threshold: float = 0.80,
    dedup_threshold: float = 0.95,
    cluster_merge_threshold: float = 0.85,
    max_embeddings_per_cluster: int = 8,
    max_clusters_per_label: int = 20,
    enabled: bool = True,
) -> SimpleNamespace:
    """SimpleNamespace stand-in for the real Settings — keeps the
    gallery's coupling to Settings purely structural. Avoids
    constructing the full pydantic model in every test."""
    return SimpleNamespace(
        reid_body_gallery_ttl_seconds=ttl,
        reid_body_match_patient_threshold=patient_threshold,
        reid_body_match_other_threshold=other_threshold,
        reid_body_dedup_threshold=dedup_threshold,
        reid_body_cluster_merge_threshold=cluster_merge_threshold,
        reid_body_max_embeddings_per_cluster=max_embeddings_per_cluster,
        reid_body_max_clusters_per_label=max_clusters_per_label,
        reid_body_gallery_enabled=enabled,
    )


def _unit_vec(dim: int = 512, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _aligned_vec(base: np.ndarray, similarity: float) -> np.ndarray:
    """Return a unit vector with the requested cosine similarity to
    `base`. Used to synthesize controlled similarity probes."""
    rng = np.random.default_rng(int(similarity * 1000) + 17)
    noise = rng.standard_normal(base.shape).astype(np.float32)
    noise = noise - np.dot(noise, base) * base
    noise = noise / np.linalg.norm(noise)
    v = similarity * base + np.sqrt(max(0.0, 1.0 - similarity ** 2)) * noise
    return (v / np.linalg.norm(v)).astype(np.float32)


# ── Harvest: basic ───────────────────────────────────────────────


def test_harvest_writes_patient_entry() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    ok = g.harvest(
        patient_id=pid, embedding=_unit_vec(seed=1),
        label=LABEL_PATIENT, confidence=0.85,
    )
    assert ok is True
    assert g.size(pid) == 1
    assert g.cluster_count(pid) == 1
    assert g.label_counts(pid) == {LABEL_PATIENT: 1, LABEL_OTHER: 0}
    assert g.cluster_label_counts(pid) == {LABEL_PATIENT: 1, LABEL_OTHER: 0}


def test_harvest_writes_other_entry() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    g.harvest(
        patient_id=pid, embedding=_unit_vec(seed=2),
        label=LABEL_OTHER, confidence=0.0,
    )
    assert g.label_counts(pid) == {LABEL_PATIENT: 0, LABEL_OTHER: 1}
    assert g.cluster_label_counts(pid) == {LABEL_PATIENT: 0, LABEL_OTHER: 1}


def test_harvest_rejects_invalid_label() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    ok = g.harvest(
        patient_id=pid, embedding=_unit_vec(),
        label="UNKNOWN", confidence=0.0,
    )
    assert ok is False
    assert g.size(pid) == 0


# ── Harvest: cluster merge vs seed ───────────────────────────────


def test_harvest_similar_embeddings_merge_into_one_cluster() -> None:
    """Two harvests of the same person (high cosine, but not identical)
    should fuse into a single cluster with two embeddings — the whole
    point of the cluster design vs flat entries.

    Construct b as a direct perturbation of a so cosine(a, b) is
    controllable. (Two vectors each at 0.92 cosine to a SHARED base
    can have unpredictable cosine to each other depending on the
    direction of their respective orthogonal noise — confirmed by a
    failing test that produced 0.83 instead of the expected ~0.92.)
    """
    g = CrossClipBodyGallery(settings=_settings(
        cluster_merge_threshold=0.85, dedup_threshold=0.95,
    ))
    pid = uuid4()
    a = _unit_vec(seed=300)
    b = _aligned_vec(a, similarity=0.90)    # b explicitly close to a
    assert g.harvest(patient_id=pid, embedding=a, label=LABEL_OTHER, confidence=0.0)
    assert g.harvest(patient_id=pid, embedding=b, label=LABEL_OTHER, confidence=0.0)
    # ONE cluster with TWO embeddings, NOT two separate clusters.
    assert g.cluster_count(pid) == 1
    assert g.size(pid) == 2
    assert g.cluster_label_counts(pid) == {LABEL_PATIENT: 0, LABEL_OTHER: 1}


def test_harvest_dissimilar_embeddings_seed_separate_clusters() -> None:
    """Two harvests of clearly different people (low cosine) should
    seed two distinct clusters — modeling that there's helper A AND
    helper B, not one fuzzy identity."""
    g = CrossClipBodyGallery(settings=_settings(cluster_merge_threshold=0.85))
    pid = uuid4()
    person_a = _unit_vec(seed=310)
    person_b = _unit_vec(seed=311)  # different seed → low cosine to A
    g.harvest(patient_id=pid, embedding=person_a, label=LABEL_OTHER, confidence=0.0)
    g.harvest(patient_id=pid, embedding=person_b, label=LABEL_OTHER, confidence=0.0)
    assert g.cluster_count(pid) == 2
    assert g.cluster_label_counts(pid) == {LABEL_PATIENT: 0, LABEL_OTHER: 2}


def test_harvest_intra_cluster_dedup_skips_near_identical() -> None:
    """Once a cluster exists, a new embedding that's near-identical
    to an existing member (cosine > dedup_threshold) is skipped —
    keeps the cluster buffer DIVERSE so the mean represents the
    person across poses, not N copies of one frame."""
    g = CrossClipBodyGallery(settings=_settings(
        cluster_merge_threshold=0.85, dedup_threshold=0.95,
    ))
    pid = uuid4()
    base = _unit_vec(seed=320)
    near_identical = _aligned_vec(base, similarity=0.98)
    assert g.harvest(
        patient_id=pid, embedding=base, label=LABEL_OTHER, confidence=0.0,
    )
    assert g.harvest(
        patient_id=pid, embedding=near_identical,
        label=LABEL_OTHER, confidence=0.0,
    ) is False
    # Still one cluster, still one embedding.
    assert g.cluster_count(pid) == 1
    assert g.size(pid) == 1


def test_harvest_dedup_does_not_apply_across_labels() -> None:
    """A PATIENT and an OTHER harvest with similar embeddings produce
    two clusters (one per label). The merge logic only considers
    same-label clusters."""
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    base = _unit_vec(seed=7)
    g.harvest(patient_id=pid, embedding=base, label=LABEL_PATIENT, confidence=0.9)
    g.harvest(patient_id=pid, embedding=base, label=LABEL_OTHER, confidence=0.0)
    assert g.cluster_count(pid) == 2
    assert g.label_counts(pid) == {LABEL_PATIENT: 1, LABEL_OTHER: 1}


def test_harvest_below_merge_threshold_seeds_new_cluster_even_same_person() -> None:
    """If a new embedding is too dissimilar to an existing cluster
    (below merge threshold), we seed a NEW cluster even if the operator
    knows it's the same person. This is the conservative safety
    behavior: a wrong merge contaminates the cluster mean permanently."""
    g = CrossClipBodyGallery(settings=_settings(cluster_merge_threshold=0.95))
    pid = uuid4()
    base = _unit_vec(seed=330)
    # 0.90 cosine — high, but below the very-strict 0.95 merge threshold
    distant_view = _aligned_vec(base, similarity=0.90)
    g.harvest(patient_id=pid, embedding=base, label=LABEL_OTHER, confidence=0.0)
    g.harvest(patient_id=pid, embedding=distant_view, label=LABEL_OTHER, confidence=0.0)
    assert g.cluster_count(pid) == 2


# ── Cluster size cap (within one cluster) ────────────────────────


def test_cluster_buffer_caps_at_max_embeddings_with_fifo() -> None:
    """A single cluster's embedding buffer caps at
    `max_embeddings_per_cluster`. When full, oldest is evicted FIFO
    so the cluster always reflects the most recent N views."""
    g = CrossClipBodyGallery(settings=_settings(
        cluster_merge_threshold=0.50,    # easy merge for this test
        dedup_threshold=0.99,            # no dedup in the way
        max_embeddings_per_cluster=3,
    ))
    pid = uuid4()
    base = _unit_vec(seed=400)
    # 5 distinct-but-similar views of one person.
    similarities = [0.91, 0.92, 0.93, 0.94, 0.95]
    for s in similarities:
        g.harvest(
            patient_id=pid,
            embedding=_aligned_vec(base, similarity=s),
            label=LABEL_OTHER, confidence=0.0,
        )
    # All five high-similarity adds went into ONE cluster (merge_thr=0.5
    # is very lenient). Cluster buffer caps at 3 → only last three survive.
    assert g.cluster_count(pid) == 1
    assert g.size(pid) == 3


# ── Match: basic threshold behavior ──────────────────────────────


def test_match_returns_none_on_empty_gallery() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    pid = uuid4()
    assert g.match(patient_id=pid, probe_embedding=_unit_vec()) is None


def test_match_returns_none_when_no_cluster_above_threshold() -> None:
    g = CrossClipBodyGallery(settings=_settings(other_threshold=0.80))
    pid = uuid4()
    base = _unit_vec(seed=10)
    g.harvest(patient_id=pid, embedding=base, label=LABEL_OTHER, confidence=0.0)
    weak_probe = _aligned_vec(base, similarity=0.6)
    assert g.match(patient_id=pid, probe_embedding=weak_probe) is None


def test_match_returns_other_when_probe_above_other_threshold() -> None:
    g = CrossClipBodyGallery(settings=_settings(other_threshold=0.80))
    pid = uuid4()
    base = _unit_vec(seed=11)
    g.harvest(patient_id=pid, embedding=base, label=LABEL_OTHER, confidence=0.0)
    strong_probe = _aligned_vec(base, similarity=0.90)
    result = g.match(patient_id=pid, probe_embedding=strong_probe)
    assert result is not None
    assert result.label == LABEL_OTHER
    assert result.score >= 0.80
    assert result.cluster_size == 1


def test_match_returns_patient_when_probe_above_patient_threshold() -> None:
    g = CrossClipBodyGallery(settings=_settings(
        patient_threshold=0.72, other_threshold=0.80,
    ))
    pid = uuid4()
    patient_base = _unit_vec(seed=20)
    g.harvest(patient_id=pid, embedding=patient_base,
              label=LABEL_PATIENT, confidence=0.85)
    moderate_probe = _aligned_vec(patient_base, similarity=0.75)
    result = g.match(patient_id=pid, probe_embedding=moderate_probe)
    assert result is not None
    assert result.label == LABEL_PATIENT


def test_match_uses_mean_cosine_across_cluster_buffer() -> None:
    """The whole point of clusters: matching against the MEAN of
    multiple embeddings in a cluster is more stable than matching
    against a single embedding. Probe scoring high vs cluster's mean
    even when individual member scores are lower.

    Construct a cluster from three slightly-different views of the
    same person; probe that's similarity 0.85 to the mean would
    individually score lower against each (because each view differs).
    """
    g = CrossClipBodyGallery(settings=_settings(
        cluster_merge_threshold=0.50,    # easy merge
        dedup_threshold=0.99,
        other_threshold=0.80,
    ))
    pid = uuid4()
    base = _unit_vec(seed=500)
    # Three views, each 0.90 cosine to base — all merge into one cluster.
    g.harvest(patient_id=pid, embedding=_aligned_vec(base, 0.90),
              label=LABEL_OTHER, confidence=0.0)
    g.harvest(patient_id=pid, embedding=_aligned_vec(base, 0.91),
              label=LABEL_OTHER, confidence=0.0)
    g.harvest(patient_id=pid, embedding=_aligned_vec(base, 0.92),
              label=LABEL_OTHER, confidence=0.0)
    assert g.cluster_count(pid) == 1
    assert g.size(pid) == 3
    # Probe close to `base` itself.
    probe = _aligned_vec(base, similarity=0.93)
    result = g.match(patient_id=pid, probe_embedding=probe)
    assert result is not None
    assert result.label == LABEL_OTHER
    assert result.cluster_size == 3
    # The reported score is the mean cosine across the 3-embedding
    # cluster — should be at least the OTHER threshold.
    assert result.score >= 0.80


# ── Match: asymmetric thresholds ─────────────────────────────────


def test_match_other_threshold_higher_than_patient_threshold() -> None:
    """A probe at 0.75 matches PATIENT (threshold 0.72) but NOT OTHER
    (threshold 0.80). Asymmetric thresholds are a safety property:
    OTHER false positive = LLM skipped on real patient (severe);
    PATIENT false positive = LLM runs with patient context anyway
    (mild)."""
    g = CrossClipBodyGallery(settings=_settings(
        patient_threshold=0.72, other_threshold=0.80,
    ))
    pid = uuid4()
    base_patient = _unit_vec(seed=30)
    base_other = _unit_vec(seed=31)
    g.harvest(patient_id=pid, embedding=base_patient, label=LABEL_PATIENT, confidence=0.85)
    g.harvest(patient_id=pid, embedding=base_other, label=LABEL_OTHER, confidence=0.0)
    probe_near_other = _aligned_vec(base_other, similarity=0.75)
    result = g.match(patient_id=pid, probe_embedding=probe_near_other)
    assert (result is None) or (result.label == LABEL_PATIENT)


def test_match_prefers_patient_on_close_tie() -> None:
    """When PATIENT and OTHER both match within 0.05, PATIENT wins.
    Safety tie-break: when in doubt, run the LLM with patient context."""
    g = CrossClipBodyGallery(settings=_settings(
        patient_threshold=0.72, other_threshold=0.72,
    ))
    pid = uuid4()
    shared = _unit_vec(seed=40)
    g.harvest(patient_id=pid, embedding=_aligned_vec(shared, 0.92),
              label=LABEL_PATIENT, confidence=0.9)
    g.harvest(patient_id=pid, embedding=_aligned_vec(shared, 0.94),
              label=LABEL_OTHER, confidence=0.0)
    result = g.match(patient_id=pid, probe_embedding=shared)
    assert result is not None
    assert result.label == LABEL_PATIENT


def test_match_other_wins_when_clearly_better() -> None:
    g = CrossClipBodyGallery(settings=_settings(
        patient_threshold=0.72, other_threshold=0.72,
    ))
    pid = uuid4()
    other_base = _unit_vec(seed=50)
    patient_base = _unit_vec(seed=51)
    g.harvest(patient_id=pid, embedding=other_base,
              label=LABEL_OTHER, confidence=0.0)
    g.harvest(patient_id=pid, embedding=patient_base,
              label=LABEL_PATIENT, confidence=0.9)
    probe = _aligned_vec(other_base, similarity=0.95)
    result = g.match(patient_id=pid, probe_embedding=probe)
    assert result is not None
    assert result.label == LABEL_OTHER


# ── TTL eviction ─────────────────────────────────────────────────


def _stale_cluster(label: str, seed: int, age_s: float = 3600.0) -> _BodyCluster:
    """Build a cluster with `last_updated_at` set to `age_s` in the
    past — for testing TTL paths without needing to wait."""
    return _BodyCluster(
        label=label,
        embeddings=[_unit_vec(seed=seed)],
        seed_at=time.monotonic() - age_s,
        last_updated_at=time.monotonic() - age_s,
        seed_confidence=0.0,
    )


def test_ttl_eviction_drops_expired_clusters_on_match() -> None:
    g = CrossClipBodyGallery(settings=_settings(ttl=60))
    pid = uuid4()
    g._clusters[pid] = [_stale_cluster(LABEL_OTHER, seed=60, age_s=3600)]
    fresh = _unit_vec(seed=61)
    # match() triggers gc; expired cluster dropped, fresh probe
    # doesn't match (different unit vectors).
    assert g.match(patient_id=pid, probe_embedding=fresh) is None
    assert g.cluster_count(pid) == 0
    assert g.size(pid) == 0


def test_gc_evicts_only_expired_clusters() -> None:
    g = CrossClipBodyGallery(settings=_settings(ttl=60))
    pid = uuid4()
    g._clusters[pid] = [
        _stale_cluster(LABEL_OTHER, seed=70, age_s=3600),    # stale
        _BodyCluster(
            label=LABEL_OTHER, embeddings=[_unit_vec(seed=71)],
            seed_at=time.monotonic(), last_updated_at=time.monotonic(),
            seed_confidence=0.0,
        ),
    ]
    assert g.gc(pid) == 1
    assert g.cluster_count(pid) == 1
    assert g.size(pid) == 1


def test_gc_handles_unknown_patient_id() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    assert g.gc(uuid4()) == 0


def test_gc_all_patients_with_none_arg() -> None:
    g = CrossClipBodyGallery(settings=_settings(ttl=60))
    p1, p2 = uuid4(), uuid4()
    g._clusters[p1] = [_stale_cluster(LABEL_OTHER, seed=80, age_s=3600)]
    g._clusters[p2] = [
        _stale_cluster(LABEL_OTHER, seed=81, age_s=3600),
        _stale_cluster(LABEL_OTHER, seed=82, age_s=3600),
    ]
    assert g.gc(None) == 3
    assert g.cluster_count(p1) == 0
    assert g.cluster_count(p2) == 0


def test_active_cluster_extends_ttl_via_last_updated_at() -> None:
    """A cluster that gets a new embedding harvested into it should
    stay alive past the TTL measured from its seed time — TTL is
    measured from last_updated_at, not seed_at, so active identities
    don't expire."""
    g = CrossClipBodyGallery(settings=_settings(
        ttl=60, cluster_merge_threshold=0.50, dedup_threshold=0.99,
    ))
    pid = uuid4()
    base = _unit_vec(seed=600)
    # Seed an old cluster manually with seed_at long ago but
    # last_updated_at recent (as if it kept getting harvested into).
    now = time.monotonic()
    g._clusters[pid] = [
        _BodyCluster(
            label=LABEL_OTHER,
            embeddings=[base],
            seed_at=now - 3600,
            last_updated_at=now - 10,    # recent activity
            seed_confidence=0.0,
        )
    ]
    g.gc(pid)
    assert g.cluster_count(pid) == 1   # alive because last_updated_at recent


# ── Per-label max-clusters cap ───────────────────────────────────


def test_max_clusters_evicts_least_recently_updated_same_label() -> None:
    """Cluster cap is per-label. Excess clusters → evict the
    least-recently-updated. Mirrors LRU semantics."""
    g = CrossClipBodyGallery(settings=_settings(
        max_clusters_per_label=3,
        cluster_merge_threshold=0.95,   # very strict — every harvest seeds new
    ))
    pid = uuid4()
    # 5 distinct people → 5 distinct clusters before cap kicks in.
    for i in range(5):
        g.harvest(
            patient_id=pid, embedding=_unit_vec(seed=100 + i),
            label=LABEL_OTHER, confidence=0.0,
        )
    # Cap at 3 → only the 3 most-recently-updated survive.
    assert g.cluster_label_counts(pid) == {LABEL_PATIENT: 0, LABEL_OTHER: 3}


def test_max_clusters_per_label_independent() -> None:
    """PATIENT cap and OTHER cap are separate — filling OTHER doesn't
    push PATIENT out and vice versa."""
    g = CrossClipBodyGallery(settings=_settings(
        max_clusters_per_label=2,
        cluster_merge_threshold=0.95,
    ))
    pid = uuid4()
    for i in range(3):
        g.harvest(
            patient_id=pid, embedding=_unit_vec(seed=110 + i),
            label=LABEL_PATIENT, confidence=0.9,
        )
    for i in range(3):
        g.harvest(
            patient_id=pid, embedding=_unit_vec(seed=120 + i),
            label=LABEL_OTHER, confidence=0.0,
        )
    counts = g.cluster_label_counts(pid)
    assert counts[LABEL_PATIENT] == 2
    assert counts[LABEL_OTHER] == 2


# ── Patient isolation ────────────────────────────────────────────


def test_galleries_per_patient_are_isolated() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    p_a, p_b = uuid4(), uuid4()
    shared_emb = _unit_vec(seed=200)
    g.harvest(patient_id=p_a, embedding=shared_emb,
              label=LABEL_OTHER, confidence=0.0)
    assert g.match(patient_id=p_b, probe_embedding=shared_emb) is None
    assert g.match(patient_id=p_a, probe_embedding=shared_emb) is not None


# ── Clear ────────────────────────────────────────────────────────


def test_clear_patient_drops_only_that_patient() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    p_a, p_b = uuid4(), uuid4()
    g.harvest(patient_id=p_a, embedding=_unit_vec(seed=300),
              label=LABEL_OTHER, confidence=0.0)
    g.harvest(patient_id=p_b, embedding=_unit_vec(seed=301),
              label=LABEL_OTHER, confidence=0.0)
    g.clear(p_a)
    assert g.size(p_a) == 0
    assert g.size(p_b) == 1


def test_clear_all_with_none_arg() -> None:
    g = CrossClipBodyGallery(settings=_settings())
    for _ in range(3):
        g.harvest(patient_id=uuid4(), embedding=_unit_vec(seed=400),
                  label=LABEL_OTHER, confidence=0.0)
    g.clear(None)
    assert g._clusters == {}
