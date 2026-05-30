"""Cross-clip body Re-ID gallery — persistent identity across clips.

Solves the "we identify the helper in clip N, throw the knowledge away
at the clip boundary, then waste an LLM call in clip N+1 when their
face is briefly hidden" problem documented at 2026-05-29 with the
helper-medication false-positive incident.

Architecture (clusters, not flat entries):

  - One gallery per patient.
  - Each gallery is a list of identity *clusters*, each cluster
    representing one believed-distinct person. A cluster holds a
    buffer of up to N OSNet embeddings of that person captured across
    different clips/poses. Matching uses the mean cosine of the probe
    against ALL embeddings in the cluster — far more stable than
    matching against a single "best frame" embedding.
  - On harvest, we find the best-matching same-label cluster. If the
    score is above `cluster_merge_threshold`, we append the new
    embedding to that cluster's buffer (FIFO eviction at the cap).
    Otherwise we seed a new cluster — i.e. we believe this is a
    different person.

Why clusters: a single embedding per gallery entry is fragile.
Same person at different angles can produce embeddings only ~0.75
cosine to each other (well below the 0.80 OTHER threshold). With
8 embeddings averaged into a cluster mean, the same probe scores
~0.85 against the cluster — robust enough to actually fire SKIPs
across clips.

Safety asymmetry on thresholds (unchanged from flat design):

  - PATIENT match (default 0.72): liberal. A false-positive "this
    body is the patient" makes the LLM run WITH patient context — at
    worst slight confusion in attribution, no missed events.
  - OTHER match (default 0.80): conservative. A false-positive "this
    body is verified not-patient" makes the gate SKIP the LLM. If the
    body is actually the patient, that's a missed observation. Bias
    toward caution.
  - Cluster merge threshold (default 0.85): tighter than either match
    threshold. We only fuse into an existing identity when we're SURE
    it's the same person — a wrong merge contaminates the cluster
    mean and degrades all future matches against it.
  - Tie-breaker: if both PATIENT and OTHER clusters pass their
    thresholds within 0.05, prefer PATIENT. Always safer to run LLM
    with patient context than to skip.

TTL: 4 hours by default (per cluster, based on last_updated_at).
Covers a typical contiguous "outfit period" but expires before a
clothing change makes embeddings stale. Configurable.

OSNet (the appearance model BoT-SORT uses internally) provides the
512-dim L2-normalized body embeddings. We reuse boxmot's
`PyTorchBackend.get_features` to avoid loading a second OSNet copy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from src.settings import Settings

log = logging.getLogger(__name__)


# Track labels used in the gallery. Mirror `body_pipeline.LABEL_*`
# constants verbatim; duplicated here so this module doesn't have to
# import body_pipeline (avoids a circular dependency if body_pipeline
# ever needs to import the gallery type for hints).
LABEL_PATIENT: str = "PATIENT"
LABEL_OTHER: str = "OTHER"


@dataclass
class _BodyCluster:
    """A buffer of OSNet embeddings believed to belong to one person.

    `embeddings` are L2-normalized so cosine == dot product. FIFO
    eviction past `max_size` keeps the most recent N views.
    `last_updated_at` drives TTL eviction at the cluster level.
    `seed_confidence` records the face-match cosine that originally
    earned the label, for diagnostics.
    """

    label: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    seed_at: float = 0.0
    last_updated_at: float = 0.0
    seed_confidence: float = 0.0

    def size(self) -> int:
        return len(self.embeddings)

    def age_seconds(self, now: float) -> float:
        return now - self.last_updated_at

    def match_score(self, probe: np.ndarray) -> float:
        """Mean cosine of probe against every embedding in the cluster.

        Mean rather than max because max is vulnerable to one outlier
        embedding boosting the cluster's claim; mean across N
        embeddings is empirically more stable for cross-clip ReID.
        Returns 0.0 for an empty cluster (shouldn't happen in practice
        — seeded with one — but defensive).
        """
        if not self.embeddings:
            return 0.0
        return float(np.mean([float(np.dot(e, probe)) for e in self.embeddings]))

    def add_if_diverse(
        self,
        embedding: np.ndarray,
        max_size: int,
        dedup_threshold: float,
        now: float,
    ) -> bool:
        """Append `embedding` to the cluster's buffer iff it is
        sufficiently different from every existing member (cosine <=
        `dedup_threshold` to all). Skipping near-identical adds keeps
        the buffer diverse so the mean represents the person across
        poses, not just N copies of one frame.

        Returns True if added, False if skipped as redundant.
        """
        for e in self.embeddings:
            if float(np.dot(e, embedding)) > dedup_threshold:
                return False
        self.embeddings.append(embedding.astype(np.float32))
        if len(self.embeddings) > max_size:
            # FIFO eviction — keep the most recent N. Older views go
            # first because lighting/pose drift over hours; the
            # newest are the closest match for what the next clip
            # will look like.
            self.embeddings.pop(0)
        self.last_updated_at = now
        return True


@dataclass(frozen=True)
class GalleryMatch:
    """The result of matching a probe embedding against a patient's
    gallery. `score` is the mean cosine against the winning cluster's
    embeddings. `label` is that cluster's label. `cluster_size` is the
    number of embeddings the score was averaged over (useful for
    diagnostics — a high score from a 1-embedding cluster is less
    confident than the same score from an 8-embedding cluster)."""

    label: str
    score: float
    source_age_s: float
    cluster_size: int


class CrossClipBodyGallery:
    """Per-patient in-memory body embedding gallery (cluster-based).

    Thread-safe via a single `threading.Lock` — chosen over
    `asyncio.Lock` because the BoT-SORT body pipeline that needs to
    read this gallery runs inside `asyncio.to_thread` (sync context),
    while the post-clip harvest flow runs on the event loop. A
    threading.Lock works correctly in both, and the critical sections
    are microsecond-scale list operations that never await — no event
    loop blocking concern.

    The gallery is keyed by `patient_id` (UUID). Each patient's
    sub-gallery is a list of `_BodyCluster` rows, capped per label at
    `max_clusters_per_label` and TTL-evicted on every harvest pass.

    Concurrency model:
      - One DetectorWorker process holds one gallery instance.
      - Multiple RoomWorker tasks (one per patient room) and their
        per-tick `to_thread` body pipelines can call `harvest` and
        `match` concurrently. The lock serializes only the dict +
        list mutations; the cosine math + OSNet inference happen
        outside it.
    """

    def __init__(
        self,
        *,
        settings: "Settings",
    ) -> None:
        self._settings = settings
        self._clusters: dict[UUID, list[_BodyCluster]] = {}
        self._lock = threading.Lock()

    # ── reading ────────────────────────────────────────────────

    def match(
        self,
        *,
        patient_id: UUID,
        probe_embedding: np.ndarray,
    ) -> GalleryMatch | None:
        """Find the best gallery cluster for `probe_embedding`. Returns
        None when no cluster meets the per-label threshold OR the
        patient has no gallery yet.

        Asymmetric thresholds (see module docstring): PATIENT is more
        liberal, OTHER is more conservative. PATIENT and OTHER are
        evaluated independently; ties within 0.05 favor PATIENT.

        Always logs the attempt at INFO so we can see actual cosine
        scores in production and tune thresholds from data, not theory.
        """
        s = self._settings
        with self._lock:
            self._gc_locked(patient_id)
            clusters = self._clusters.get(patient_id, [])
            now = time.monotonic()

            best_patient: GalleryMatch | None = None
            best_other: GalleryMatch | None = None
            # Track the raw top scores so we can log even on no-match
            # (debugging "why didn't this fire?" needs to see the
            # near-miss scores, not just successes).
            top_patient_raw = 0.0
            top_other_raw = 0.0

            for cluster in clusters:
                score = cluster.match_score(probe_embedding)
                if cluster.label == LABEL_PATIENT:
                    if score > top_patient_raw:
                        top_patient_raw = score
                    if score >= s.reid_body_match_patient_threshold and (
                        best_patient is None or score > best_patient.score
                    ):
                        best_patient = GalleryMatch(
                            label=LABEL_PATIENT,
                            score=score,
                            source_age_s=cluster.age_seconds(now),
                            cluster_size=cluster.size(),
                        )
                elif cluster.label == LABEL_OTHER:
                    if score > top_other_raw:
                        top_other_raw = score
                    if score >= s.reid_body_match_other_threshold and (
                        best_other is None or score > best_other.score
                    ):
                        best_other = GalleryMatch(
                            label=LABEL_OTHER,
                            score=score,
                            source_age_s=cluster.age_seconds(now),
                            cluster_size=cluster.size(),
                        )

            # PATIENT vs OTHER tie-break: prefer PATIENT when within
            # 0.05. Always safer to run LLM with patient context than
            # to skip.
            verdict: GalleryMatch | None
            if best_patient is not None and best_other is not None:
                if best_patient.score + 0.05 >= best_other.score:
                    verdict = best_patient
                else:
                    verdict = best_other
            else:
                verdict = best_patient or best_other

            n_patient_clusters = sum(1 for c in clusters if c.label == LABEL_PATIENT)
            n_other_clusters = sum(1 for c in clusters if c.label == LABEL_OTHER)

        # Log OUTSIDE the lock — string formatting is non-trivial work
        # and we don't want it serialized with critical section.
        log.info(
            "cross_clip_gallery.match_attempt patient=%s top_patient=%.3f "
            "top_other=%.3f patient_thr=%.2f other_thr=%.2f verdict=%s "
            "clusters_patient=%d clusters_other=%d",
            patient_id,
            top_patient_raw,
            top_other_raw,
            s.reid_body_match_patient_threshold,
            s.reid_body_match_other_threshold,
            (f"{verdict.label}@{verdict.score:.3f}" if verdict else "no_match"),
            n_patient_clusters,
            n_other_clusters,
        )
        return verdict

    # ── writing ────────────────────────────────────────────────

    def harvest(
        self,
        *,
        patient_id: UUID,
        embedding: np.ndarray,
        label: str,
        confidence: float,
    ) -> bool:
        """Absorb a verified-label body embedding into the gallery.

        Behavior:
          1. Compute match score against every existing same-label
             cluster.
          2. If the best score is >= `cluster_merge_threshold`, append
             to that cluster's buffer (subject to intra-cluster
             dedup + per-cluster size cap).
          3. Otherwise, seed a new cluster.
          4. Enforce per-label cluster cap (evict least-recently-
             updated on overflow).

        Returns True if the gallery state changed (new cluster seeded
        OR embedding appended to existing cluster). Returns False on
        invalid label OR when the embedding was deduplicated against
        an existing cluster member.
        """
        if label not in (LABEL_PATIENT, LABEL_OTHER):
            log.warning(
                "cross_clip_gallery.harvest_rejected reason=invalid_label "
                "label=%s patient=%s",
                label, patient_id,
            )
            return False

        s = self._settings
        normalized = embedding.astype(np.float32)
        with self._lock:
            self._gc_locked(patient_id)
            clusters = self._clusters.setdefault(patient_id, [])

            # Find best matching same-label cluster.
            best_cluster: _BodyCluster | None = None
            best_score = -1.0
            for c in clusters:
                if c.label != label:
                    continue
                score = c.match_score(normalized)
                if score > best_score:
                    best_cluster = c
                    best_score = score

            now = time.monotonic()
            if (
                best_cluster is not None
                and best_score >= s.reid_body_cluster_merge_threshold
            ):
                added = best_cluster.add_if_diverse(
                    normalized,
                    max_size=s.reid_body_max_embeddings_per_cluster,
                    dedup_threshold=s.reid_body_dedup_threshold,
                    now=now,
                )
                if not added:
                    log.debug(
                        "cross_clip_gallery.harvest_dedup label=%s "
                        "patient=%s merge_score=%.3f cluster_size=%d",
                        label, patient_id, best_score, best_cluster.size(),
                    )
                    return False
                log.info(
                    "cross_clip_gallery.harvest_merged label=%s patient=%s "
                    "merge_score=%.3f cluster_size=%d confidence=%.3f",
                    label, patient_id, best_score, best_cluster.size(),
                    confidence,
                )
                return True

            # Seed a new cluster.
            new_cluster = _BodyCluster(
                label=label,
                embeddings=[normalized],
                seed_at=now,
                last_updated_at=now,
                seed_confidence=confidence,
            )
            clusters.append(new_cluster)
            self._cap_clusters_per_label_locked(patient_id, label)
            n_label = sum(1 for c in clusters if c.label == label)
            log.info(
                "cross_clip_gallery.harvest_new_cluster label=%s patient=%s "
                "best_existing_score=%.3f merge_thr=%.2f clusters_for_label=%d "
                "confidence=%.3f",
                label, patient_id,
                (best_score if best_cluster is not None else 0.0),
                s.reid_body_cluster_merge_threshold,
                n_label,
                confidence,
            )
            return True

    # ── maintenance ────────────────────────────────────────────

    def gc(self, patient_id: UUID | None = None) -> int:
        """Public GC entry point. When `patient_id` is None, gc all
        patients. Returns the number of clusters evicted.
        """
        with self._lock:
            if patient_id is not None:
                return self._gc_locked(patient_id)
            total = 0
            for pid in list(self._clusters.keys()):
                total += self._gc_locked(pid)
            return total

    def _gc_locked(self, patient_id: UUID) -> int:
        """Evict TTL-expired clusters for `patient_id`. Must be called
        under `self._lock`. Returns count evicted.

        A cluster's TTL is measured from `last_updated_at` so an
        actively-seen identity stays warm; only stale ones expire.
        """
        ttl = self._settings.reid_body_gallery_ttl_seconds
        clusters = self._clusters.get(patient_id)
        if not clusters:
            return 0
        now = time.monotonic()
        kept = [c for c in clusters if (now - c.last_updated_at) < ttl]
        evicted = len(clusters) - len(kept)
        if evicted:
            self._clusters[patient_id] = kept
            log.debug(
                "cross_clip_gallery.gc patient=%s evicted_clusters=%d "
                "remaining=%d",
                patient_id, evicted, len(kept),
            )
        return evicted

    def _cap_clusters_per_label_locked(self, patient_id: UUID, label: str) -> None:
        """Enforce per-label max clusters; evict the least-recently-
        updated. Must be called under `self._lock`."""
        cap = self._settings.reid_body_max_clusters_per_label
        clusters = self._clusters.get(patient_id, [])
        same_label = [(i, c) for i, c in enumerate(clusters) if c.label == label]
        if len(same_label) <= cap:
            return
        # Sort by last_updated_at ascending (stale first), evict the
        # oldest until we're at cap.
        same_label.sort(key=lambda ic: ic[1].last_updated_at)
        n_to_evict = len(same_label) - cap
        to_evict_indices = {ic[0] for ic in same_label[:n_to_evict]}
        self._clusters[patient_id] = [
            c for i, c in enumerate(clusters) if i not in to_evict_indices
        ]

    # ── introspection (diagnostics + tests) ────────────────────

    def size(self, patient_id: UUID) -> int:
        """Total number of embeddings stored for `patient_id` across
        all clusters. (Not cluster count — that's `cluster_count`.)"""
        with self._lock:
            return sum(
                c.size() for c in self._clusters.get(patient_id, [])
            )

    def cluster_count(self, patient_id: UUID) -> int:
        with self._lock:
            return len(self._clusters.get(patient_id, []))

    def label_counts(self, patient_id: UUID) -> dict[str, int]:
        """Return {label: total_embeddings_for_label}. Embeddings, not
        clusters — preserves the semantics tests have always asserted
        against."""
        with self._lock:
            counts: dict[str, int] = {LABEL_PATIENT: 0, LABEL_OTHER: 0}
            for c in self._clusters.get(patient_id, []):
                counts[c.label] = counts.get(c.label, 0) + c.size()
            return counts

    def cluster_label_counts(self, patient_id: UUID) -> dict[str, int]:
        """Return {label: cluster_count_for_label}. Distinct from
        `label_counts` (which counts embeddings). Useful when you want
        to know "how many distinct people do we believe we've seen."""
        with self._lock:
            counts: dict[str, int] = {LABEL_PATIENT: 0, LABEL_OTHER: 0}
            for c in self._clusters.get(patient_id, []):
                counts[c.label] = counts.get(c.label, 0) + 1
            return counts

    def clear(self, patient_id: UUID | None = None) -> None:
        """Clear gallery(ies). Used by tests + manual ops; not part of
        normal hot-path flow."""
        with self._lock:
            if patient_id is None:
                self._clusters.clear()
            else:
                self._clusters.pop(patient_id, None)
