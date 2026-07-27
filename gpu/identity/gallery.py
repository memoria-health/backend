"""Cross-clip body Re-ID gallery — persistent identity across clips.

Solves the "we identify the helper in clip N, throw the knowledge away
at the clip boundary, then waste an LLM call in clip N+1 when their
face is briefly hidden" problem documented at 2026-05-29 with the
helper-medication false-positive incident.

POOL architecture (identity-persistence W4, 2026-07-19 — replaces the
original cluster design):

  - One gallery per patient; per label (PATIENT / OTHER) a flat POOL
    of up to `reid_body_pool_max` OSNet embeddings harvested across
    clips/poses, FIFO-evicted, TTL-expired per embedding.
  - MATCH = mean of the top-3 cosines of the probe against the pool
    (a robust max — one lucky outlier can't carry a match, but the
    best few views decide, DeepStream-style many-vector matching).

Why the cluster design died: it merged a new harvest into an existing
cluster only at cosine ≥0.85, but two real views of the same person
across pose/viewpoint score 0.6-0.75 (soak #2 measurements + torchreid
issue reports) — so every harvest seeded a NEW singleton cluster
(observed: 11 one-embedding clusters), the mean-of-8 stabilization
never happened, and match scores rode a coin flip around the 0.72
threshold. A flat pool matched on best-of needs no merge decision at
all — the failure mode is structurally deleted.

Safety asymmetry on thresholds (unchanged):

  - PATIENT match (default 0.72): liberal. A false-positive "this
    body is the patient" makes the LLM run WITH patient context — at
    worst slight confusion in attribution, no missed events.
  - OTHER match (default 0.80): conservative. A false-positive "this
    body is verified not-patient" makes the gate SKIP the LLM. If the
    body is actually the patient, that's a missed observation. Bias
    toward caution.
  - Tie-breaker: if both PATIENT and OTHER pass their thresholds
    within 0.05, prefer PATIENT. Always safer to run the LLM with
    patient context than to skip.

Harvest quality: the caller (inference.py) only harvests from face
locks that passed the W5 quality floors and the
`reid_harvest_min_confidence` bar — marginal locks can no longer
poison the pool (the lunch green-box lesson).

TTL: 4 hours per embedding. Covers a typical contiguous "outfit
period" but expires before a clothing change makes embeddings stale.

OSNet (the appearance model BoT-SORT uses internally) provides the
512-dim L2-normalized body embeddings.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from src.services.reid import trace

if TYPE_CHECKING:
    from src.settings import Settings

log = logging.getLogger(__name__)


# Track labels used in the gallery. Mirror `body_pipeline.LABEL_*`
# constants verbatim; duplicated here so this module doesn't have to
# import body_pipeline (avoids a circular dependency if body_pipeline
# ever needs to import the gallery type for hints).
LABEL_PATIENT: str = "PATIENT"
LABEL_OTHER: str = "OTHER"

# Top-k cosines averaged for a pool match. 3 = robust max: immune to a
# single outlier, decided by the best few views.
_MATCH_TOP_K = 3


@dataclass
class _PooledEmbedding:
    embedding: np.ndarray            # L2-normalized, float32
    harvested_at: float              # time.monotonic()
    confidence: float                # face-match score that earned it


@dataclass(frozen=True)
class GalleryMatch:
    """The result of matching a probe embedding against a patient's
    gallery. `score` is the top-k mean cosine against the winning
    label's pool. `pool_size` is that pool's population (a high score
    against a 1-embedding pool is weaker evidence than the same score
    against 30)."""

    label: str
    score: float
    source_age_s: float
    pool_size: int


class CrossClipBodyGallery:
    """Per-patient in-memory body embedding pools (PATIENT + OTHER).

    Thread-safe via a single `threading.Lock` — the BoT-SORT body
    pipeline that reads this runs inside `asyncio.to_thread` (sync
    context) while the post-clip harvest runs on the event loop. The
    critical sections are microsecond-scale list operations.
    """

    def __init__(self, *, settings: "Settings") -> None:
        self._settings = settings
        # patient_id → label → list[_PooledEmbedding] (insertion order
        # = harvest order; FIFO eviction pops index 0).
        self._pools: dict[UUID, dict[str, list[_PooledEmbedding]]] = {}
        # MOT re-bind lane (2026-07-23): the patient's MOST-RECENT body
        # embeddings, populated every tick they're matched, SHORT-TTL. On a
        # track break + re-entry, a returning body matches THIS (temporal
        # locality → high cosine) instead of re-climbing the stale harvest
        # pool for ~6 ticks. patient_id → deque-like list (FIFO cap, TTL).
        # Kills the re-acquire trail for short breaks; a cold long-absence
        # re-entry (buffer expired) still climbs the pool, as before.
        self._recent: dict[UUID, list[_PooledEmbedding]] = {}
        # (patient_id, label) → (window_start, accepted_count). The
        # hygiene rate cap is BURST-AWARE: one lock event may deposit
        # up to `reid_gallery_harvest_burst` poses (the multi-pose
        # design), then the window closes for
        # `reid_gallery_harvest_min_interval_s`.
        self._harvest_window: dict[tuple[UUID, str], tuple[float, int]] = {}
        self._lock = threading.Lock()

    # ── reading ────────────────────────────────────────────────

    def match(
        self,
        *,
        patient_id: UUID,
        probe_embedding: np.ndarray,
    ) -> GalleryMatch | None:
        """Best label for `probe_embedding`, or None below thresholds.

        Always logs the attempt at INFO so actual cosine scores are
        tunable from production data, not theory.
        """
        s = self._settings
        probe = probe_embedding.astype(np.float32)
        with self._lock:
            self._gc_locked(patient_id)
            pools = self._pools.get(patient_id, {})
            now = time.monotonic()

            # MOT re-bind: the patient's short-TTL recent embeddings join the
            # PATIENT scoring set, so a returning body matches its last-seen
            # self (high cosine) instead of re-climbing the harvest pool.
            # Prune by TTL here (match is the hot path). Empty lane => this is
            # byte-for-byte the pre-MOT behaviour.
            ttl_r = float(getattr(s, "reid_reacquire_recent_ttl_s", 120.0))
            recent = [
                e for e in self._recent.get(patient_id, [])
                if (now - e.harvested_at) < ttl_r
            ]
            self._recent[patient_id] = recent
            # Lean eval rig: the recent-lane is a re-acquire optimization
            # we're measuring the system WITHOUT. Score against the raw
            # harvest pool only, so a returning body must earn its match.
            if getattr(s, "reid_lean_eval_mode", False):
                recent = []

            def score_pool(label: str, extra=()) -> tuple[float, float, int]:
                pool = list(pools.get(label, [])) + list(extra)
                if not pool:
                    return 0.0, 0.0, 0
                sims = sorted(
                    (float(np.dot(e.embedding, probe)), e.harvested_at)
                    for e in pool
                )[-_MATCH_TOP_K:]
                score = float(np.mean([sim for sim, _ in sims]))
                newest_at = max(at for _, at in sims)
                return score, now - newest_at, len(pool)

            p_score, p_age, p_n = score_pool(LABEL_PATIENT, recent)
            o_score, o_age, o_n = score_pool(LABEL_OTHER)

        best_patient = (
            GalleryMatch(LABEL_PATIENT, p_score, p_age, p_n)
            if p_n and p_score >= s.reid_body_match_patient_threshold
            else None
        )
        best_other = (
            GalleryMatch(LABEL_OTHER, o_score, o_age, o_n)
            if o_n and o_score >= s.reid_body_match_other_threshold
            else None
        )

        # PATIENT vs OTHER tie-break: prefer PATIENT within 0.05 —
        # always safer to run the LLM with patient context than skip.
        verdict: GalleryMatch | None
        if best_patient is not None and best_other is not None:
            verdict = (
                best_patient
                if best_patient.score + 0.05 >= best_other.score
                else best_other
            )
        else:
            verdict = best_patient or best_other

        log.info(
            "cross_clip_gallery.match_attempt patient=%s top_patient=%.3f "
            "top_other=%.3f patient_thr=%.2f other_thr=%.2f verdict=%s "
            "pool_patient=%d pool_other=%d",
            patient_id, p_score, o_score,
            s.reid_body_match_patient_threshold,
            s.reid_body_match_other_threshold,
            (f"{verdict.label}@{verdict.score:.3f}" if verdict else "no_match"),
            p_n, o_n,
        )
        trace.body_score(
            patient=p_score, other=o_score,
            verdict=(f"{verdict.label}@{verdict.score:.3f}" if verdict else "no_match"),
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
        """Absorb a verified-label body embedding into the label's pool.

        Dedup: skipped when it cosine-matches any existing member above
        `reid_body_dedup_threshold` (keeps the pool diverse — N
        representative views, not N copies of one frame). FIFO past
        `reid_body_pool_max`. Returns True iff the pool changed.
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
            pool = self._pools.setdefault(patient_id, {}).setdefault(
                label, [],
            )
            # Harvest hygiene (2026-07-19 late soak): unthrottled, the
            # pool absorbed embeddings every tick — FIFO churned out
            # good entries within minutes. Burst-aware cap: up to
            # `harvest_burst` accepts per window (one lock event's
            # multi-pose deposit), then the window closes.
            now = time.monotonic()
            min_gap = s.reid_gallery_harvest_min_interval_s
            burst = getattr(s, "reid_gallery_harvest_burst", 4)
            key = (patient_id, label)
            win_start, win_n = self._harvest_window.get(key, (0.0, 0))
            if (now - win_start) >= min_gap:
                win_start, win_n = now, 0
            if win_n >= burst:
                return False
            for e in pool:
                if float(np.dot(e.embedding, normalized)) > s.reid_body_dedup_threshold:
                    log.debug(
                        "cross_clip_gallery.harvest_dedup label=%s "
                        "patient=%s pool_size=%d",
                        label, patient_id, len(pool),
                    )
                    return False
            pool.append(_PooledEmbedding(
                embedding=normalized,
                harvested_at=now,
                confidence=confidence,
            ))
            self._harvest_window[key] = (win_start, win_n + 1)
            cap = (
                s.reid_other_pool_max if label == LABEL_OTHER
                else s.reid_body_pool_max
            )
            evicted = 0
            while len(pool) > cap:
                # Golden protection: the top-N highest-confidence
                # entries survive eviction — churn removes the oldest
                # of the REST, so the pool never loses its best
                # evidence to a run of mediocre harvests.
                golden_cut = sorted(
                    (e.confidence for e in pool), reverse=True,
                )[:s.reid_gallery_golden_count]
                floor = golden_cut[-1] if golden_cut else float("inf")
                victim_idx = next(
                    (i for i, e in enumerate(pool) if e.confidence < floor),
                    0,
                )
                pool.pop(victim_idx)
                evicted += 1
            size = len(pool)
        log.info(
            "cross_clip_gallery.harvested label=%s patient=%s pool_size=%d "
            "evicted=%d confidence=%.3f",
            label, patient_id, size, evicted, confidence,
        )
        # Pool-write tap: a swap that harvests a guest poisons the pool and
        # drives FPs later — logging every write lets the dashboard trace an
        # FP back to the harvest (and the tick) that caused it.
        trace.harvest(tid=0, pool=label, score=confidence)
        return True

    def remember_recent(
        self, *, patient_id: UUID, embedding: np.ndarray, confidence: float,
    ) -> None:
        """MOT re-bind lane: stash the patient's CURRENT body embedding as a
        short-TTL recent view. Called every tick the patient is matched, so a
        track that breaks and re-enters matches THIS (temporal locality) and
        re-binds fast, instead of re-climbing the harvest pool for ~6 ticks.
        FIFO-capped; TTL-pruned in match()/gc(). Additive: with an empty lane
        match() is byte-for-byte the pre-MOT behaviour."""
        cap = int(getattr(self._settings, "reid_reacquire_recent_max", 6))
        normalized = embedding.astype(np.float32)
        with self._lock:
            lane = self._recent.setdefault(patient_id, [])
            lane.append(_PooledEmbedding(
                embedding=normalized,
                harvested_at=time.monotonic(),
                confidence=confidence,
            ))
            while len(lane) > cap:
                lane.pop(0)

    # ── maintenance ────────────────────────────────────────────

    def gc(self, patient_id: UUID | None = None) -> int:
        """Evict TTL-expired embeddings. Returns count evicted."""
        with self._lock:
            if patient_id is not None:
                return self._gc_locked(patient_id)
            total = 0
            for pid in list(self._pools.keys()):
                total += self._gc_locked(pid)
            return total

    def _gc_locked(self, patient_id: UUID) -> int:
        s = self._settings
        now = time.monotonic()
        evicted = 0
        # MOT recent (re-bind) lane — short TTL; prune even if pools empty so
        # an idle patient's lane can't survive a clothing change.
        ttl_r = float(getattr(s, "reid_reacquire_recent_ttl_s", 120.0))
        lane = self._recent.get(patient_id)
        if lane:
            kept_r = [e for e in lane if (now - e.harvested_at) < ttl_r]
            evicted += len(lane) - len(kept_r)
            self._recent[patient_id] = kept_r
        pools = self._pools.get(patient_id)
        if not pools:
            return evicted
        for label, pool in pools.items():
            # Negative (OTHER) entries are outfit-bound and deliberately
            # SHORT-LIVED (2h vs the patient pool's 4h): a stale guest
            # embedding is wrong-mute risk, not evidence.
            ttl = (
                s.reid_other_pool_ttl_seconds if label == LABEL_OTHER
                else s.reid_body_gallery_ttl_seconds
            )
            kept = [e for e in pool if (now - e.harvested_at) < ttl]
            evicted += len(pool) - len(kept)
            pools[label] = kept
        if evicted:
            log.debug(
                "cross_clip_gallery.gc patient=%s evicted=%d",
                patient_id, evicted,
            )
        return evicted

    # ── introspection (diagnostics + tests) ────────────────────

    def size(self, patient_id: UUID) -> int:
        """Total embeddings stored for `patient_id` across labels."""
        with self._lock:
            return sum(
                len(p) for p in self._pools.get(patient_id, {}).values()
            )

    def cluster_count(self, patient_id: UUID) -> int:
        """Pool-era compatibility: number of non-empty label pools
        (0-2). The cluster concept is gone; callers wanting population
        use `size` / `label_counts`."""
        with self._lock:
            return sum(
                1 for p in self._pools.get(patient_id, {}).values() if p
            )

    def label_counts(self, patient_id: UUID) -> dict[str, int]:
        """{label: embeddings_for_label}."""
        with self._lock:
            counts = {LABEL_PATIENT: 0, LABEL_OTHER: 0}
            for label, pool in self._pools.get(patient_id, {}).items():
                counts[label] = len(pool)
            return counts

    def cluster_label_counts(self, patient_id: UUID) -> dict[str, int]:
        """Pool-era compatibility: {label: 1 if pool non-empty else 0}."""
        with self._lock:
            counts = {LABEL_PATIENT: 0, LABEL_OTHER: 0}
            for label, pool in self._pools.get(patient_id, {}).items():
                counts[label] = 1 if pool else 0
            return counts

    def clear(self, patient_id: UUID | None = None) -> None:
        with self._lock:
            if patient_id is None:
                self._pools.clear()
                self._recent.clear()
            else:
                self._pools.pop(patient_id, None)
                self._recent.pop(patient_id, None)
