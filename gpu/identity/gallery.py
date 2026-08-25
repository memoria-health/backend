"""Cross-clip body Re-ID gallery — persistent identity across clips.

Solves the "we identify the helper in clip N, throw the knowledge away
at the clip boundary, then waste an LLM call in clip N+1 when their
face is briefly hidden" problem documented at 2026-05-29 with the
helper-medication false-positive incident.

POOL architecture (identity-persistence W4, 2026-07-19 — replaces the
original cluster design):

  - One flat POOL per patient of up to `reid_body_pool_max` OSNet
    embeddings harvested across clips/poses, FIFO-evicted, TTL-expired
    per embedding.
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

MULTI-PATIENT (2026-08-20). A camera can watch more than one resident
— a shared lounge is one camera, not one per person — so `match` takes
the SET of patients that camera covers and asks the same binary
question of each pool in turn: is this body that person? The best
score above the threshold wins; below it for everyone, the answer is
nobody. There is no default and no fallback to any "primary" patient:
an unrecognized body stays unrecognized, because a wrong attribution
lands a routine event on someone else's record and an UNKNOWN one
merely stays quiet.

  Cost note: one patient means one pool scored, byte-for-byte the
  pre-2026-08-20 path. N patients mean N top-3 scans over ~30 vectors
  — microseconds, against a face pipeline costing ~25ms per crop.

  Accuracy note: an unenrolled stranger now gets N chances to clear
  the bar instead of one, so false accepts scale with the number of
  enrolled residents on a camera. That is inherent to asking a
  multi-way question and is a tuning target once measured, not
  something to pre-correct here (Aryaman, 2026-08-20).

WHAT DIED WITH THE LABELS (2026-08-20). This gallery used to key each
patient's pools by PATIENT / OTHER, with an asymmetric OTHER threshold,
a shorter OTHER TTL, a separate OTHER cap, and a "prefer PATIENT within
0.05" tie-break. The identity node never wrote or read an OTHER pool —
it harvested LABEL_PATIENT only and ignored any non-PATIENT verdict —
so every one of those was unreachable. They are gone rather than
generalized: "not this patient" is now simply "below the threshold",
which is also what makes the multi-patient question well-posed.

Harvest quality: the caller (loop.py) only harvests from face locks
that passed the seat's face-confirmation bar — marginal locks can no
longer poison the pool (the lunch green-box lesson).

TTL: 4 hours per embedding. Covers a typical contiguous "outfit
period" but expires before a clothing change makes embeddings stale.

OSNet (the appearance model BoT-SORT uses internally) provides the
512-dim L2-normalized body embeddings.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from src.services.identity import trace

if TYPE_CHECKING:
    from src.settings import Settings

log = logging.getLogger(__name__)

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
    """Which patient this body is, and how strongly.

    `score` is the top-k mean cosine against that patient's pool.
    `pool_size` is the pool's population (a high score against a
    1-embedding pool is weaker evidence than the same score against
    30)."""

    patient_id: UUID
    score: float
    source_age_s: float
    pool_size: int


class CrossClipBodyGallery:
    """In-memory body embedding pools, one per patient.

    Thread-safe via a single `threading.Lock` — the body pipeline that
    reads this runs inside `asyncio.to_thread` (sync context) while the
    post-clip harvest runs on the event loop. The critical sections are
    microsecond-scale list operations.
    """

    def __init__(self, *, settings: "Settings") -> None:
        self._settings = settings
        # patient_id → list[_PooledEmbedding] (insertion order = harvest
        # order; eviction picks a victim below the golden cut).
        self._pools: dict[UUID, list[_PooledEmbedding]] = {}
        # MOT re-bind lane (2026-07-23): a patient's MOST-RECENT body
        # embeddings, populated every tick they're matched, SHORT-TTL. On a
        # track break + re-entry, a returning body matches THIS (temporal
        # locality → high cosine) instead of re-climbing the stale harvest
        # pool for ~6 ticks. patient_id → list (FIFO cap, TTL).
        # Kills the re-acquire trail for short breaks; a cold long-absence
        # re-entry (buffer expired) still climbs the pool, as before.
        self._recent: dict[UUID, list[_PooledEmbedding]] = {}
        # patient_id → (window_start, accepted_count). The hygiene rate
        # cap is BURST-AWARE: one lock event may deposit up to
        # `reid_gallery_harvest_burst` poses (the multi-pose design),
        # then the window closes for
        # `reid_gallery_harvest_min_interval_s`.
        self._harvest_window: dict[UUID, tuple[float, int]] = {}
        self._lock = threading.Lock()

    # ── reading ────────────────────────────────────────────────

    def match(
        self,
        *,
        patient_ids: Sequence[UUID],
        probe_embedding: np.ndarray,
    ) -> GalleryMatch | None:
        """Best-scoring patient for `probe_embedding`, or None if no
        pool clears `reid_body_match_patient_threshold`.

        `patient_ids` is every resident the camera covers. Each is asked
        the same binary question against their own pool; the winner is
        the highest score, and "nobody" is a first-class answer.

        Always logs the attempt at INFO so actual cosine scores are
        tunable from production data, not theory.
        """
        s = self._settings
        probe = probe_embedding.astype(np.float32)
        scored: list[tuple[UUID, float, float, int]] = []
        with self._lock:
            now = time.monotonic()
            ttl_r = float(getattr(s, "reid_reacquire_recent_ttl_s", 120.0))
            for pid in patient_ids:
                self._gc_locked(pid)
                # MOT re-bind: this patient's short-TTL recent embeddings
                # join their scoring set, so a returning body matches its
                # last-seen self (high cosine) instead of re-climbing the
                # harvest pool. Prune by TTL here (match is the hot path).
                # Empty lane => byte-for-byte the pre-MOT behaviour.
                recent = [
                    e for e in self._recent.get(pid, [])
                    if (now - e.harvested_at) < ttl_r
                ]
                self._recent[pid] = recent
                # Lean eval rig: the recent-lane is a re-acquire
                # optimization we're measuring the system WITHOUT. Score
                # against the raw harvest pool only, so a returning body
                # must earn its match.
                if getattr(s, "reid_lean_eval_mode", False):
                    recent = []
                pool = list(self._pools.get(pid, [])) + recent
                if not pool:
                    continue
                sims = sorted(
                    (float(np.dot(e.embedding, probe)), e.harvested_at)
                    for e in pool
                )[-_MATCH_TOP_K:]
                scored.append((
                    pid,
                    float(np.mean([sim for sim, _ in sims])),
                    now - max(at for _, at in sims),
                    len(pool),
                ))

        # Greedy pick (Aryaman, 2026-08-20): highest score above the bar
        # takes it. Sorting by (score, patient_id) rather than max() keeps
        # an exact numeric tie from depending on the order the caller
        # happened to pass the patients in.
        threshold = s.reid_body_match_patient_threshold
        eligible = sorted(
            (r for r in scored if r[1] >= threshold),
            key=lambda r: (r[1], str(r[0])),
        )
        verdict = (
            GalleryMatch(
                patient_id=eligible[-1][0], score=eligible[-1][1],
                source_age_s=eligible[-1][2], pool_size=eligible[-1][3],
            )
            if eligible else None
        )

        log.info(
            "cross_clip_gallery.match_attempt scores=%s thr=%.2f verdict=%s",
            [(str(pid)[:8], round(sc, 3), n) for pid, sc, _, n in scored],
            threshold,
            (
                f"{str(verdict.patient_id)[:8]}@{verdict.score:.3f}"
                if verdict else "no_match"
            ),
        )
        trace.body_score(
            scores={str(pid): sc for pid, sc, _, _ in scored},
            verdict=(
                f"{str(verdict.patient_id)[:8]}@{verdict.score:.3f}"
                if verdict else "no_match"
            ),
        )
        return verdict

    # ── writing ────────────────────────────────────────────────

    def harvest(
        self,
        *,
        patient_id: UUID,
        embedding: np.ndarray,
        confidence: float,
    ) -> bool:
        """Absorb a confirmed body embedding into that patient's pool.

        Dedup: skipped when it cosine-matches any existing member above
        `reid_body_dedup_threshold` (keeps the pool diverse — N
        representative views, not N copies of one frame). Evicts past
        `reid_body_pool_max`. Returns True iff the pool changed.
        """
        s = self._settings
        normalized = embedding.astype(np.float32)
        with self._lock:
            self._gc_locked(patient_id)
            pool = self._pools.setdefault(patient_id, [])
            # Harvest hygiene (2026-07-19 late soak): unthrottled, the
            # pool absorbed embeddings every tick — eviction churned out
            # good entries within minutes. Burst-aware cap: up to
            # `harvest_burst` accepts per window (one lock event's
            # multi-pose deposit), then the window closes.
            now = time.monotonic()
            min_gap = s.reid_gallery_harvest_min_interval_s
            burst = getattr(s, "reid_gallery_harvest_burst", 4)
            win_start, win_n = self._harvest_window.get(patient_id, (0.0, 0))
            if (now - win_start) >= min_gap:
                win_start, win_n = now, 0
            if win_n >= burst:
                return False
            for e in pool:
                if float(np.dot(e.embedding, normalized)) > s.reid_body_dedup_threshold:
                    log.debug(
                        "cross_clip_gallery.harvest_dedup patient=%s "
                        "pool_size=%d",
                        patient_id, len(pool),
                    )
                    return False
            pool.append(_PooledEmbedding(
                embedding=normalized,
                harvested_at=now,
                confidence=confidence,
            ))
            self._harvest_window[patient_id] = (win_start, win_n + 1)
            evicted = 0
            while len(pool) > s.reid_body_pool_max:
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
            "cross_clip_gallery.harvested patient=%s pool_size=%d "
            "evicted=%d confidence=%.3f",
            patient_id, size, evicted, confidence,
        )
        # Pool-write tap: a swap that harvests the wrong person poisons the
        # pool and drives FPs later — logging every write lets the dashboard
        # trace an FP back to the harvest (and the tick) that caused it.
        trace.harvest(tid=0, pool=str(patient_id)[:8], score=confidence)
        return True

    def remember_recent(
        self, *, patient_id: UUID, embedding: np.ndarray, confidence: float,
    ) -> None:
        """MOT re-bind lane: stash this patient's CURRENT body embedding as a
        short-TTL recent view. Called every tick they're matched, so a track
        that breaks and re-enters matches THIS (temporal locality) and
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
            for pid in list(self._pools.keys() | self._recent.keys()):
                total += self._gc_locked(pid)
            return total

    def _gc_locked(self, patient_id: UUID) -> int:
        s = self._settings
        now = time.monotonic()
        evicted = 0
        # MOT recent (re-bind) lane — short TTL; prune even if the pool is
        # empty so an idle patient's lane can't survive a clothing change.
        ttl_r = float(getattr(s, "reid_reacquire_recent_ttl_s", 120.0))
        lane = self._recent.get(patient_id)
        if lane:
            kept_r = [e for e in lane if (now - e.harvested_at) < ttl_r]
            evicted += len(lane) - len(kept_r)
            self._recent[patient_id] = kept_r
        pool = self._pools.get(patient_id)
        if pool:
            kept = [
                e for e in pool
                if (now - e.harvested_at) < s.reid_body_gallery_ttl_seconds
            ]
            evicted += len(pool) - len(kept)
            self._pools[patient_id] = kept
        if evicted:
            log.debug(
                "cross_clip_gallery.gc patient=%s evicted=%d",
                patient_id, evicted,
            )
        return evicted

    # ── introspection (diagnostics + tests) ────────────────────

    def size(self, patient_id: UUID) -> int:
        """Embeddings stored for `patient_id`."""
        with self._lock:
            return len(self._pools.get(patient_id, []))

    def clear(self, patient_id: UUID | None = None) -> None:
        with self._lock:
            if patient_id is None:
                self._pools.clear()
                self._recent.clear()
            else:
                self._pools.pop(patient_id, None)
                self._recent.pop(patient_id, None)
