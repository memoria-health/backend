"""Bind-once-hold identity ledger — the EdgeTAM-continuous binding brain.

The pod runs one persistent EdgeTAM session per camera and hands the
backend a STABLE ``track_id`` that lives as long as a person stays in
view (SAM3 re-grounds are re-mapped onto the same id by IoU on the pod —
see the pod's ``continuous.py``). So this ledger keys entries DIRECTLY by
the pod ``track_id`` and never re-associates: no geometric stitcher, no
fragment merge, no own persistent-id space.

Two labels only: **PATIENT** or **UNKNOWN** (Aryaman, 2026-07-22 — "we
only care about identifying the patient; everyone else is UNKNOWN"). The
model:

  1. Every new stable track is born UNKNOWN.
  2. A track that arrives FACE-locked (or gallery-POOL matched) to the
     enrolled patient binds PATIENT. Highest score wins the single
     patient seat; a FACE categorically outranks a POOL match; a pool
     (outfit) match alone can never ESTABLISH the patient on a camera
     where the patient is already held elsewhere. That FP guard — never
     let someone ELSE be designated the patient — is the one edge case
     that survives into the EdgeTAM era.
  3. Once bound, the patient is HELD on the stable track unconditionally
     until the track dies (retire on grace). No mid-track face
     re-litigation — the northstar: "stay locked no matter where they
     go or what they do". This is what stops the couch-at-20ft
     false-unbind: a face that fails at distance never drops a lock
     earned at entry.

2026-08-14 deletions (Aryaman: "delete the useless patches"): the
scene-extension and the blackout carry are gone. Both were written for
a payoff that no longer exists — a pod track number NEVER returns after
death in the EdgeTAM-continuous era, and the carry's 5s physical window
could not fire when judged at the backend's 10s tick cadence (dead on
arrival, verified 2026-08-14). The retire grace below survives with ONE
remaining job, documented at _retire.

Deleted vs the track-anchored ledger (all compensations for SAM3's
per-clip renumbering, dissolved by EdgeTAM's stable ids): the geometric
boundary stitcher + velocity extrapolation, fragment merge, the two-way
mismatch/smallface accumulators, LABEL_OTHER and the verified-OTHER audit.

Same seam as before: ``apply(identity, tick_mono)`` folds bindings into
the tick's IdentityResult; ``stash_fields()`` feeds the analyses row.
Pure in-memory, single-writer. Emergency detectors never depend on this.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING

from src.services.detection.patient_presence import (
    TIER_BODY,
    TIER_FACE,
    TIER_HELD,
)
from src.services.identity import trace
from src.services.identity.observations import LABEL_PATIENT, LABEL_UNKNOWN

if TYPE_CHECKING:
    from src.services.identity.result import IdentityResult

log = logging.getLogger(__name__)

# A bound patient may donate body crops to the pool only this soon after
# face-grade evidence (pool-provenance bindings never harvest — the
# pool-corruption guard: no pool->pool self-feeding, ever).
_HARVEST_AFTERGLOW_S = 60.0

# Evidence tiers for the single-patient contest: a FACE lock categorically
# outranks a POOL (outfit) match. Face cosines (~0.45-0.6) and pool cosines
# (~0.72+) are different scales and must never be compared numerically —
# 2026-07-22 live finding: a 0.72 pool bind stole from a 0.54 face bind.
_TIER = {"face": 2, "pool": 1}


def _bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


@dataclasses.dataclass
class _Entry:
    pid: int                 # == the pod's stable track_id (no own id space)
    binding: str             # LABEL_PATIENT | LABEL_UNKNOWN
    provenance: str | None   # "face" | "pool" | None
    confidence: float
    bound_at: float
    first_seen: float
    last_seen: float
    last_bbox: tuple[float, float, float, float]
    # Anchor for the harvest afterglow (last face-grade confirmation).
    last_face_confirm: float = 0.0


class TrackLedger:
    """Per-camera bind-once-hold ledger, keyed by the pod's stable ids."""

    def __init__(self, *, camera_id, registry, settings) -> None:
        self._camera_id = camera_id
        self._registry = registry
        self._settings = settings
        self._entries: dict[int, _Entry] = {}
        self._last_tick_mono = 0.0
        self._last_patient_event: str | None = None
        # Throttle + truth state (read by CameraIngest before each tick).
        self._patient_locked_last_tick = False
        self._patient_seen_last_tick = False
        self._new_birth_last_tick = False
        self._face_heartbeat = 0
        self._last_apply_now = 0.0

        g = lambda n, d: float(getattr(settings, n, d))  # noqa: E731
        self._retire_after_s = g("reid_binding_retire_after_s", 45.0)

    # ── external surface (IdentityLock-compatible seam) ────────────────

    @property
    def is_active(self) -> bool:
        """True while a PATIENT binding is live (the W2 freeze gate)."""
        return any(e.binding == LABEL_PATIENT for e in self._entries.values())

    @property
    def patient_seen(self) -> bool:
        """Was the patient actually SEEN on the last processed tick? Feeds
        the home registry's holder_seen heartbeat."""
        return self._patient_seen_last_tick

    def all_bound(self) -> bool:
        """Lean-sampling trigger. In the PATIENT/UNKNOWN model 'everyone
        identified' collapses to 'the patient is held' — guests stay
        UNKNOWN and need no per-tick facing once the patient is secured."""
        return self._patient_locked_last_tick

    def wants_face_evidence(self) -> bool:
        """Binding-aware face throttle. Full cadence while the patient is
        NOT locked (still searching) or a new track just appeared (a
        possible patient entry); a 1-in-3 heartbeat once the patient is
        held and the scene is stable. Called by CameraIngest before each
        tick; reads the previous tick's outcome."""
        if not self._patient_locked_last_tick or self._new_birth_last_tick:
            self._face_heartbeat = 0
            return True
        self._face_heartbeat += 1
        if self._face_heartbeat >= 3:
            self._face_heartbeat = 0
            return True
        return False

    def harvest_hint(self) -> tuple | None:
        """((x0,y0,x1,y1), confidence) for the patient's last box while
        the binding is in FACE afterglow. Pool-provenance never emits — the
        pool must not feed itself (Aryaman's corruption guard)."""
        e = next(
            (x for x in self._entries.values() if x.binding == LABEL_PATIENT),
            None,
        )
        if (
            e is None
            or e.provenance != "face"
            or not self._patient_seen_last_tick
            or self._last_apply_now - e.last_face_confirm > _HARVEST_AFTERGLOW_S
        ):
            return None
        return (e.last_bbox, float(e.confidence))

    def apply(
        self, identity: IdentityResult, *, tick_mono: float | None = None,
    ) -> IdentityResult:
        """Fold this tick's tracks onto the ledger by stable id, bind/hold
        the patient, and rewrite the IdentityResult through the bindings.
        Never raises — a ledger bug must not break a tick."""
        try:
            return self._apply_inner(identity, tick_mono)
        except Exception:
            log.exception("track_ledger.apply failed cam=%s", self._camera_id)
            return identity

    # ── core ───────────────────────────────────────────────────────────

    def _apply_inner(self, identity, tick_mono):
        if not identity.reid_ran:
            return identity          # infra failure != evidence: freeze.
        now = tick_mono if tick_mono is not None else time.monotonic()
        if tick_mono is not None:
            if tick_mono < self._last_tick_mono:
                # Stale window: read-only fold, never mutate state.
                return self._rewrite(identity, degraded_hold=True)
            self._last_tick_mono = tick_mono

        tracks = tuple(identity.tracks or ())
        degraded = bool(identity.quality_degraded)

        seen: set[int] = set()
        born = False
        # Largest box first so a body-level bind leads if the pod ever
        # emits two ids for one person on a tick (defensive; EdgeTAM
        # normally gives exactly one id per person).
        ordered = sorted(
            (t for t in tracks if t.bbox_series),
            key=lambda t: _bbox_area(t.bbox_series[-1][1]),
            reverse=True,
        )
        for t in ordered:
            tid = int(t.track_id)
            e = self._entries.get(tid)
            if e is None:
                e = self._birth(tid, t, now)
                born = True
            else:
                e.last_seen = now
                e.last_bbox = t.bbox_series[-1][1]
            self._fold_evidence(e, t, degraded)
            seen.add(tid)

        self._takeover_check(tracks)
        if not degraded:
            self._retire(now, seen)

        # Truth + throttle state for the next tick.
        self._last_apply_now = now
        self._new_birth_last_tick = born
        seen_patient = next(
            (
                self._entries[p] for p in seen
                if p in self._entries
                and self._entries[p].binding == LABEL_PATIENT
            ),
            None,
        )
        self._patient_seen_last_tick = seen_patient is not None
        self._patient_locked_last_tick = self.is_active

        # Keep the home-registry seat warm while the patient is actually in
        # view (parity with the old lock's TIER_HELD refresh). Runs AFTER
        # the takeover check so an outclaimed camera never re-warms a seat
        # it just lost.
        if self._registry is not None and seen_patient is not None:
            self._registry.claim(
                self._camera_id, tier=TIER_HELD,
                confidence=seen_patient.confidence,
            )
        return self._rewrite(identity, degraded_hold=degraded)

    def _birth(self, tid: int, t, now: float) -> _Entry:
        e = _Entry(
            pid=tid,
            binding=LABEL_UNKNOWN,
            provenance=None,
            confidence=0.0,
            bound_at=now,
            first_seen=now,
            last_seen=now,
            last_bbox=t.bbox_series[-1][1],
        )
        self._entries[tid] = e
        return e

    def _fold_evidence(self, e: _Entry, t, degraded: bool) -> None:
        """Bind PATIENT on a fresh face lock or gallery-pool match; else
        HOLD. No OTHER, no accumulators, no mid-track audit — a lock earned
        at entry is held on the stable track until it dies."""
        fresh_face = t.label == LABEL_PATIENT and t.face_locked
        fresh_pool = (
            t.label == LABEL_PATIENT
            and t.pre_labeled
            and t.pre_label_source == "gallery"
        )
        if not (fresh_face or fresh_pool):
            return
        # FP guard: an outfit (pool) match alone may MAINTAIN but never
        # ESTABLISH the patient on a camera where the patient is confirmed
        # elsewhere — only a FACE crosses the single-holder boundary.
        establishing = e.binding != LABEL_PATIENT
        if (
            fresh_pool
            and not fresh_face
            and establishing
            and self._registry is not None
            and self._registry.confirmed_elsewhere(self._camera_id)
        ):
            log.info(
                "track_ledger.pool_establish_blocked cam=%s pid=%d — patient "
                "confirmed on another camera; face required",
                self._camera_id, e.pid,
            )
            return
        if fresh_face:
            e.last_face_confirm = e.last_seen
        self._bind_patient(
            e,
            score=(
                float(t.confidence) if fresh_face
                else max(float(t.confidence), 0.72)
            ),
            provenance="face" if fresh_face else "pool",
        )

    def _bind_patient(self, e: _Entry, *, score: float, provenance: str) -> None:
        """Single patient seat — the FP guard. A challenger takes the seat
        only by a higher tier (face > pool) or, at equal tier, a higher
        score. The loser falls to UNKNOWN, never a wrong patient."""
        holder = next(
            (
                x for x in self._entries.values()
                if x.binding == LABEL_PATIENT and x is not e
            ),
            None,
        )
        if holder is not None:
            ct = _TIER.get(provenance, 0)
            ht = _TIER.get(holder.provenance or "", 0)
            # A fresh FACE lock on a track seen THIS tick beats a holder
            # that was NOT seen this tick (a scene-extended phantom): the
            # face-verified patient is physically present; the phantom is
            # stale. Pool challengers can NOT steal a held seat this way —
            # that stays the cross-camera outfit-FP guard.
            holder_stale = holder.last_seen < e.last_seen
            challenger_wins = (
                (provenance == "face" and holder_stale)
                or ct > ht
                or (ct == ht and score > holder.confidence)
            )
            if not challenger_wins:
                log.info(
                    "track_ledger.patient_conflict_kept cam=%s held=%d(%.2f) "
                    "over pid=%d(%.2f)",
                    self._camera_id, holder.pid, holder.confidence, e.pid, score,
                )
                trace.idevent(
                    tid=holder.pid, event="SEAT_KEPT", cause="conflict_incumbent",
                    score=holder.confidence, challenger=f"T{e.pid}",
                )
                return
            log.info(
                "track_ledger.patient_conflict_stolen cam=%s pid=%d(%.2f) takes "
                "PATIENT from pid=%d(%.2f)",
                self._camera_id, e.pid, score, holder.pid, holder.confidence,
            )
            trace.idevent(
                tid=e.pid, event="SEAT_STOLEN", cause=provenance,
                score=score, prev_tid=holder.pid,
            )
            holder.binding = LABEL_UNKNOWN
            holder.provenance = None
            holder.confidence = 0.0
        rebind = e.binding != LABEL_PATIENT
        # Reset confidence to the new score on a (re)bind OR a provenance
        # change: a face re-confirm must not inherit a pool-scale number
        # (face cosines ~0.5, pool ~0.72 — different scales, never mix).
        provenance_changed = (not rebind) and e.provenance != provenance
        e.binding = LABEL_PATIENT
        e.provenance = provenance
        e.confidence = (
            float(score) if (rebind or provenance_changed)
            else max(float(score), e.confidence)
        )
        if rebind:
            e.bound_at = e.last_seen
            self._last_patient_event = f"bound_{provenance}"
            log.info(
                "track_ledger.patient_bound cam=%s pid=%d via=%s score=%.2f",
                self._camera_id, e.pid, provenance, score,
            )
            trace.idevent(tid=e.pid, event="BOUND", cause=provenance, score=score)
        if self._registry is not None:
            held = self._registry.claim(
                self._camera_id,
                tier=TIER_FACE if provenance == "face" else TIER_BODY,
                confidence=e.confidence,
            )
            if not held:
                # A better claim rules elsewhere: single holder per home.
                self._unbind_patient(e, reason="outclaimed", vacate=False)

    def _unbind_patient(self, e: _Entry, *, reason: str, vacate: bool = True) -> None:
        log.info(
            "track_ledger.patient_unbound cam=%s pid=%d reason=%s held_for=%.0fs",
            self._camera_id, e.pid, reason, max(0.0, e.last_seen - e.bound_at),
        )
        trace.idevent(
            tid=e.pid, event="UNBOUND", cause=reason,
            gap_s=max(0.0, e.last_seen - e.bound_at),
        )
        e.binding = LABEL_UNKNOWN
        e.provenance = None
        e.confidence = 0.0
        self._last_patient_event = reason
        if vacate and self._registry is not None:
            self._registry.vacate(self._camera_id)

    def _takeover_check(self, tracks) -> None:
        """Single holder per home: a better claim seated on another camera
        unbinds this camera's PATIENT unless THIS tick carried fresh
        patient evidence (fresh evidence already re-claimed in
        _bind_patient)."""
        if self._registry is None:
            return
        holder_cam = self._registry.holder()
        if holder_cam is None or holder_cam == self._camera_id:
            return
        fresh = any(
            t.label == LABEL_PATIENT
            and (
                t.face_locked
                or (t.pre_labeled and t.pre_label_source == "gallery")
            )
            for t in tracks
        )
        if fresh:
            return
        for e in self._entries.values():
            if e.binding == LABEL_PATIENT:
                self._unbind_patient(e, reason="takeover_elsewhere", vacate=False)

    def _retire(self, now: float, seen: set[int]) -> None:
        # The 45s grace's ONE remaining job (2026-08-14 audit): keeping
        # `is_active` true through a short blind window feeds the
        # degraded-tick state freeze in camera_ingest — the thing that
        # stops a lights-off IR flip from wrongly closing the room as
        # "patient left". The grace's ORIGINAL payoff (the same pod
        # track number reappearing and re-binding instantly) is extinct:
        # EdgeTAM-continuous never reuses a dead number. When the freeze
        # is re-keyed to presence directly, this grace goes too.
        for pid in list(self._entries):
            e = self._entries[pid]
            if pid in seen:
                continue
            if now - e.last_seen > self._retire_after_s:
                if e.binding == LABEL_PATIENT:
                    log.info(
                        "track_ledger.patient_left_view cam=%s pid=%d "
                        "held_for=%.0fs",
                        self._camera_id, pid, max(0.0, e.last_seen - e.bound_at),
                    )
                    trace.idevent(
                        tid=pid, event="UNBOUND", cause="left_view",
                        gap_s=max(0.0, e.last_seen - e.bound_at),
                    )
                    self._last_patient_event = "left_view"
                    if self._registry is not None:
                        self._registry.vacate(self._camera_id)
                del self._entries[pid]

    # ── result rewrite ─────────────────────────────────────────────────

    def _rewrite(self, identity, *, degraded_hold: bool):
        """Project bindings onto the tick's IdentityResult: each track gets
        its entry's label (PATIENT/UNKNOWN); the patient's box feeds
        downstream. Entry key == pod track_id, so track ids pass through
        unchanged."""
        patient_entry = next(
            (e for e in self._entries.values() if e.binding == LABEL_PATIENT),
            None,
        )
        new_tracks = []
        patient_track = None
        for t in identity.tracks or ():
            e = self._entries.get(int(t.track_id))
            if e is None:
                new_tracks.append(t)
                continue
            held_relabel = e.binding != t.label
            nt = dataclasses.replace(
                t,
                label=e.binding,
                confidence=(
                    e.confidence if e.binding == LABEL_PATIENT else t.confidence
                ),
                pre_labeled=t.pre_labeled or held_relabel,
                pre_label_source="held" if held_relabel else t.pre_label_source,
            )
            new_tracks.append(nt)
            if patient_entry is not None and e.pid == patient_entry.pid:
                patient_track = nt

        patient_in = patient_track is not None
        patient_bbox = identity.patient_bbox
        per_frame = identity.per_frame_bboxes
        confidence = identity.confidence
        if patient_track is not None and patient_track.bbox_series:
            patient_bbox = patient_track.bbox_series[-1][1]
            if per_frame is not None:
                filled = list(per_frame)
                for fi, bb in patient_track.bbox_series:
                    if 0 <= fi < len(filled):
                        filled[fi] = bb
                per_frame = filled
            confidence = max(confidence, patient_entry.confidence)
        elif patient_entry is not None and degraded_hold:
            # Patient bound but invisible on a degraded/stale tick:
            # presence persists for the state machine, no box claims.
            patient_in = True
            confidence = max(confidence, patient_entry.confidence)

        return dataclasses.replace(
            identity,
            tracks=tuple(new_tracks),
            patient_in_clip=patient_in,
            patient_bbox=patient_bbox if patient_in else identity.patient_bbox,
            per_frame_bboxes=per_frame,
            confidence=confidence,
            # No OTHER in this model — the cross-camera suppression gate no
            # longer keys off "only verified others" (see camera_ingest).
            has_only_verified_other_tracks=False,
        )

    # ── observability ───────────────────────────────────────────────────

    def stash_fields(self) -> dict:
        patient = next(
            (e for e in self._entries.values() if e.binding == LABEL_PATIENT),
            None,
        )
        out: dict = {
            "id_binding": "patient" if patient is not None else "none",
            "id_tracks": {
                str(e.pid): e.binding for e in self._entries.values()
            },
        }
        if patient is not None:
            out["id_binding_source"] = patient.provenance
            out["id_binding_age_s"] = round(
                max(0.0, patient.last_seen - patient.bound_at),
            )
            out["id_binding_seen"] = self._patient_seen_last_tick
            out["id_last_seen_s"] = round(
                max(0.0, self._last_apply_now - patient.last_seen),
            )
        elif self._last_patient_event:
            out["id_binding_last_event"] = self._last_patient_event
        return out
