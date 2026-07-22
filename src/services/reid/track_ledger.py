"""Track-anchored identity ledger — the binding brain of the
track-anchored architecture (2026-07-21, replaces IdentityLock when
`settings.reid_binding_mode == "track"`).

Product requirement (Aryaman): "confirm the patient and stay locked to
that patient no matter where they go or what they do" — and
symmetrically for OTHER and UNKNOWN: every body keeps a persistent
identity for as long as it stays in view.

Layered model: SAM gives flawless WITHIN-clip tracks; this ledger
stitches them ACROSS clips geometrically (consecutive tick windows
overlap, so the boundary gap is ~one sample interval); the pool and
face machinery are demoted from per-tick prosecutors to notaries —
they speak at track BIRTH (confirm), on a bound track only as AUDIT
(strong contradiction = tracker-swap suspicion), and at re-entry
after a track dies.

What this deletes relative to IdentityLock: park-on-crowd (new people
are new tracks; nothing re-litigates an existing binding), per-tick
re-proof, release hysteresis as a separate rule (its intent survives
as the audit contra streak), and the carry ghost buffer (binding mode
runs carry-less; geometry + birth-time pool cover it).

Same seam as IdentityLock: constructed per camera by CameraIngest,
`apply(identity, tick_mono)` folds bindings into the tick's
IdentityResult, `stash_fields()` feeds the analyses row. Pure
in-memory, single-writer. Emergency detectors never depend on this.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Optional
from uuid import UUID

from src.services.reid.body_pipeline import (
    LABEL_OTHER,
    LABEL_PATIENT,
    LABEL_UNKNOWN,
)
from src.services.detection.patient_presence import (
    TIER_BODY,
    TIER_FACE,
    TIER_HELD,
)

if TYPE_CHECKING:
    from src.services.detection.patient_presence import PatientPresence
    from src.services.reid.inference import IdentityResult
    from src.settings import Settings

log = logging.getLogger(__name__)

# Persistent ids start high so they can never collide with within-clip
# SAM/BoT-SORT track ids (small ints) in downstream consumers' eyes.
_PID_BASE = 1000

# Sustained small-face bind bar: independent above-lock-bar match
# events required before a sub-quality face may bind PATIENT. 3x the
# instant big-face hits_to_lock (2) — measured couch data 2026-07-21:
# the patient produces these on nearly every face-active tick from
# 20ft; his uncle produced ZERO in ~26 attempts from the same seat.
_SMALLFACE_BIND_EVENTS = 6

# Harvest afterglow: a bound track may donate body crops to the pool
# only this soon after face-grade evidence (Aryaman's corruption
# guard: pool-provenance bindings never harvest — no pool→pool
# self-feeding, ever).
_HARVEST_AFTERGLOW_S = 60.0


@dataclasses.dataclass
class _Entry:
    pid: int
    binding: str                      # LABEL_PATIENT / OTHER / UNKNOWN
    provenance: str | None            # "face" | "pool" | None
    confidence: float
    bound_at: float
    first_seen: float
    last_seen: float
    last_bbox: tuple[float, float, float, float]
    # px/s in extracted-frame space, from within-clip motion (EMA).
    vel: tuple[float, float] = (0.0, 0.0)
    contra_streak: int = 0
    seen_ticks: int = 1
    # True while THIS tick's clip actively contradicts a PATIENT
    # binding (verified OTHER) — the binding survives one such tick
    # but must not ride it (F1: a verified guest never wears the
    # patient label).
    suppressed: bool = False
    # Two-way evidence accumulators (2026-07-21, both born from live
    # findings): verified face mismatches accumulate toward OTHER
    # across ticks (the uncle bug — one clip can no longer reach the
    # in-clip bar at 2fps budgets); sustained above-lock-bar small-
    # face matches accumulate toward PATIENT (the couch-at-20ft
    # finding — identity information present, pixels insufficient).
    # Contradicting evidence resets the opposing accumulator.
    mismatch_total: int = 0
    smallface_total: int = 0
    smallface_score: float = 0.0
    # Monotonic ts of the last face-grade confirmation (bind or
    # supporting face evidence) — the harvest afterglow anchor.
    last_face_confirm: float = 0.0


def _center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1.0, area_a + area_b - inter)


def _shift(b, dx, dy):
    return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _containment(inner, outer) -> float:
    """area(inner ∩ outer) / area(inner)."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    return (iw * ih) / max(1.0, _bbox_area(inner))


class TrackLedger:
    """Per-camera persistent-track ledger + identity bindings."""

    def __init__(
        self,
        *,
        camera_id: UUID,
        registry: "PatientPresence | None",
        settings: "Settings",
    ) -> None:
        self._camera_id = camera_id
        self._registry = registry
        self._settings = settings
        self._entries: dict[int, _Entry] = {}
        self._next_pid = _PID_BASE
        self._last_tick_mono = 0.0
        self._scene_present = False
        self._last_patient_event: str | None = None
        # Face-throttle state (optimization #1): False until a tick
        # ends with every seen track bound and streak-free.
        self._all_bound_last_tick = False
        self._face_heartbeat = 0
        # Truth fields for the dashboard (Aryaman: "I want to measure
        # the true latency"): was the patient actually SEEN this tick,
        # and the apply-time anchor for last-seen age.
        self._patient_seen_last_tick = False
        self._last_apply_now = 0.0

        g = lambda n, d: float(getattr(settings, n, d))  # noqa: E731
        # REVERTED to 45s (2026-07-22 00:55): the 30s "unification"
        # regressed live — under load-shedding and normal host-a-guest
        # in-and-out movement, 30s retires bindings the 45s grace
        # bridged, reading as spotty patient tracking. Display
        # staleness is solved by id_binding_seen/id_last_seen_s, not
        # by shortening the memory.
        self._retire_after_s = g("reid_binding_retire_after_s", 45.0)
        self._iou_floor = g("reid_binding_iou_floor", 0.20)
        # Distance gate grows with the boundary gap: base + speed·dt.
        self._center_base_px = g("reid_binding_center_base_px", 80.0)
        self._center_px_per_s = g("reid_binding_center_px_per_s", 150.0)
        self._ambig_margin = g("reid_binding_ambig_margin", 0.08)
        self._audit_ticks = int(g("reid_binding_audit_contra_ticks", 2))
        self._min_birth_frames = int(g("reid_binding_min_birth_frames", 3))
        self._max_gap_s = g("reid_binding_max_stitch_gap_s", 30.0)

    # ── external surface (IdentityLock-compatible) ─────────────────

    def note_scene_presence(self, present: bool) -> None:
        self._scene_present = bool(present)

    def all_bound(self) -> bool:
        """True when last tick ended with a patient present and EVERY
        seen track identified (nobody unbound/suspect). Drives lean
        sampling (headroom fix): only run the camera lean when the
        patient is locked and there's nobody left to identify."""
        return self._all_bound_last_tick and any(
            e.binding == LABEL_PATIENT for e in self._entries.values()
        )

    def wants_face_evidence(self) -> bool:
        """Binding-aware face throttle (optimization #1, 2026-07-21).
        Face evidence (SCRFD + ArcFace + the SAM countersign ≈ a third
        of an occupied tick's GPU cost) is REQUIRED while any body in
        view is unbound or under audit suspicion — and drops to a
        1-in-3 heartbeat once everyone is bound (the passive-audit
        cadence). A new track birthing flips this back to full cadence
        on the very next tick, so guest branding is never starved.
        Called by CameraIngest BEFORE each tick; reads the previous
        tick's outcome."""
        if not self._all_bound_last_tick:
            self._face_heartbeat = 0
            return True
        self._face_heartbeat += 1
        if self._face_heartbeat >= 3:
            self._face_heartbeat = 0
            return True
        return False

    @property
    def is_active(self) -> bool:
        """True while a PATIENT binding is live (W2 freeze gate)."""
        return any(
            e.binding == LABEL_PATIENT and not e.suppressed
            for e in self._entries.values()
        )

    @property
    def patient_seen(self) -> bool:
        """Was the patient actually SEEN on the last processed tick?
        Feeds the home registry's holder_seen heartbeat — the governor
        escalation trigger for other rooms."""
        return self._patient_seen_last_tick

    def harvest_hint(self) -> tuple | None:
        """((x0,y0,x1,y1), confidence) for the patient's last box when
        the binding is in FACE-AFTERGLOW: face-provenance, patient
        actually seen last tick, and face-grade evidence within the
        envelope. Pool-provenance bindings never emit a hint — the
        pool must not feed itself (Aryaman's corruption guard)."""
        e = next(
            (
                x for x in self._entries.values()
                if x.binding == LABEL_PATIENT and not x.suppressed
            ),
            None,
        )
        if (
            e is None
            or e.provenance != "face"
            or not self._patient_seen_last_tick
            or self._last_apply_now - e.last_face_confirm
            > _HARVEST_AFTERGLOW_S
        ):
            return None
        return (e.last_bbox, float(e.confidence))

    def apply(
        self,
        identity: "IdentityResult",
        *,
        tick_mono: float | None = None,
    ) -> "IdentityResult":
        """Stitch this tick's tracks onto the ledger, fold evidence
        into bindings, and rewrite the IdentityResult through the
        bindings. Never raises — a ledger bug must not break a tick."""
        try:
            return self._apply_inner(identity, tick_mono)
        except Exception:  # noqa: BLE001
            log.exception("track_ledger.apply failed cam=%s", self._camera_id)
            return identity

    # ── core ───────────────────────────────────────────────────────

    def _apply_inner(
        self, identity: "IdentityResult", tick_mono: float | None,
    ) -> "IdentityResult":
        if not identity.reid_ran:
            # Infra failure ≠ evidence: freeze everything.
            return identity
        now = tick_mono if tick_mono is not None else time.monotonic()
        if tick_mono is not None:
            if tick_mono < self._last_tick_mono:
                # Stale window: read-only fold, never mutate state.
                return self._rewrite(identity, {}, degraded_hold=True)
            self._last_tick_mono = tick_mono

        for e in self._entries.values():
            e.suppressed = False

        tracks = tuple(identity.tracks or ())
        degraded = bool(identity.quality_degraded)

        assign = self._stitch(tracks, now)
        by_id = {t.track_id: t for t in tracks}

        # Geometry refresh for matched tracks (evidence folds AFTER
        # the fragment merge below, so it lands on merged entries).
        for tid, entry in assign.items():
            self._refresh_geometry(entry, by_id[tid], identity, now)

        # Births — substantial unmatched tracks only (birth fragments
        # churn the ledger; a real person will still be here next tick).
        for t in tracks:
            if t.track_id in assign:
                continue
            if len(t.bbox_series) < self._min_birth_frames:
                continue
            assign[t.track_id] = self._birth(t, now)

        # FRAGMENT MERGE (2026-07-22 keystone fix): SAM regularly
        # splits one body into a small extra track (face region, arm)
        # inside the real body track. As two ledger entries they FIGHT
        # over the identity — the observed 1000→1001→1002 label churn
        # that kept the lock unstable and lean-mode disengaged. Merging
        # BEFORE the evidence fold means a face lock lands on the one
        # merged entry and the fight never starts.
        self._merge_fragments(assign)

        # Evidence fold — when several clip tracks map to one entry
        # (merged fragments), fold the dominant (largest-box) track
        # first so body-level evidence leads.
        by_pid: dict[int, list] = {}
        for tid, entry in assign.items():
            by_pid.setdefault(entry.pid, []).append(by_id[tid])
        for pid, tlist in by_pid.items():
            entry = self._entries.get(pid)
            if entry is None:
                continue
            tlist.sort(
                key=lambda t: (
                    _bbox_area(t.bbox_series[-1][1]) if t.bbox_series else 0.0
                ),
                reverse=True,
            )
            for t in tlist:
                self._fold_evidence(entry, t, degraded)

        self._takeover_check(assign, tracks)

        if not degraded:
            self._retire(now, seen={e.pid for e in assign.values()})

        # Face-throttle input: full evidence cadence resumes whenever
        # any body in view is unbound or under audit suspicion — and
        # on EMPTY rooms (costs nothing without tracks, and keeps the
        # cold-entry bind instant when someone walks in).
        seen_entries = list(assign.values())
        self._all_bound_last_tick = bool(seen_entries) and all(
            e.binding != LABEL_UNKNOWN and e.contra_streak == 0
            for e in seen_entries
        )

        # Truth fields: seen-this-tick + apply-time anchor.
        self._last_apply_now = now
        seen_patient = next(
            (
                e for e in seen_entries
                if e.binding == LABEL_PATIENT and not e.suppressed
            ),
            None,
        )
        self._patient_seen_last_tick = seen_patient is not None

        # Keep the home-registry seat warm while the patient is
        # actually in view (parity with the old lock's TIER_HELD
        # refresh): without this the claim TTLs out mid-sit and
        # patient_elsewhere suppression silently stops. Runs AFTER
        # the takeover check, so an outclaimed camera never re-warms
        # a seat it just lost.
        if self._registry is not None and seen_patient is not None:
            self._registry.claim(
                self._camera_id, tier=TIER_HELD,
                confidence=seen_patient.confidence,
            )

        # takeover_check may have consulted stale suppression; the
        # rewrite below reads entry state as of now.
        return self._rewrite(identity, assign, degraded_hold=degraded)

    # ── stitcher ───────────────────────────────────────────────────

    def _stitch(self, tracks, now: float) -> dict[int, _Entry]:
        """Greedy one-to-one geometric match: clip tracks' FIRST bbox
        vs ledger entries' velocity-extrapolated LAST bbox. Returns
        {clip track_id -> entry}. Ambiguity (two candidates within the
        margin) inherits nothing — the conservative failure is a new
        UNKNOWN id, which the pool/face machinery re-binds; the
        unacceptable failure is a wrong inherit."""
        live = [
            e for e in self._entries.values()
            if now - e.last_seen <= self._max_gap_s
        ]
        cands: list[tuple[float, object, _Entry]] = []
        for t in tracks:
            if not t.bbox_series:
                continue
            first = t.bbox_series[0][1]
            fc = _center(first)
            for e in live:
                dt = max(0.0, now - e.last_seen)
                pred = _shift(
                    e.last_bbox,
                    max(-200.0, min(200.0, e.vel[0] * dt)),
                    max(-200.0, min(200.0, e.vel[1] * dt)),
                )
                pc = _center(pred)
                dist = ((fc[0] - pc[0]) ** 2 + (fc[1] - pc[1]) ** 2) ** 0.5
                gate = self._center_base_px + self._center_px_per_s * dt
                iou = _iou(first, pred)
                if iou < self._iou_floor and dist > gate:
                    continue
                score = iou + max(0.0, 1.0 - dist / max(1.0, gate)) * 0.2
                cands.append((score, t, e))
        cands.sort(key=lambda c: c[0], reverse=True)

        best_by_track: dict[int, float] = {}
        second_by_track: dict[int, float] = {}
        for score, t, _e in cands:
            tid = t.track_id
            if tid not in best_by_track:
                best_by_track[tid] = score
            elif tid not in second_by_track:
                second_by_track[tid] = score

        assign: dict[int, _Entry] = {}
        used: set[int] = set()
        for score, t, e in cands:
            if t.track_id in assign or e.pid in used:
                continue
            second = second_by_track.get(t.track_id, 0.0)
            if score - second < self._ambig_margin and score < 0.5:
                # Boundary-instant crossing between comparable
                # candidates: refuse to guess.
                log.info(
                    "track_ledger.ambiguous_stitch cam=%s tid=%d "
                    "best=%.2f second=%.2f — new id",
                    self._camera_id, t.track_id, score, second,
                )
                continue
            assign[t.track_id] = e
            used.add(e.pid)
        return assign

    def _refresh_geometry(self, e: _Entry, t, identity, now: float) -> None:
        dt = max(1e-3, now - e.last_seen)
        last = t.bbox_series[-1][1]
        # Within-clip velocity (px/s) — sample_fps gives the series'
        # time base; fall back to boundary displacement when absent.
        fps = identity.sample_fps or 0
        if fps and len(t.bbox_series) >= 2:
            (f0, b0), (f1, b1) = t.bbox_series[0], t.bbox_series[-1]
            span = max(1e-3, (f1 - f0) / float(fps))
            c0, c1 = _center(b0), _center(b1)
            v = ((c1[0] - c0[0]) / span, (c1[1] - c0[1]) / span)
        else:
            c_prev, c_new = _center(e.last_bbox), _center(last)
            v = ((c_new[0] - c_prev[0]) / dt, (c_new[1] - c_prev[1]) / dt)
        e.vel = (0.5 * e.vel[0] + 0.5 * v[0], 0.5 * e.vel[1] + 0.5 * v[1])
        e.last_bbox = last
        e.last_seen = now
        e.seen_ticks += 1

    # ── bindings ───────────────────────────────────────────────────

    def _merge_fragments(self, assign: dict[int, "_Entry"]) -> None:
        """Entries seen THIS tick whose boxes near-coincide with a
        strong size skew are one physical body: containment(small in
        big) ≥ 0.7 AND area(small)/area(big) ≤ 0.5. The size-ratio
        guard is the safety property — two similar-sized adults (a
        hug, a close pass) can NEVER merge, so a guest can never be
        fused into the patient; only genuine fragments qualify.
        Survivor: PATIENT-bound > OTHER-bound > older. The duplicate's
        clip tracks repoint to the survivor and the duplicate dies."""
        changed = True
        while changed:
            changed = False
            uniq: dict[int, _Entry] = {}
            for e in assign.values():
                uniq[e.pid] = e
            ents = list(uniq.values())
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    a, b = ents[i], ents[j]
                    small, big = sorted(
                        (a, b), key=lambda x: _bbox_area(x.last_bbox),
                    )
                    ratio = _bbox_area(small.last_bbox) / max(
                        1.0, _bbox_area(big.last_bbox),
                    )
                    if ratio > 0.5:
                        continue
                    if _containment(small.last_bbox, big.last_bbox) < 0.7:
                        continue
                    keep, drop = self._merge_priority(a, b)
                    self._absorb(keep, drop)
                    for tid, e in list(assign.items()):
                        if e.pid == drop.pid:
                            assign[tid] = keep
                    self._entries.pop(drop.pid, None)
                    log.info(
                        "track_ledger.fragment_merged cam=%s kept=%d "
                        "absorbed=%d ratio=%.2f",
                        self._camera_id, keep.pid, drop.pid, ratio,
                    )
                    changed = True
                    break
                if changed:
                    break

    @staticmethod
    def _merge_priority(a: "_Entry", b: "_Entry"):
        for label in (LABEL_PATIENT, LABEL_OTHER):
            if a.binding == label and b.binding != label:
                return a, b
            if b.binding == label and a.binding != label:
                return b, a
        return (a, b) if a.first_seen <= b.first_seen else (b, a)

    @staticmethod
    def _absorb(keep: "_Entry", drop: "_Entry") -> None:
        if drop.binding == LABEL_PATIENT and keep.binding != LABEL_PATIENT:
            keep.binding = drop.binding
            keep.provenance = drop.provenance
            keep.confidence = drop.confidence
            keep.bound_at = drop.bound_at
        keep.first_seen = min(keep.first_seen, drop.first_seen)
        keep.last_face_confirm = max(
            keep.last_face_confirm, drop.last_face_confirm,
        )
        keep.smallface_total = max(keep.smallface_total, drop.smallface_total)
        keep.smallface_score = max(keep.smallface_score, drop.smallface_score)
        # Mismatch/contra state deliberately NOT imported: a fragment's
        # face evidence is unreliable and must not poison the survivor.

    def _birth(self, t, now: float) -> _Entry:
        e = _Entry(
            pid=self._next_pid,
            binding=LABEL_UNKNOWN,
            provenance=None,
            confidence=0.0,
            bound_at=now,
            first_seen=now,
            last_seen=now,
            last_bbox=t.bbox_series[-1][1],
        )
        self._next_pid += 1
        self._entries[e.pid] = e
        return e

    def _fold_evidence(self, e: _Entry, t, degraded: bool) -> None:
        """Fold this clip's per-track verdict (produced by the
        within-clip brain: face locks, verified mismatches, pool
        pre-labels) into the entry's binding."""
        fresh_face = t.label == LABEL_PATIENT and t.face_locked
        fresh_pool = (
            t.label == LABEL_PATIENT
            and t.pre_labeled
            and t.pre_label_source == "gallery"
        )
        verified_other = t.label == LABEL_OTHER

        if fresh_face or fresh_pool:
            # PRECISION GUARD (2026-07-22, live false-confirm): pool
            # (outfit) may MAINTAIN an existing patient binding but must
            # NOT ESTABLISH a new one on this camera while the patient
            # is confirmed on ANOTHER camera. A relative in a similar
            # outfit pool-matched the patient gallery and got confirmed
            # as the patient while the real patient was locked in the
            # other room (single-holder violated, both cameras PATIENT).
            # Only a FACE can cross the single-holder boundary — outfit
            # is too weak to move the patient to a new camera. Cold
            # start (no holder) and same-camera maintenance are
            # unaffected: the guard fires only when someone ELSE holds.
            establishing = e.binding != LABEL_PATIENT
            if (
                fresh_pool
                and not fresh_face
                and establishing
                and self._registry is not None
                and self._registry.confirmed_elsewhere(self._camera_id)
            ):
                log.info(
                    "track_ledger.pool_establish_blocked cam=%s pid=%d "
                    "— patient confirmed on another camera; outfit alone "
                    "cannot establish the patient here (face required)",
                    self._camera_id, e.pid,
                )
                return
            e.contra_streak = 0
            e.mismatch_total = 0
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
            return

        if verified_other:
            face_grade = bool(t.face_locked)
            if e.binding == LABEL_PATIENT:
                # Audit path: the tracker says this is the patient's
                # body; the face says verified-not-patient. Either the
                # tracker swapped or the evidence is poisoned. One tick
                # never destroys a binding (hysteresis, B3 lesson);
                # sustained FACE-verified contradiction unbinds to
                # UNKNOWN — never straight to OTHER — and lets
                # confirmation re-run. Pool-OTHER pre-labels (outfit
                # echo, the B3 poison class) are NOT audit-grade: they
                # suppress this tick's relabel (F1) but build no
                # streak. Degraded ticks are inadmissible entirely.
                e.suppressed = True
                if degraded or not face_grade:
                    return
                e.contra_streak += 1
                if e.contra_streak >= self._audit_ticks:
                    log.warning(
                        "track_ledger.swap_suspected cam=%s pid=%d — "
                        "sustained verified-OTHER on PATIENT binding; "
                        "unbinding to UNKNOWN",
                        self._camera_id, e.pid,
                    )
                    self._unbind_patient(e, reason="audit_contradiction")
                return
            if e.binding != LABEL_OTHER:
                e.binding = LABEL_OTHER
                e.provenance = "face" if face_grade else "pool"
                e.confidence = 0.0
                e.bound_at = e.last_seen
            e.contra_streak = 0
            return

        # Neutral clip (UNKNOWN label): binding persists; on UNBOUND
        # entries the two-way accumulators fold sub-verdict evidence.
        e.contra_streak = 0
        if e.binding != LABEL_UNKNOWN:
            return
        if t.verified_mismatches > 0:
            # Pure not-patient face evidence (zero hits this clip):
            # kills any small-face run and accumulates toward OTHER.
            e.smallface_total = 0
            e.smallface_score = 0.0
            if degraded:
                return
            e.mismatch_total += int(t.verified_mismatches)
            other_bar = int(getattr(
                self._settings, "reid_face_checks_for_other", 5,
            ))
            if e.mismatch_total >= other_bar:
                e.binding = LABEL_OTHER
                e.provenance = "face"
                e.confidence = 0.0
                e.bound_at = e.last_seen
                log.info(
                    "track_ledger.other_bound_accumulated cam=%s "
                    "pid=%d mismatches=%d over %d ticks",
                    self._camera_id, e.pid, e.mismatch_total,
                    e.seen_ticks,
                )
            return
        if t.smallface_events > 0:
            # Above-lock-bar matches refused only by pixels: sustained
            # runs bind PATIENT (distance identification).
            e.mismatch_total = 0
            if degraded:
                return
            e.smallface_total += int(t.smallface_events)
            e.smallface_score = max(
                e.smallface_score, float(t.best_match_score),
            )
            if e.smallface_total >= _SMALLFACE_BIND_EVENTS:
                log.info(
                    "track_ledger.smallface_bound cam=%s pid=%d "
                    "events=%d score=%.3f",
                    self._camera_id, e.pid, e.smallface_total,
                    e.smallface_score,
                )
                e.last_face_confirm = e.last_seen
                self._bind_patient(
                    e, score=e.smallface_score, provenance="face",
                )

    def _bind_patient(self, e: _Entry, *, score: float, provenance: str) -> None:
        holder = next(
            (
                x for x in self._entries.values()
                if x.binding == LABEL_PATIENT and x is not e
            ),
            None,
        )
        if holder is not None:
            # Single-patient invariant (one patient per home, a
            # fortiori per camera). Conflicts compare by EVIDENCE TIER
            # first — face categorically outranks pool (Aryaman's
            # day-one hierarchy; face cosines ~0.45-0.6 and pool
            # cosines ~0.72+ are different scales and must never be
            # compared numerically across tiers — 2026-07-22 live
            # finding: a 0.72 pool bind stole from a 0.54 face bind).
            # Equal tiers compare by score. Loser → UNKNOWN, never
            # OTHER.
            tier = {"face": 2, "pool": 1}
            ct = tier.get(provenance, 0)
            ht = tier.get(holder.provenance or "", 0)
            challenger_wins = ct > ht or (ct == ht and score > holder.confidence)
            if not challenger_wins:
                log.info(
                    "track_ledger.patient_conflict_kept cam=%s "
                    "held=%d(%.2f) over pid=%d(%.2f)",
                    self._camera_id, holder.pid, holder.confidence,
                    e.pid, score,
                )
                return
            log.info(
                "track_ledger.patient_conflict_stolen cam=%s "
                "pid=%d(%.2f) takes PATIENT from pid=%d(%.2f)",
                self._camera_id, e.pid, score, holder.pid,
                holder.confidence,
            )
            holder.binding = LABEL_UNKNOWN
            holder.provenance = None
            holder.confidence = 0.0
        rebind = e.binding != LABEL_PATIENT
        e.binding = LABEL_PATIENT
        e.provenance = provenance
        e.confidence = max(float(score), e.confidence if not rebind else 0.0)
        if rebind:
            e.bound_at = e.last_seen
            self._last_patient_event = f"bound_{provenance}"
            log.info(
                "track_ledger.patient_bound cam=%s pid=%d via=%s "
                "score=%.2f",
                self._camera_id, e.pid, provenance, score,
            )
        if self._registry is not None:
            held = self._registry.claim(
                self._camera_id,
                tier=TIER_FACE if provenance == "face" else TIER_BODY,
                confidence=e.confidence,
            )
            if not held:
                # A better claim rules elsewhere: tick-local truth
                # stands, the binding does not (single holder / home).
                self._unbind_patient(e, reason="outclaimed", vacate=False)

    def _unbind_patient(
        self, e: _Entry, *, reason: str, vacate: bool = True,
    ) -> None:
        log.info(
            "track_ledger.patient_unbound cam=%s pid=%d reason=%s "
            "held_for=%.0fs",
            self._camera_id, e.pid, reason,
            max(0.0, e.last_seen - e.bound_at),
        )
        e.binding = LABEL_UNKNOWN
        e.provenance = None
        e.confidence = 0.0
        e.contra_streak = 0
        self._last_patient_event = reason
        if vacate and self._registry is not None:
            self._registry.vacate(self._camera_id)

    def _takeover_check(self, assign: dict, tracks) -> None:
        """Single holder per home: a better claim seated on another
        camera unbinds this camera's PATIENT unless THIS tick carried
        fresh evidence (fresh evidence already re-claimed in
        _bind_patient)."""
        if self._registry is None:
            return
        holder_cam = self._registry.holder()
        if holder_cam is None or holder_cam == self._camera_id:
            return
        fresh = any(
            (t.label == LABEL_PATIENT and (t.face_locked or (
                t.pre_labeled and t.pre_label_source == "gallery"
            )))
            for t in tracks
        )
        if fresh:
            return
        for e in self._entries.values():
            if e.binding == LABEL_PATIENT:
                self._unbind_patient(
                    e, reason="takeover_elsewhere", vacate=False,
                )

    def _retire(self, now: float, seen: set[int]) -> None:
        for pid in list(self._entries):
            e = self._entries[pid]
            if pid in seen:
                continue
            grace = self._retire_after_s
            if e.binding == LABEL_PATIENT and self._scene_present:
                # Scene layer still sees a person the tracker can't
                # (lying in bed) — presence extends, mirror of the
                # lock's scene extension.
                e.last_seen = now
                continue
            if now - e.last_seen > grace:
                if e.binding == LABEL_PATIENT:
                    log.info(
                        "track_ledger.patient_left_view cam=%s pid=%d "
                        "held_for=%.0fs",
                        self._camera_id, pid,
                        max(0.0, e.last_seen - e.bound_at),
                    )
                    self._last_patient_event = "left_view"
                    if self._registry is not None:
                        self._registry.vacate(self._camera_id)
                del self._entries[pid]

    # ── result rewrite ─────────────────────────────────────────────

    def _rewrite(
        self, identity: "IdentityResult", assign: dict, *,
        degraded_hold: bool,
    ) -> "IdentityResult":
        """Project the bindings back onto the tick's IdentityResult:
        every consumer (signals, crops, evidence frames, gates) sees
        persistent ids and binding labels. Clip-verified OTHER tracks
        are never overridden (F1)."""
        patient_entry = next(
            (
                e for e in self._entries.values()
                if e.binding == LABEL_PATIENT and not e.suppressed
            ),
            None,
        )

        new_tracks = []
        patient_track = None
        for t in identity.tracks or ():
            e = assign.get(t.track_id)
            if e is None:
                new_tracks.append(t)
                continue
            label = e.binding
            if t.label == LABEL_OTHER and label == LABEL_PATIENT:
                # Should be unreachable (fold suppresses), belt anyway.
                label = LABEL_OTHER
            held_relabel = label != t.label
            nt = dataclasses.replace(
                t,
                track_id=e.pid,
                label=label,
                confidence=(
                    e.confidence if label == LABEL_PATIENT else t.confidence
                ),
                pre_labeled=t.pre_labeled or held_relabel,
                pre_label_source=(
                    "held" if held_relabel else t.pre_label_source
                ),
            )
            new_tracks.append(nt)
            if patient_entry is not None and e.pid == patient_entry.pid:
                # Merged fragments: several clip tracks may map to the
                # patient entry — the LARGEST box (the body, not the
                # face fragment) carries the patient bbox downstream.
                if patient_track is None or (
                    nt.bbox_series and patient_track.bbox_series
                    and _bbox_area(nt.bbox_series[-1][1])
                    > _bbox_area(patient_track.bbox_series[-1][1])
                ):
                    patient_track = nt
        patient_in = patient_track is not None
        patient_bbox = identity.patient_bbox
        per_frame = identity.per_frame_bboxes
        confidence = identity.confidence
        if patient_track is not None:
            if patient_track.bbox_series:
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

        seen_bindings = [
            assign[t.track_id].binding for t in (identity.tracks or ())
            if t.track_id in assign
        ]
        only_others = (
            len(seen_bindings) > 0
            and all(b == LABEL_OTHER for b in seen_bindings)
            and patient_entry is None
        )
        return dataclasses.replace(
            identity,
            tracks=tuple(new_tracks),
            patient_in_clip=patient_in,
            patient_bbox=patient_bbox if patient_in else identity.patient_bbox,
            per_frame_bboxes=per_frame,
            confidence=confidence,
            has_only_verified_other_tracks=only_others,
        )

    # ── observability ──────────────────────────────────────────────

    def stash_fields(self) -> dict:
        patient = next(
            (
                e for e in self._entries.values()
                if e.binding == LABEL_PATIENT
            ),
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
            # Truth split (measure the real latency): seen NOW vs
            # riding a memory, and how stale that memory is.
            out["id_binding_seen"] = self._patient_seen_last_tick
            out["id_last_seen_s"] = round(
                max(0.0, self._last_apply_now - patient.last_seen),
            )
        elif self._last_patient_event:
            out["id_binding_last_event"] = self._last_patient_event
        return out
