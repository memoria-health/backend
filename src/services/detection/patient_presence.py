"""Home-wide patient identity registry — single holder + cross-camera
suppression (identity-persistence W3, 2026-07-19; grew out of the
2026-07-18 "if patient is confirmed in one camera, others do not fire"
cost gate).

One instance per RoomWorker (= per patient = per home); every
CameraIngest in the home shares it. Two jobs:

1. SINGLE HOLDER (Aryaman: "one PATIENT per home" — the invariant is
   home-level, facilities later extend to N patients = N registries).
   Cameras submit CLAIMS carrying an evidence tier + confidence; the
   best fresh claim holds the home-wide patient lock. Greedy takeover:
   a strictly better claim (higher tier, then higher confidence) moves
   the lock; the losing camera's IdentityLock observes `holder()` and
   releases to UNKNOWN — never to OTHER.

   Tiers: 2 = quality face lock this tick (the strongest evidence),
   1 = body-gallery match this tick, 0 = held (an active lock coasting
   on continuity without fresh evidence this tick). Held claims can
   REFRESH their own camera's hold but can never steal the lock.

2. CROSS-CAMERA DETECTOR SUPPRESSION (unchanged semantics): while a
   fresh holder exists, OTHER cameras' patient-scoped detectors below
   the emergency severity floor skip with `patient_elsewhere`. The
   scheduler's tick-local sighting override and the severity≥80
   exemption sit on top exactly as before.

Fail-open is the law: staleness, ambiguity, or an empty registry mean
NO suppression and NO takeover — absence of certainty never mutes a
safety detector and never releases a lock by itself.

Single-writer-per-camera within one asyncio loop — no lock needed
(same contract as MotionCache / FrameBuffer).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID

# How long one claim stays fresh. Long enough to cover consecutive
# ticks between confirmations of a seated patient; short enough that
# walking to another room re-enables its detectors within seconds of
# the track/face being lost.
CONFIRM_TTL_S = 45.0

# Evidence tiers (see module docstring).
TIER_FACE = 2
TIER_BODY = 1
TIER_HELD = 0


@dataclass
class _Claim:
    camera_id: UUID
    tier: int
    confidence: float
    at: float


class PatientPresence:
    def __init__(self, *, ttl_s: float = CONFIRM_TTL_S) -> None:
        self._ttl = ttl_s
        self._claim: _Claim | None = None
        # Patient-focus governor state (2026-07-22): per-camera
        # occupancy heartbeats + whether the holder camera actually
        # SAW the patient on its last tick. Drives the per-tick reid
        # mode (lean / minimal / split) so total GPU demand always
        # fits — the fix for the both-cameras-full saturation spiral.
        self._occupancy: dict[UUID, tuple[bool, float]] = {}
        self._holder_seen: tuple[bool, float] | None = None

    # ── writing ────────────────────────────────────────────────

    def claim(
        self,
        camera_id: UUID,
        *,
        tier: int,
        confidence: float,
    ) -> bool:
        """Submit a claim. Returns True iff `camera_id` now holds the
        home-wide lock.

        Rules: the current holder always refreshes itself; a vacant or
        stale seat is taken by any FRESH-evidence claim (tier ≥ 1); a
        held (tier 0) claim never seats a NEW holder; a challenger
        takes the seat only with a strictly better (tier, confidence)
        than the incumbent's claim.
        """
        now = time.monotonic()
        cur = self._claim
        # Staleness first (review: a stale seat is VACANT for everyone
        # — including its previous owner's held refreshes; without
        # this, tier-0 coasting resurrects an expired claim).
        if cur is not None and (now - cur.at) > self._ttl:
            self._claim = cur = None
        if cur is not None and cur.camera_id == camera_id:
            if tier <= TIER_HELD:
                # Held refresh: keep the seat and lower its defense to
                # (0, conf) — but DO NOT renew the timestamp. The old
                # registry's contract stands: cross-camera suppression
                # hard-expires TTL seconds after the last FRESH
                # evidence, no matter how long a lock coasts.
                self._claim = _Claim(camera_id, tier, confidence, cur.at)
            else:
                self._claim = _Claim(camera_id, tier, confidence, now)
            return True
        if cur is None:
            if tier <= TIER_HELD:
                return False
            self._claim = _Claim(camera_id, tier, confidence, now)
            return True
        if (tier, confidence) > (cur.tier, cur.confidence):
            self._claim = _Claim(camera_id, tier, confidence, now)
            return True
        return False

    def vacate(self, camera_id: UUID) -> None:
        """Withdraw `camera_id`'s claim immediately (lock released on
        positive contradiction or verified emptiness) — suppression of
        other cameras must not outlive disproven presence."""
        if self._claim is not None and self._claim.camera_id == camera_id:
            self._claim = None

    def mark(self, camera_id: UUID) -> None:
        """Back-compat shim for the pre-W3 API (tests + any stragglers):
        a bare mark is a body-tier claim at the old 0.6 bar."""
        self.claim(camera_id, tier=TIER_BODY, confidence=0.6)

    # ── reading ────────────────────────────────────────────────

    def holder(self) -> UUID | None:
        """Camera currently holding the home-wide patient lock, or
        None when vacant/stale."""
        cur = self._claim
        if cur is None or (time.monotonic() - cur.at) > self._ttl:
            return None
        return cur.camera_id

    def confirmed_elsewhere(self, camera_id: UUID) -> bool:
        """True when ANOTHER camera holds a fresh claim. Drives the
        scheduler's patient_elsewhere suppression (unchanged API)."""
        h = self.holder()
        return h is not None and h != camera_id

    # ── patient-focus governor (2026-07-22) ────────────────────

    def note_tick(
        self,
        camera_id: UUID,
        *,
        people_present: bool,
        holder_seen: bool | None = None,
    ) -> None:
        """Per-tick heartbeat from each camera: occupancy, and — when
        this camera holds the seat — whether the patient was actually
        SEEN this tick (the escalation trigger for other rooms)."""
        now = time.monotonic()
        self._occupancy[camera_id] = (bool(people_present), now)
        if holder_seen is not None:
            self._holder_seen = (bool(holder_seen), now)

    def occupied_count(self, *, fresh_s: float = 30.0) -> int:
        """Cameras reporting people within the freshness window."""
        now = time.monotonic()
        return sum(
            1 for present, at in self._occupancy.values()
            if present and (now - at) <= fresh_s
        )

    def holder_seen(self, *, fresh_s: float = 30.0) -> bool:
        """True only when the holder camera FRESHLY reports the
        patient in sight. Fail-open is False: unknown or stale sight
        means other rooms must NOT relax into minimal mode — they keep
        looking, so a room change is never blinded (the handoff
        escalation trigger)."""
        if self.holder() is None or self._holder_seen is None:
            return False
        seen, at = self._holder_seen
        return seen and (time.monotonic() - at) <= fresh_s


class PresenceRegistry:
    """One `PatientPresence` per patient, created on demand.

    `PatientPresence` holds a single claim meaning "the patient is on
    camera X", and RoomWorker kept one instance per ROOM. That was sound
    while a room was a patient: `room_worker` says so in its own comment,
    "one per room (= per patient)".

    Account rooms broke it. Every sensor on an account now shares one
    room, so one registry spanned every resident: Dad face-confirmed in
    the kitchen made `confirmed_elsewhere()` true for MOM's bedroom
    sensor, which skips her patient-scoped detectors, degrades her re-id
    fidelity and can skip her tick outright. The registry has to be keyed
    by the thing the claim is actually about.
    """

    def __init__(self, *, ttl_s: float = CONFIRM_TTL_S) -> None:
        self._ttl_s = ttl_s
        self._by_patient: dict[UUID, PatientPresence] = {}

    def for_patient(self, patient_id: UUID | None) -> PatientPresence | None:
        """The registry for one patient. None for an unassigned sensor —
        there is no patient to be present or elsewhere, and every caller
        already treats a missing registry as "no opinion"."""
        if patient_id is None:
            return None
        got = self._by_patient.get(patient_id)
        if got is None:
            got = self._by_patient[patient_id] = PatientPresence(
                ttl_s=self._ttl_s)
        return got
