"""The single home-wide patient seat — Aryaman's rule (2026-07-24).

Seat = highest combined score. A challenger takes it only by beating
the seated reference score by δ. The reference lives and dies with its
track: silence never costs the seat (the tracker carries it between
scores), and there is no decay machinery — when the guest's track dies,
their reference dies with it (deliberately deferred: add decay only if
the pilot shows a stuck wrong seat; plan §6).

Empty-seat rule: lock the best candidate only if it clears the floor
AND beats the runner-up by δ — two plausible patients lock neither
(UNKNOWN is recoverable; a wrong patient is not).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

log = logging.getLogger(__name__)

TrackKey = tuple[str, int]  # (camera_id, track_id)


@dataclass(frozen=True)
class Candidate:
    camera_id: str
    track_id: int
    score: float
    anchor: str                    # "face" | "body"
    # Marginal face evidence not yet backed by a fused track template
    # (few sightings, none overwhelming). Provisional candidates may
    # HOLD or RATCHET a seat they already have, but can never TAKE or
    # STEAL one — the aunt lesson (2026-07-24): one flattering frame
    # among twenty rejections must not seat an impostor.
    provisional: bool = False

    @property
    def key(self) -> TrackKey:
        return (self.camera_id, self.track_id)


@dataclass(frozen=True)
class Seat:
    camera_id: str
    track_id: int
    score: float                   # the reference a challenger must beat
    anchor: str                    # provenance of the CURRENT reference
    confirmed_at: float            # when the reference was last earned
    since: float                   # when this track first took the seat
    # "The face confirmation lives on the TRACK": once this track has
    # EVER produced face-grade evidence, it stays face-confirmed for
    # its whole life (harvest eligibility) — even after a higher
    # body-pool score later takes over the anchor (live incident
    # 2026-07-24: the anchor flipped to "body" on self-matching fresh
    # harvests and harvesting silently stopped).
    face_confirmed: bool = False
    # How many SEPARATE probe rounds produced face-grade evidence for
    # this track. Harvest requires ≥2 (anti-poisoning, 2026-07-24): a
    # wrong lock that cannot re-confirm itself by face must never get
    # to write the body gallery — it was the harvest, not the lock,
    # that entrenched the aunt.
    face_confirms: int = 0

    @property
    def key(self) -> TrackKey:
        return (self.camera_id, self.track_id)


def update_seat(
    seat: Optional[Seat],
    candidates: list[Candidate],
    *,
    live: set[TrackKey],
    floor: float,
    delta: float,
    now: float,
) -> Optional[Seat]:
    """Fold one round of identity scores into the seat. Pure."""
    # 1. The seat dies with its track — no ghost references.
    if seat is not None and seat.key not in live:
        log.info("identity.seat_died key=%s", seat.key)
        seat = None

    scored = [c for c in candidates if c.key in live]
    if not scored:
        return seat                        # silence holds the seat

    best = max(scored, key=lambda c: c.score)

    if seat is None:
        if best.score < floor:
            return None
        if best.provisional:
            # Evidence not yet persistent — wait for the fused template
            # (a genuine patient re-confirms within a probe or two; an
            # impostor's outlier can't).
            return None
        runner_up = max(
            (c.score for c in scored if c.key != best.key), default=0.0,
        )
        if best.score - runner_up < delta and runner_up >= floor:
            # Two plausible patients — lock neither.
            log.info(
                "identity.seat_ambiguous best=%.2f runner_up=%.2f", best.score,
                runner_up,
            )
            return None
        log.info(
            "identity.seat_locked key=%s score=%.2f anchor=%s",
            best.key, best.score, best.anchor,
        )
        return Seat(
            camera_id=best.camera_id, track_id=best.track_id,
            score=best.score, anchor=best.anchor,
            confirmed_at=now, since=now,
            face_confirmed=best.anchor == "face",
            face_confirms=1 if best.anchor == "face" else 0,
        )

    if best.key == seat.key:
        # Same track: ONLY FACE evidence ratchets the reference. Once
        # this track's harvests fill the pool, its own body scores are
        # partly self-referential (matching crops of itself from
        # seconds ago hit ~0.95+ → conf ~1.0) — letting them raise the
        # bar made the seat unassailable live on 2026-07-24. Weak later
        # evidence never demotes either way (silence holds).
        updated = seat
        if best.anchor == "face":
            if best.score > seat.score:
                updated = replace(
                    updated, score=best.score, anchor="face",
                    confirmed_at=now,
                )
            if not updated.face_confirmed:
                updated = replace(
                    updated, face_confirmed=True, confirmed_at=now,
                )
            # A face round deepens the confirmation count ONLY when it
            # clears the floor — sub-floor face evidence is not a
            # confirmation (live 2026-07-24: a wrongly-seated track's
            # 0.31 round would have counted as its second confirmation
            # and opened the harvest). The harvest gate (≥2) reads this.
            if best.score >= floor:
                updated = replace(
                    updated, face_confirms=updated.face_confirms + 1,
                )
        return updated

    return _challenge(seat, best, delta=delta, now=now)


def notary_contradiction(
    seat: Optional[Seat], candidates: list[Candidate], *, floor: float,
) -> Optional[Candidate]:
    """The swap signature (2026-07-24, twice live): the tracker traded
    two people's track numbers, so the SEATED number's own evidence
    collapses below the floor while another track sustains face-grade,
    history-backed evidence. Returns that challenger, else None. The
    caller requires this verdict on CONSECUTIVE rounds before healing
    — the seat follows the face, never a single frame. Silence alone
    never triggers this: a challenger must exist AND the holder must
    be scoring under the floor."""
    if seat is None:
        return None
    seated = next((c for c in candidates if c.key == seat.key), None)
    if seated is not None and seated.score >= floor:
        return None                        # holder still earns its seat
    return max(
        (
            c for c in candidates
            if c.key != seat.key and c.anchor == "face"
            and not c.provisional and c.score >= floor
        ),
        key=lambda c: c.score, default=None,
    )


def heal_seat(challenger: Candidate, *, now: float) -> Seat:
    """Transfer the seat to where the face went (swap healing)."""
    log.info(
        "identity.seat_healed to=%s score=%.2f — notary moved",
        challenger.key, challenger.score,
    )
    return Seat(
        camera_id=challenger.camera_id, track_id=challenger.track_id,
        score=challenger.score, anchor="face",
        confirmed_at=now, since=now,
        face_confirmed=True, face_confirms=1,
    )


def _challenge(
    seat: Seat, best: Candidate, *, delta: float, now: float,
) -> Optional[Seat]:
    if best.score >= seat.score + delta:
        if best.provisional:
            # An outlier frame can't STEAL a seat either (same aunt
            # mechanism, occupied variant).
            return seat
        log.info(
            "identity.seat_moved from=%s to=%s score=%.2f>%.2f+δ",
            seat.key, best.key, best.score, seat.score,
        )
        return Seat(
            camera_id=best.camera_id, track_id=best.track_id,
            score=best.score, anchor=best.anchor,
            confirmed_at=now, since=now,
            face_confirmed=best.anchor == "face",
            face_confirms=1 if best.anchor == "face" else 0,
        )
    return seat
