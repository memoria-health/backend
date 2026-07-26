"""Swap healing — the seat follows the face, not the track number.

Live 2026-07-24 (twice): the tracker traded two people's track ids
and the seat rode the number onto the wrong person; the ratcheted
reference then made an ordinary steal impossible. The contradiction
verdict fires only when the HOLDER's evidence has collapsed below the
floor AND a history-backed face challenger sustains above it — never
on silence, never on a single frame (the caller demands consecutive
rounds).
"""

from src.services.identity.seat import (
    Candidate,
    Seat,
    heal_seat,
    notary_contradiction,
)

FLOOR = 0.4


def _seat(tid=2, score=0.87):
    return Seat(
        camera_id="a", track_id=tid, score=score, anchor="body",
        confirmed_at=0.0, since=0.0, face_confirmed=True, face_confirms=3,
    )


def _cand(tid, score, anchor="face", provisional=False):
    return Candidate(
        camera_id="a", track_id=tid, score=score, anchor=anchor,
        provisional=provisional,
    )


def test_swap_signature_names_the_challenger():
    # Post-swap live pattern: holder (now the aunt) collapsed to 0.1,
    # the real patient sustains history-backed 0.44 on another track.
    ch = notary_contradiction(
        _seat(tid=2), [_cand(2, 0.10), _cand(1, 0.44)], floor=FLOOR,
    )
    assert ch is not None and ch.track_id == 1


def test_working_holder_blocks_contradiction():
    # Holder at body 1.0 (flywheel armed) is untouchable regardless of
    # how well another track scores.
    assert notary_contradiction(
        _seat(tid=2), [_cand(2, 1.0, anchor="body"), _cand(1, 0.60)],
        floor=FLOOR,
    ) is None


def test_silence_is_not_contradiction():
    # Holder absent from candidates AND no challenger → nothing fires
    # (patient asleep: silence holds the seat).
    assert notary_contradiction(_seat(tid=2), [], floor=FLOOR) is None


def test_provisional_challenger_cannot_heal():
    ch = notary_contradiction(
        _seat(tid=2),
        [_cand(2, 0.05), _cand(1, 0.95, provisional=True)],
        floor=FLOOR,
    )
    assert ch is None


def test_body_challenger_cannot_heal():
    # The notary is the FACE — a body-only challenger never moves the
    # seat this way (body pools can be contaminated by adjacency).
    ch = notary_contradiction(
        _seat(tid=2), [_cand(2, 0.05), _cand(1, 0.90, anchor="body")],
        floor=FLOOR,
    )
    assert ch is None


def test_heal_transfers_with_fresh_reference():
    seat = heal_seat(_cand(1, 0.44), now=100.0)
    assert (seat.track_id, seat.anchor, seat.face_confirmed) == (1, "face", True)
    assert seat.score == 0.44                 # fresh reference, no ratchet
