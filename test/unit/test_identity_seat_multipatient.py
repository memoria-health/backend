"""The second arbitration: which RESIDENT is this track (2026-08-20).

`update_seat` answers "which track is Alice", once per resident, and is
unchanged — test_identity_seat.py pins it. This file pins the rule that
sits on top once a camera can watch more than one person:

    one body is one person, and the stronger claim keeps the track.

The two arbitrations are deliberately separate. Keeping them apart is
what makes the one-resident case identical to the pre-multi-patient
code, which is the property the whole change rests on — hence the
parity test at the bottom.
"""

from __future__ import annotations

from uuid import UUID

from identity.seat import Seat, resolve_track_conflicts

ALICE = UUID("11111111-1111-1111-1111-111111111111")
BOB = UUID("22222222-2222-2222-2222-222222222222")


def _seat(tid: int, score: float, cam: str = "a") -> Seat:
    return Seat(
        camera_id=cam, track_id=tid, score=score, anchor="face",
        confirmed_at=0.0, since=0.0, face_confirmed=True, face_confirms=2,
    )


def test_one_resident_is_a_no_op():
    """PARITY GUARD. With a single resident there is nothing to
    arbitrate and the seats must come back untouched — same object
    graph, same scores, no reordering."""
    seats = {ALICE: _seat(7, 0.81)}
    assert resolve_track_conflicts(seats) == seats


def test_separate_tracks_both_keep_their_seats():
    """Alice on track 7, Bob on track 9 — the ordinary shared-lounge
    case, and nothing is contested."""
    seats = {ALICE: _seat(7, 0.81), BOB: _seat(9, 0.74)}
    assert resolve_track_conflicts(seats) == seats


def test_same_track_keeps_only_the_stronger_claim():
    """One body cannot be two people. If both residents' independent
    contests landed on track 7, the weaker claim is dropped — two
    people's routines must never be recorded off one body."""
    seats = {ALICE: _seat(7, 0.81), BOB: _seat(7, 0.62)}
    out = resolve_track_conflicts(seats)
    assert set(out) == {ALICE}
    assert out[ALICE].track_id == 7


def test_loser_is_unseated_not_moved_to_its_runner_up():
    """The loser goes to UNKNOWN rather than falling back to whatever
    track it scored second-best on. That runner-up already lost Bob's
    own δ contest, and re-seating it here would smuggle in a lock the
    arbitration rejected — the losing direction has to be the
    recoverable one."""
    seats = {ALICE: _seat(7, 0.90), BOB: _seat(7, 0.55)}
    out = resolve_track_conflicts(seats)
    assert BOB not in out


def test_contest_is_per_track_not_per_camera():
    """Two residents on the SAME camera but different tracks is the
    normal shared-room state, not a conflict. Only an identical
    (camera, track) is contested."""
    seats = {ALICE: _seat(7, 0.60, cam="a"), BOB: _seat(8, 0.95, cam="a")}
    assert resolve_track_conflicts(seats) == seats


def test_tie_resolves_the_same_way_every_time():
    """Coverage comes from a query with no guaranteed ORDER BY, so an
    exact tie must not depend on dict insertion order — otherwise
    attribution flaps between two residents tick to tick, and every
    flap is a wrong record for one of them."""
    forward = resolve_track_conflicts(
        {ALICE: _seat(7, 0.70), BOB: _seat(7, 0.70)},
    )
    reverse = resolve_track_conflicts(
        {BOB: _seat(7, 0.70), ALICE: _seat(7, 0.70)},
    )
    assert set(forward) == set(reverse)
    assert len(forward) == 1


def test_three_residents_leave_exactly_one_holder():
    carol = UUID("33333333-3333-3333-3333-333333333333")
    out = resolve_track_conflicts({
        ALICE: _seat(4, 0.55), BOB: _seat(4, 0.88), carol: _seat(4, 0.71),
    })
    assert set(out) == {BOB}


def test_empty_seats_survive_the_pass():
    assert resolve_track_conflicts({}) == {}
