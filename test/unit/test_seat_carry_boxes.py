"""`carry_seat` needs the box of a track that is already dead.

This is the seam the bug lived in, and it is why testing `carry_seat`
alone would not have caught it. Given the dead track's box the function
is correct and always was. But its only caller sourced boxes from
`state.tracks` — LIVE tracks — and `carry_seat` is reached only once the
seated track is gone, so the lookup returned None and the function bailed
on its third line. Every time, since it was written.

Measured on the node 2026-08-25 before the fix: zero carries on either
camera, against 40 seat deaths that had exactly the one overlapping
successor it exists to find. The seat was instead re-earned by face
match, at 20-60s a time, roughly every 40 seconds all day.

So these tests assert against the CALLER's box supply, not the pure
function.
"""

from __future__ import annotations

from identity.loop import DEAD_BOX_TTL_S, IdentityHome
from identity.seat import Seat, carry_seat


class _FakeState:
    def __init__(self, tracks):
        self.tracks = tracks


class _FakeTrack:
    def __init__(self, bbox):
        self.bbox = bbox


class _FakeLoop:
    """Just enough of CameraLoop for `home_seat_boxes` to read it."""

    def __init__(self, live: dict, dead: dict):
        self.state = _FakeState({t: _FakeTrack(b) for t, b in live.items()})
        self._dead_boxes = dead


def _home(live, dead):
    home = IdentityHome.__new__(IdentityHome)
    home.cameras = {"cam1": _FakeLoop(live, dead)}
    return home


def test_a_just_dead_track_still_has_a_box():
    """The whole bug, in one assertion."""
    import time
    boxes = _home({7: (10, 10, 50, 90)},
                  {6: ((12, 10, 52, 90), time.time())}).home_seat_boxes()
    assert boxes[("cam1", 6)] == (12, 10, 52, 90)


def test_a_live_track_wins_over_a_recycled_dead_number():
    """Track ids get recycled. A stale corpse must never describe a
    number that is alive again."""
    import time
    boxes = _home({6: (99, 99, 120, 150)},
                  {6: ((12, 10, 52, 90), time.time())}).home_seat_boxes()
    assert boxes[("cam1", 6)] == (99, 99, 120, 150)


def test_an_old_corpse_is_forgotten():
    """A box older than the carry window is not evidence about where
    anyone is now — otherwise a seat could land on someone who walked
    into the space minutes later."""
    import time
    boxes = _home({}, {6: ((12, 10, 52, 90),
                           time.time() - DEAD_BOX_TTL_S - 1)}).home_seat_boxes()
    assert ("cam1", 6) not in boxes


def _seat(tid):
    return Seat(camera_id="cam1", track_id=tid, score=0.9, anchor="face",
                confirmed_at=0.0, since=0.0, face_confirmed=True)


def test_the_seat_moves_to_the_renumbered_track_end_to_end():
    """Death at T6, rebirth as T7 over the same space, seat follows —
    with its face confirmation intact, which is the point. Re-earning
    that costs 20-60s of somebody standing in front of a camera."""
    import time
    boxes = _home({7: (12, 11, 52, 91)},
                  {6: ((12, 10, 52, 90), time.time())}).home_seat_boxes()
    carried = carry_seat(_seat(6), {("cam1", 7)}, boxes)
    assert carried is not None
    assert carried.track_id == 7
    assert carried.face_confirmed is True
    assert carried.since == 0.0          # same person, not a new lock


def test_two_overlapping_successors_still_refuse_to_guess():
    """The conservatism that must survive this fix. Two candidates means
    we cannot say which is the person, and a wrong carry is a
    misattribution — the failure this system must not have."""
    import time
    boxes = _home({7: (12, 11, 52, 91), 8: (13, 12, 53, 92)},
                  {6: ((12, 10, 52, 90), time.time())}).home_seat_boxes()
    assert carry_seat(
        _seat(6), {("cam1", 7), ("cam1", 8)}, boxes) is None


def test_a_successor_holding_another_resident_is_never_taken():
    import time
    boxes = _home({7: (12, 11, 52, 91)},
                  {6: ((12, 10, 52, 90), time.time())}).home_seat_boxes()
    assert carry_seat(_seat(6), {("cam1", 7)}, boxes,
                      taken=frozenset({("cam1", 7)})) is None
