"""Display-veto unit suite — the screen-face incident, as tests."""

from __future__ import annotations

import numpy as np

from identity.display_veto import ScreenMap

# Fractions of the frame — everything in the map is normalised.
_REGION = (0.10, 0.10, 0.40, 0.40)


def _track(dx: float, tid: int = 7):
    """A small box moving `dx` per frame, always INSIDE _REGION."""
    return {
        i: np.array([[0.15 + i * dx, 0.15, 0.20 + i * dx, 0.25, tid]])
        for i in range(3)
    }


def _veto(dx: float) -> bool:
    m = ScreenMap()
    m.add([_REGION], now=0.0)
    return bool(m.veto_track_ids(_track(dx), now=1.0))


def test_a_thing_inside_a_region_is_dropped():
    """The phantom case: painted faces on a wall."""
    assert _veto(dx=0.0) is True


def test_movement_no_longer_spares_anything():
    """There WAS a movement exemption here, and it is gone.

    It existed to protect a person standing in front of their own
    artwork. Measured on the real geometry (2026-08-25), it protected
    nothing: a painting panel is 0.05 x 0.11 of frame and a standing
    adult is 0.19 x 0.20, so the most of a person that can ever fall
    inside one is 0.16 containment — against a floor of 0.93. The
    containment floor alone already makes vetoing a person impossible,
    six times over.

    What the exemption did do was spare the paintings: a mask wobble of
    a couple of pixels, amplified by a downshifted frame where one pixel
    is 3.6x larger once normalised, crossed the movement bar often
    enough to keep the phantoms alive indefinitely.
    """
    assert _veto(dx=0.03) is True


def test_a_person_sized_box_is_never_vetoed():
    """The guard that actually does the work, stated directly.

    Real numbers from the TV Room: the father's track and one grounded
    painting panel. He cannot reach the floor however still he stands.
    """
    m = ScreenMap()
    panel = (0.9006, 0.2006, 0.9549, 0.3086)
    m.add([panel], now=0.0)
    person = (724 / 2304, 518 / 1296, 1156 / 2304, 778 / 1296)
    rows = {
        i: np.array([[person[0], person[1], person[2], person[3], 5]])
        for i in range(3)
    }
    assert m.veto_track_ids(rows, now=1.0) == set()



def test_the_containment_floor_is_above_the_measured_paintings():
    """Every painted face actually dropped in the TV Room scored
    0.957-1.000, so a floor of 0.93 loses none of them — and a person
    has to be engulfed far more completely before they are at risk."""
    assert 0.93 <= ScreenMap().containment_floor <= 0.95


# ── regions survive a restart (2026-08-24) ────────────────────────
#
# The map lived only in memory, so every restart began blind and
# re-tracked every painted face on the wall until the next search two
# minutes later — and the tracks born in that window live on. Measured:
# seven phantom tracks, all with the same 17.6s age, seconds after a
# restart. Restarts are not rare and not all deliberate; the pod
# watchdog exits on its own several times a day.
import json
import tempfile
from pathlib import Path

from identity.display_veto import load_regions, save_regions


def _tmp() -> str:
    return str(Path(tempfile.mkdtemp()) / "regions.json")


def test_regions_survive_a_restart():
    """Furniture does not move, so the map should not start blind.

    No `kind` any more: with one rule, a television and a painting are
    treated identically, and the field existed only to pick who got the
    movement exemption.
    """
    path = _tmp()
    m = ScreenMap()
    m.add([(0.1, 0.2, 0.3, 0.4)])
    m.add([(0.5, 0.6, 0.7, 0.8)])
    save_regions(path, m.live())

    assert load_regions(path) == [
        (0.1, 0.2, 0.3, 0.4),
        (0.5, 0.6, 0.7, 0.8),
    ]


def test_a_missing_file_is_not_an_error():
    """First boot on a fresh node. Blind for one search interval is the
    correct outcome; refusing to start is not."""
    assert load_regions("/nonexistent/regions.json") == []


def test_a_corrupt_file_is_not_an_error():
    """A half-written file must cost one blind window, never a tick."""
    path = _tmp()
    Path(path).write_text("{{{ not json")
    assert load_regions(path) == []


def test_rows_that_make_no_sense_are_skipped_individually():
    """One bad row must not discard the good ones alongside it.

    Rows carry a format version now: v1 held PIXELS, v2 holds fractions
    of the frame, and reading one as the other vetoes nothing while
    looking exactly like a room with no displays in it.
    """
    from identity.display_veto import _REGION_FORMAT
    path = _tmp()
    Path(path).write_text(json.dumps([
        {"box": [0.1, 0.2, 0.3, 0.4], "v": _REGION_FORMAT},
        {"box": ["x", "y"], "v": _REGION_FORMAT},
        {"nope": True},
        {"box": [0.9, 0.9, 0.95, 0.95], "v": _REGION_FORMAT},
    ]))
    got = load_regions(path)
    assert got == [(0.1, 0.2, 0.3, 0.4), (0.9, 0.9, 0.95, 0.95)]


# ── Shields: signals that can only SPARE a track, never condemn one ────
#
# The geometry above is one model's opinion filtered by containment, so a
# wrong region and a wrong veto are the same mistake — and the mistake is
# silent, which for a fall detector is the worst shape a bug can have.
# These two shields fail differently from the geometry, so a real person
# now has to lose three arguments instead of one.


def _still_person_inside_a_big_region(tid: int = 3):
    """The hole the geometry cannot close on its own.

    A person standing still in front of a artwork large enough to contain
    them, whose track was renumbered so `moved` is empty. Containment is
    1.0 on every frame; nothing in the geometry can tell them from paint.
    """
    region = (0.05, 0.05, 0.95, 0.95)
    person = (0.40, 0.40, 0.55, 0.75)
    rows = {i: np.array([[*person, tid]]) for i in range(3)}
    return region, person, rows


def test_a_still_person_in_a_huge_artwork_is_vetoed_without_a_shield():
    """Stated as a test so the hole cannot quietly reopen."""
    region, _person, rows = _still_person_inside_a_big_region()
    m = ScreenMap()
    m.add([region], now=0.0)
    assert m.veto_track_ids(rows, now=1.0) == {3}


def test_yolo_agreement_spares_that_person():
    """A second model with different failure modes breaks the tie.

    Measured on the live TV Room frame 2026-08-25: YOLO26x put zero boxes
    on twelve painted faces even at conf 0.05, while SAM3 grounded seven
    of them as people at 0.4.
    """
    region, person, rows = _still_person_inside_a_big_region()
    m = ScreenMap()
    m.add([region], now=0.0)
    assert m.veto_track_ids(rows, now=1.0, person_boxes=(person,)) == set()


def test_an_identity_seat_spares_that_person():
    """Not geometric, so it holds when the geometry is wrong."""
    region, _person, rows = _still_person_inside_a_big_region()
    m = ScreenMap()
    m.add([region], now=0.0)
    assert m.veto_track_ids(rows, now=1.0, protected={3}) == set()


def test_yolo_silence_changes_nothing():
    """The whole reason YOLO is a shield and not a judge.

    It misses real people — small, dark, partly hidden, at night. If its
    silence could condemn, every miss would become a suppressed person.
    Here it sees nobody and the painting is vetoed exactly as before.
    """
    m = ScreenMap()
    m.add([_REGION], now=0.0)
    assert m.veto_track_ids(_track(dx=0.0), now=1.0, person_boxes=()) == {7}


def test_a_yolo_box_elsewhere_does_not_spare_the_painting():
    """The shield is per-track, not per-frame.

    A real person in the room must not resurrect the painted faces on the
    wall behind them — which is the ordinary state of that living room.
    """
    m = ScreenMap()
    m.add([_REGION], now=0.0)
    person_across_the_room = (0.60, 0.60, 0.75, 0.90)
    assert m.veto_track_ids(
        _track(dx=0.0), now=1.0,
        person_boxes=(person_across_the_room,)) == {7}
