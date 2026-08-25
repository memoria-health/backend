"""A track that blinks is not a track that left, and a renumbered
person is not a new person.

Both rules exist because of one measurement. On 2026-08-25 the resident
sat still in his bedroom and identity was lost repeatedly:

    05:47:34  seat_locked  face 0.90
    05:48:55  seat_died                 (81s)
    05:48:55  seat_locked  body 1.00
    05:49:01  seat_died                 (6s)
    05:49:02  seat_locked  body 1.00
    05:50:39  seat_died                 (97s)

He did not move. The tracker dropped his box and issued a new number,
and because the seat is keyed on that number, a face match costing
20-60 seconds to earn was destroyed each time.
"""

from __future__ import annotations

from identity.seat import Seat, carry_seat
from identity.state import MISSES_BEFORE_DEATH, CameraState

CAM = "cam1"


def _state():
    return CameraState(camera_id=CAM)


def _seat(tid: int, **kw) -> Seat:
    return Seat(camera_id=CAM, track_id=tid, score=0.9, anchor="face",
                confirmed_at=0.0, since=0.0, face_confirmed=True, **kw)


# ── the track blink ──────────────────────────────────────────────────

def test_one_missed_batch_does_not_kill_a_track() -> None:
    st = _state()
    st.apply({1: (0.0, 0.0, 10.0, 10.0)}, now=0.0)
    ev = st.apply({}, now=0.2)
    assert ev.deaths == [], "a single noisy batch killed the track"
    assert 1 in st.live_ids, "the track must stay live through the gap"


def test_a_track_that_really_left_still_dies() -> None:
    st = _state()
    st.apply({1: (0.0, 0.0, 10.0, 10.0)}, now=0.0)
    for i in range(MISSES_BEFORE_DEATH):
        ev = st.apply({}, now=0.2 * (i + 1))
    assert ev.deaths == [1]
    assert 1 not in st.live_ids


def test_a_blink_does_not_accumulate_into_a_death() -> None:
    """Seen, missing, seen, missing... must never reach the threshold."""
    st = _state()
    st.apply({1: (0.0, 0.0, 10.0, 10.0)}, now=0.0)
    for i in range(20):
        st.apply({}, now=1.0 + i)
        ev = st.apply({1: (0.0, 0.0, 10.0, 10.0)}, now=1.5 + i)
        assert ev.deaths == [], f"died on blink {i}"
    assert 1 in st.live_ids


def test_the_carried_track_keeps_its_last_box() -> None:
    """Downstream reads bbox every batch; a hole would read as a jump."""
    st = _state()
    st.apply({1: (5.0, 5.0, 15.0, 15.0)}, now=0.0)
    st.apply({}, now=0.2)
    assert st.tracks[1].bbox == (5.0, 5.0, 15.0, 15.0)


# ── the renumbering ──────────────────────────────────────────────────

def test_a_seat_moves_to_the_renumbered_track() -> None:
    """tid 22 dies, tid 23 is born in the same place. Same person."""
    box = (100.0, 100.0, 200.0, 300.0)
    carried = carry_seat(
        _seat(22), live={(CAM, 23)},
        boxes={(CAM, 22): box, (CAM, 23): (102.0, 101.0, 201.0, 299.0)},
    )
    assert carried is not None
    assert carried.track_id == 23
    # Everything earned travels with it — this is the whole point.
    assert carried.face_confirmed is True
    assert carried.anchor == "face"
    assert carried.score == 0.9


def test_a_seat_on_a_live_track_is_untouched() -> None:
    s = _seat(22)
    assert carry_seat(s, live={(CAM, 22)}, boxes={(CAM, 22): (0, 0, 1, 1)}) is s


def test_two_overlapping_candidates_carry_nothing() -> None:
    """Ambiguity must not be resolved by guessing.

    A wrong carry hands one resident's identity to another person, which
    is the failure this system cannot have.
    """
    box = (100.0, 100.0, 200.0, 300.0)
    assert carry_seat(
        _seat(22), live={(CAM, 23), (CAM, 24)},
        boxes={(CAM, 22): box,
               (CAM, 23): (101.0, 101.0, 199.0, 299.0),
               (CAM, 24): (103.0, 102.0, 202.0, 301.0)},
    ) is None


def test_a_distant_track_does_not_inherit() -> None:
    """Someone else across the room is not you."""
    assert carry_seat(
        _seat(22), live={(CAM, 23)},
        boxes={(CAM, 22): (100.0, 100.0, 200.0, 300.0),
               (CAM, 23): (900.0, 100.0, 1000.0, 300.0)},
    ) is None


def test_a_track_already_seated_by_someone_else_does_not_inherit() -> None:
    """Two residents, and the other one's track is where yours was.

    Carrying here would hand your identity to your housemate.
    """
    box = (100.0, 100.0, 200.0, 300.0)
    assert carry_seat(
        _seat(22), live={(CAM, 23)},
        boxes={(CAM, 22): box, (CAM, 23): (101.0, 101.0, 199.0, 299.0)},
        taken=frozenset({(CAM, 23)}),
    ) is None


def test_a_seat_never_carries_across_cameras() -> None:
    assert carry_seat(
        _seat(22), live={("cam2", 23)},
        boxes={(CAM, 22): (100.0, 100.0, 200.0, 300.0),
               ("cam2", 23): (100.0, 100.0, 200.0, 300.0)},
    ) is None
