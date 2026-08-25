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


def test_both_seat_carry_call_sites_use_a_method_their_owner_has() -> None:
    """The carry runs from TWO classes, and they are not the same class.

    `CameraLoop` has `_seat_boxes`; `IdentityHome` does not — seats live
    on the home, which owns several cameras and has no single one to
    ask. Calling the loop's method from the home raised AttributeError
    inside `fold_candidates`, so EVERY probe failed and no seat could
    ever lock. The symptom was standing in front of a camera and never
    being recognised, with nothing in the logs but `probe_failed`
    (2026-08-25).

    Checked structurally rather than by running a probe, because the
    failure was an attribute that does not exist on the receiver — the
    kind of thing only the real object graph reveals.
    """
    import ast
    import pathlib

    tree = ast.parse(
        pathlib.Path("gpu/identity/loop.py").read_text(encoding="utf-8"))
    owns: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            owns[node.name] = {
                m.name for m in node.body if isinstance(m, ast.FunctionDef)
            }

    # Every `self.<name>()` called inside a class must exist on it.
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for call in ast.walk(node):
            if (isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "self"
                    and call.func.attr in ("_seat_boxes", "home_seat_boxes")):
                assert call.func.attr in owns[node.name], (
                    f"{node.name} calls self.{call.func.attr}() but does not "
                    f"define it — this is the bug that stopped every seat "
                    f"from locking")


def test_a_rejected_track_dies_at_once() -> None:
    """The display filter's verdict is not a blink.

    The grace period exists so one noisy batch cannot destroy a face
    match. A track the filter has positively judged to be a painting is
    not missing — we looked and decided it is not a person.

    Measured 2026-08-25 in the TV Room: the canvas faces were dropped on
    99% of batches and spared on the odd one, when a mask wobble crossed
    the movement bar (inflated by a downshifted frame, where one pixel
    is 3.6x larger once normalised). Every spare reset the miss counter,
    so the phantoms never died and `solo` never became true — which in
    turn blocked face harvesting for the person who WAS in the room.
    """
    st = _state()
    st.apply({1: (0.0, 0.0, 10.0, 10.0)}, now=0.0)
    ev = st.apply({}, now=0.2, rejected={1})
    assert ev.deaths == [1], "a rejected track was given the grace period"
    assert 1 not in st.live_ids


def test_an_occasional_spare_cannot_keep_a_rejected_track_alive() -> None:
    """Reject, reject, spare, reject — the pattern that was surviving."""
    st = _state()
    st.apply({7: (0.0, 0.0, 10.0, 10.0)}, now=0.0)
    st.apply({}, now=0.2, rejected={7})          # rejected -> dead
    assert 7 not in st.live_ids
    # It comes back as a fresh track, and is rejected again just as fast.
    st.apply({7: (0.0, 0.0, 10.0, 10.0)}, now=0.4)
    ev = st.apply({}, now=0.6, rejected={7})
    assert ev.deaths == [7]


def test_a_merely_missing_track_still_gets_its_grace() -> None:
    """The fix must not undo the reason the counter exists."""
    st = _state()
    st.apply({3: (0.0, 0.0, 10.0, 10.0)}, now=0.0)
    ev = st.apply({}, now=0.2, rejected=set())
    assert ev.deaths == [], "a blink was treated as a rejection"
    assert 3 in st.live_ids
