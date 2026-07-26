"""Full-frame face → track assignment (the lag-immune probe path).

Contract: a lone track owns every face; with several tracks the face
center must land inside exactly ONE padded box — ambiguity is always
discarded (wrong-person assignment is the failure identity exists to
prevent).
"""

from src.services.identity.loop import _assign_face


def test_t20_replay_no_single_track_shortcut():
    # 2026-07-24, the third wrong lock: the patient re-entered during
    # the ~50s before his track existed; the room's only track was the
    # aunt's, and "one track owns every face" credited HIS doorway
    # face to HER couch track until she was seated. A face outside the
    # sole track's box is now DISCARDED.
    assert _assign_face(999.0, 999.0, [(7, 0, 0, 10, 10)]) is None


def test_single_track_face_inside_box_assigns():
    assert _assign_face(5.0, 5.0, [(7, 0, 0, 10, 10)]) == 7


def test_face_inside_exactly_one_box_assigns():
    boxes = [(1, 0, 0, 100, 200), (2, 500, 0, 600, 200)]
    assert _assign_face(50.0, 100.0, boxes) == 1
    assert _assign_face(550.0, 100.0, boxes) == 2


def test_overlapping_boxes_discard():
    boxes = [(1, 0, 0, 100, 200), (2, 50, 0, 150, 200)]
    assert _assign_face(75.0, 100.0, boxes) is None


def test_face_outside_all_boxes_discards():
    boxes = [(1, 0, 0, 100, 200), (2, 500, 0, 600, 200)]
    assert _assign_face(300.0, 100.0, boxes) is None


def test_padding_admits_face_just_above_box():
    # Face slightly above the person box (head cropped by the tracker's
    # tight mask) still assigns via the 20% pad.
    assert _assign_face(50.0, -30.0, [(1, 0, 0, 100, 200), (2, 500, 500, 600, 700)]) == 1
