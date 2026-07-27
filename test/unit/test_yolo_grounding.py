"""YOLO-grounding remap — persistent ids via box-IoU (2026-07-24).

Same contract as the SAM3 mask remap: fresh grounder boxes keep their
carried id on sufficient overlap, get a new id otherwise, and an empty
detection set clears everyone (grounds stay authoritative on
existence regardless of which model grounds).
"""

from sam_service.continuous import _box_iou, remap_boxes_to_ids

IOU = 0.30


def test_overlapping_box_keeps_carried_id():
    seeded, nxt = remap_boxes_to_ids(
        [(10, 10, 50, 100)], {7: (12, 12, 52, 102)}, next_id=8,
        iou_floor=IOU,
    )
    assert list(seeded) == [7] and nxt == 8


def test_new_person_gets_new_id():
    seeded, nxt = remap_boxes_to_ids(
        [(10, 10, 50, 100), (400, 10, 440, 100)],
        {7: (12, 12, 52, 102)}, next_id=8, iou_floor=IOU,
    )
    assert set(seeded) == {7, 8} and nxt == 9


def test_empty_detections_clear_everyone():
    seeded, _ = remap_boxes_to_ids(
        [], {7: (12, 12, 52, 102)}, next_id=8, iou_floor=IOU,
    )
    assert seeded == {}


def test_one_id_never_claimed_twice():
    seeded, nxt = remap_boxes_to_ids(
        [(10, 10, 50, 100), (11, 11, 51, 101)],
        {7: (12, 12, 52, 102)}, next_id=8, iou_floor=IOU,
    )
    assert set(seeded) == {7, 8} and nxt == 9


def test_box_iou_sane():
    assert _box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert _box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# ── YOLO-seg grounding (2026-07-25): mask currency, same remap ──────

import numpy as np

from sam_service.continuous import CameraSession


def _m(y0, y1, x0, x1, h=36, w=64):
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def test_remap_masks_overlap_keeps_carried_id():
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {7: _m(5, 30, 10, 25)}
    s.next_id = 8
    seeded = s._remap_masks([_m(6, 31, 11, 26)])
    assert list(seeded) == [7] and s.next_id == 8


def test_remap_masks_disjoint_gets_new_id():
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {7: _m(5, 30, 10, 25)}
    s.next_id = 8
    seeded = s._remap_masks([_m(6, 31, 11, 26), _m(5, 30, 45, 60)])
    assert set(seeded) == {7, 8} and s.next_id == 9


def test_remap_masks_adjacent_silhouettes_do_not_swap():
    """Two side-by-side people: each fresh mask must reclaim ITS OWN
    carried id even though their bounding boxes would overlap — the
    exact failure mode that retired the box grounder."""
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {3: _m(5, 30, 10, 24), 4: _m(5, 30, 26, 40)}
    s.next_id = 5
    seeded = s._remap_masks([_m(6, 31, 27, 41), _m(6, 31, 11, 25)])
    # fresh[0] sits on carried 4, fresh[1] on carried 3
    assert (seeded[4] == _m(6, 31, 27, 41)).all()
    assert (seeded[3] == _m(6, 31, 11, 25)).all()
    assert s.next_id == 5


def test_ground_yoloseg_none_signals_sam3_fallback(monkeypatch):
    from sam_service import person_trigger

    monkeypatch.setattr(person_trigger, "person_masks", lambda _f: None)
    s = CameraSession(camera_id="cam-t")
    frames = [np.zeros((36, 64, 3), dtype=np.uint8)]
    assert s._ground_yoloseg(frames) is None


def test_ground_yoloseg_empty_is_authoritative_nobody(monkeypatch):
    from sam_service import person_trigger

    monkeypatch.setattr(person_trigger, "person_masks", lambda _f: [])
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {7: _m(5, 30, 10, 25)}
    frames = [np.zeros((36, 64, 3), dtype=np.uint8)]
    assert s._ground_yoloseg(frames) == {}


def test_ground_yoloseg_masks_flow_through_shared_remap(monkeypatch):
    from sam_service import person_trigger

    monkeypatch.setattr(
        person_trigger, "person_masks", lambda _f: [_m(6, 31, 11, 26)],
    )
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {7: _m(5, 30, 10, 25)}
    s.next_id = 8
    seeded = s._ground_yoloseg([np.zeros((36, 64, 3), dtype=np.uint8)])
    assert list(seeded) == [7]
    assert seeded[7].dtype == bool and seeded[7].shape == (36, 64)
