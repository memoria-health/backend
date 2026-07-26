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
