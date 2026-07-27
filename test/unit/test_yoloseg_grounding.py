"""YOLO-seg grounding — mask currency through the shared remap.

The seg grounder feeds the SAME mask-IoU remap contract as SAM3
(fresh mask keeps its carried id on sufficient overlap, new id
otherwise, empty set clears everyone). The retired box-grounding mode
and its remap were deleted 2026-07-26 — rectangles were the wrong
matching currency for adjacent people.
"""

# ── YOLO-seg grounding (2026-07-25): mask currency, same remap ──────

import numpy as np

from perception.continuous import CameraSession


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
    from perception import person_trigger

    monkeypatch.setattr(person_trigger, "person_masks", lambda _f: None)
    s = CameraSession(camera_id="cam-t")
    frames = [np.zeros((36, 64, 3), dtype=np.uint8)]
    assert s._ground_yoloseg(frames) is None


def test_ground_yoloseg_empty_is_authoritative_nobody(monkeypatch):
    from perception import person_trigger

    monkeypatch.setattr(person_trigger, "person_masks", lambda _f: [])
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {7: _m(5, 30, 10, 25)}
    frames = [np.zeros((36, 64, 3), dtype=np.uint8)]
    assert s._ground_yoloseg(frames) == {}


def test_ground_yoloseg_masks_flow_through_shared_remap(monkeypatch):
    from perception import person_trigger

    monkeypatch.setattr(
        person_trigger, "person_masks", lambda _f: [_m(6, 31, 11, 26)],
    )
    s = CameraSession(camera_id="cam-t")
    s.last_masks = {7: _m(5, 30, 10, 25)}
    s.next_id = 8
    seeded = s._ground_yoloseg([np.zeros((36, 64, 3), dtype=np.uint8)])
    assert list(seeded) == [7]
    assert seeded[7].dtype == bool and seeded[7].shape == (36, 64)
