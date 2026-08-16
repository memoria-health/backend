"""CameraSession portable state (stateless-worker mode) — round-trip,
degradation, and size sanity. numpy-only: no torch/SAM stack needed."""

import numpy as np
import pytest

from perception.continuous import CameraSession


def _mask(h=360, w=640, fill_from=100, fill_to=200):
    m = np.zeros((h, w), dtype=bool)
    m[fill_from:fill_to, fill_from:fill_to] = True
    return m


def test_round_trip_preserves_everything():
    s = CameraSession("cam1", reground_every=5)
    s.tick = 42
    s.next_id = 7
    s.last_ground = 40
    s._under_streak = 2
    s.last_masks = {3: _mask(), 5: _mask(fill_from=10, fill_to=50)}

    r = CameraSession.from_state("cam1", s.to_state(), reground_every=5)

    assert r.epoch == s.epoch
    assert r.tick == 42
    assert r.next_id == 7
    assert r.last_ground == 40
    assert r._under_streak == 2
    assert set(r.last_masks) == {3, 5}
    for tid in (3, 5):
        assert np.array_equal(r.last_masks[tid], s.last_masks[tid])


def test_state_is_json_serializable():
    import json

    s = CameraSession("cam1")
    s.last_masks = {1: _mask()}
    parsed = json.loads(json.dumps(s.to_state()))
    r = CameraSession.from_state("cam1", parsed)
    assert np.array_equal(r.last_masks[1], s.last_masks[1])


def test_ground_sentinel_survives():
    s = CameraSession("cam1")
    s.last_ground = -(10 ** 9)
    r = CameraSession.from_state("cam1", s.to_state())
    assert r.last_ground == -(10 ** 9)


@pytest.mark.parametrize("bad", [
    None,
    {},
    {"v": 2, "epoch": "x", "tick": 1},
    {"v": 1, "epoch": "x"},                      # missing fields
    {"v": 1, "epoch": "x", "tick": 1, "next_id": 1, "last_ground": 0,
     "masks": {"1": {"shape": [99999, 99999], "bits": ""}}},
])
def test_bad_state_degrades_to_fresh_session(bad):
    r = CameraSession.from_state("cam1", bad)
    assert r.tick == 0
    assert r.last_masks == {}
    # fresh epoch, not inherited from the bad state
    assert r.epoch != (bad or {}).get("epoch")


def test_empty_masks_dropped_on_restore():
    s = CameraSession("cam1")
    s.last_masks = {1: np.zeros((10, 10), dtype=bool), 2: _mask(10, 10, 2, 5)}
    r = CameraSession.from_state("cam1", s.to_state())
    assert set(r.last_masks) == {2}


def test_state_size_is_loopback_noise():
    import json

    s = CameraSession("cam1")
    s.last_masks = {i: _mask() for i in range(4)}   # 4 tracked people
    assert len(json.dumps(s.to_state())) < 200_000  # < 200KB at 4 tracks
