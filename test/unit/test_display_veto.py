"""Display-veto unit suite — the screen-face incident, as tests."""

from __future__ import annotations

import numpy as np

from identity.display_veto import ScreenMap, apply_veto, containment
from identity.wire import RemoteObservations

# The measured geometry from the 2026-07-21 8:44 incident: laptop
# screen region ~[626,163,707,222]; phantom screen-person inside it
# (containment 1.0); the real patient's body box overlapped ~0.13.
SCREEN = (626.0, 163.0, 707.0, 222.0)
PHANTOM = (655.0, 174.0, 683.0, 218.0)
PATIENT = (639.0, 121.0, 784.0, 294.0)


def rows(*entries):
    """entries: (tid, box) -> one frame's (N, 8) row array."""
    return np.array(
        [[*box, float(tid), 0.9, 0.0, 0.0] for tid, box in entries],
        dtype=np.float32,
    )


def obs(rows_by_frame, face_tids=()):
    o = RemoteObservations()
    o.rows_by_frame = rows_by_frame
    o.face_obs = {(tid, 0): ["dummy"] for tid in face_tids}
    return o


def test_measured_incident_separation():
    assert containment(PHANTOM, SCREEN) == 1.0
    # Live probe measured ≤0.131 per-frame; this fixture pair rounds
    # to ~0.16 — either way a chasm below the 0.8 veto floor.
    assert containment(PATIENT, SCREEN) < 0.2


def test_phantom_vetoed_patient_survives():
    m = ScreenMap()
    m.add([SCREEN], now=100.0)
    o = obs(
        {0: rows((0, PATIENT), (1, PHANTOM)),
         2: rows((0, PATIENT), (1, PHANTOM)),
         4: rows((0, PATIENT), (1, PHANTOM))},
        face_tids=(0, 1),
    )
    o, dropped = apply_veto(o, m, now=101.0)
    assert dropped == {1}
    for fi in o.rows_by_frame:
        assert all(int(r[4]) == 0 for r in o.rows_by_frame[fi])
    assert (0, 0) in o.face_obs and (1, 0) not in o.face_obs


def test_walker_crossing_screen_not_vetoed():
    m = ScreenMap()
    big_tv = (300.0, 100.0, 700.0, 400.0)
    m.add([big_tv], now=100.0)
    inside = (400.0, 150.0, 500.0, 350.0)      # fully in front of TV
    outside = (650.0, 150.0, 780.0, 380.0)     # stepping out
    o = obs({0: rows((5, inside)), 2: rows((5, inside)),
             4: rows((5, outside))})
    _o, dropped = apply_veto(o, m, now=101.0)
    assert dropped == set()                    # one dip clears the veto


def test_single_frame_flicker_not_vetoed():
    m = ScreenMap()
    m.add([SCREEN], now=100.0)
    o = obs({0: rows((3, PHANTOM))})
    _o, dropped = apply_veto(o, m, now=101.0)
    assert dropped == set()


def test_region_expiry():
    m = ScreenMap(ttl_s=60.0)
    m.add([SCREEN], now=100.0)
    o = obs({0: rows((1, PHANTOM)), 2: rows((1, PHANTOM))})
    _o, dropped = apply_veto(o, m, now=200.0)  # 100s later: expired
    assert dropped == set()


def test_region_refresh_and_union_bound():
    m = ScreenMap(ttl_s=60.0)
    m.add([SCREEN], now=0.0)
    m.add([SCREEN], now=50.0)                  # same display refreshes
    assert len(m.live(now=55.0)) == 1
    # 100+ junk regions never grow the map unbounded.
    for i in range(40):
        m.add([(i * 30.0, 500.0, i * 30.0 + 20.0, 520.0)], now=50.0)
    assert len(m.live(now=55.0)) <= 16


def test_empty_map_is_inert():
    m = ScreenMap()
    o = obs({0: rows((1, PHANTOM))})
    _o, dropped = apply_veto(o, m, now=10.0)
    assert dropped == set()
