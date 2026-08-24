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


# ── movement, and why a screen is different (2026-08-24) ───────────
#
# The camera has no depth — one lens, no stereo, no sensor — so a person
# standing against a painted wall and a face painted ON it are the same
# pixels. Containment alone cannot separate them, and the rule discards
# anything sitting >=80% inside a region on every frame it is seen.
#
# Movement separates them: a painting is nailed to a wall. But that
# reasoning does NOT transfer to a screen, because a screen shows people
# who walk about — which is the 8:44 incident this whole file exists
# for, a face on a laptop minting person tracks and parking a lock for
# eleven minutes. So the two kinds of region are judged differently.
import numpy as np

from identity.display_veto import KIND_SCREEN, KIND_STATIC, ScreenMap

_REGION = (100.0, 100.0, 400.0, 400.0)


def _track(dx: float, tid: int = 7):
    """A small box moving `dx` per frame, always INSIDE _REGION."""
    return {
        i: np.array([[150.0 + i * dx, 150.0, 200.0 + i * dx, 250.0, tid]])
        for i in range(3)
    }


def _veto(kind: str, dx: float) -> bool:
    m = ScreenMap()
    m.add([_REGION], now=0.0, kind=kind)
    return bool(m.veto_track_ids(_track(dx), now=1.0))


def test_a_still_thing_inside_a_painting_is_dropped():
    """The phantom case: painted faces on a wall, which do not move."""
    assert _veto(KIND_STATIC, dx=0) is True


def test_a_moving_thing_inside_a_painting_is_kept():
    """THE guard. A person standing in front of their own artwork is
    contained exactly like the artwork is, and discarding them is
    silent — nothing downstream ever learns they were in the room."""
    assert _veto(KIND_STATIC, dx=30) is False


def test_a_moving_thing_inside_a_SCREEN_is_still_dropped():
    """The 8:44 incident. A televised person walks about, so movement
    proves nothing on a display and the containment rule must stand on
    its own. Without this the movement guard would re-open the exact
    hole this file was written to close."""
    assert _veto(KIND_SCREEN, dx=30) is True


def test_a_still_thing_inside_a_screen_is_dropped():
    assert _veto(KIND_SCREEN, dx=0) is True


def test_screen_wins_when_a_region_is_seen_as_both():
    """Concepts are searched one per pass and can land on the same box.
    The stricter rule has an incident behind it, so it wins."""
    m = ScreenMap()
    m.add([_REGION], now=0.0, kind=KIND_STATIC)
    m.add([_REGION], now=0.0, kind=KIND_SCREEN)
    assert bool(m.veto_track_ids(_track(30), now=1.0)) is True


def test_the_movement_floor_is_above_measured_jitter():
    """Static painted faces drifted 6.5-9.0px with the room EMPTY —
    mask jitter, not motion. A floor at or below that spares the very
    things the veto exists to drop."""
    assert ScreenMap().move_px > 9.0


def test_the_containment_floor_is_above_the_measured_paintings():
    """Every painted face actually dropped in the TV Room scored
    0.957-1.000, so a floor of 0.93 loses none of them — and a person
    has to be engulfed far more completely before they are at risk."""
    assert 0.93 <= ScreenMap().containment_floor <= 0.95
