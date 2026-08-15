"""Empty-room economy — a still empty room costs ZERO GPU.

2026-07-24 (Aryaman): an empty room's poll used to ship a dense
40-frame batch (~4s GPU per 20s per room). Now: no latched motion →
no pod call at all; latched motion → a handful of fresh frames only.
Occupied rooms are untouched. The latch is consumed only on a
successful poll, and starts True so the first-ever poll always runs.
"""

import asyncio
from types import SimpleNamespace

import numpy as np

from identity.loop import EMPTY_POLL_FRAMES, CameraLoop


class _SamSpy:
    def __init__(self):
        self.calls: list[int] = []  # frame counts per observe()

    async def observe(self, jpegs, **_kw):
        self.calls.append(len(jpegs))
        return SimpleNamespace(
            rows_by_frame={}, screen_regions=(), tracks=(), epoch="e1",
        )


def _loop_with(sam) -> CameraLoop:
    # `sam_for` mirrors IdentityHome's per-camera pod routing (2026-08-01).
    # With no extra workers configured every camera resolves to `sam`,
    # which is exactly what the real one does when NEMARA_POD_URLS is
    # unset — so this double stays faithful without modelling the pool.
    home = SimpleNamespace(
        sam=sam, pods=[], sam_for=lambda _cam_id: sam,
        pod_lock=asyncio.Lock(),
        lock_for=lambda _client: asyncio.Lock(),  # per-worker admission
        seat=None, cameras={},
        journal=None, name_of=lambda cam_id: cam_id,
    )
    lp = CameraLoop(camera_id="a", home=home)
    frame = np.zeros((36, 64, 3), dtype=np.uint8)
    for i in range(30):
        lp.push_frame(1000.0 + i * 0.1, frame)
    return lp


def test_motion_latch_starts_armed():
    assert _loop_with(_SamSpy()).motion_pending is True


def test_still_empty_room_skips_pod_entirely():
    sam = _SamSpy()
    lp = _loop_with(sam)
    lp._last_shipped = lp.ring[-1]        # not the first-ever poll
    lp.motion_pending = False             # still room
    asyncio.run(lp._step())
    assert sam.calls == []                # zero GPU


def test_motion_poll_ships_handful_and_consumes_latch(monkeypatch):
    import identity.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "time", lambda: 1003.0)
    sam = _SamSpy()
    lp = _loop_with(sam)
    lp._last_shipped = lp.ring[0]
    lp.motion_pending = True
    asyncio.run(lp._step())
    assert sam.calls == [EMPTY_POLL_FRAMES]   # fresh frames, not dense
    assert lp.motion_pending is False          # latch consumed on success


def test_pod_error_keeps_latch_armed(monkeypatch):
    import identity.loop as loop_mod

    monkeypatch.setattr(loop_mod.time, "time", lambda: 1003.0)

    class _Boom:
        async def observe(self, jpegs, **_kw):
            raise RuntimeError("pod down")

    lp = _loop_with(_Boom())
    lp._last_shipped = lp.ring[0]
    lp.motion_pending = True
    try:
        asyncio.run(lp._step())
    except RuntimeError:
        pass                               # run() catches this in prod
    assert lp.motion_pending is True       # retry next poll
