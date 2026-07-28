"""Unit: CameraIngest._verify_wake — the wiring between the gateway's
motion hint and the block-motion authority (review fix 2026-07-17:
the veto mechanism itself was untested).

Pins: grain window ⇒ incoherent verdict (caller vetoes); coherent
window ⇒ coherent verdict (caller trusts hint); assess raising ⇒ None
(fail OPEN — analyze); cold/short buffer ⇒ None (fail open).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from src.services.detection.camera_ingest import CameraIngest
from src.services.detection.motion_cache import MotionCache

W, H = 640, 360


def _ingest() -> CameraIngest:
    return CameraIngest(
        camera_id=uuid4(),
        patient_id=uuid4(),
        sessionmaker=None,          # _verify_wake never touches the DB
        motion_cache=MotionCache(ttl_seconds=60),
        settings=SimpleNamespace(
            frame_buffer_retention_seconds=20.0,
            livekit_worker_target_fps=4,
            frame_buffer_target_fps=4,
            clip_fps=4,
        ),
        reid=None,
    )


def _push(ingest: CameraIngest, y: np.ndarray, at: float) -> None:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    ingest.frame_buffer.push(
        width=W, height=H, data=y.tobytes() + uv.tobytes(), captured_at=at,
    )


def _grain(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(
        np.full(W * H, 60.0) + rng.normal(0, 2.0, W * H), 0, 255,
    ).astype(np.uint8)


def _with_blob(y: np.ndarray, x0: int = 300) -> np.ndarray:
    y2 = y.reshape(H, W).copy()
    y2[150:190, x0:x0 + 40] = 200
    return y2.reshape(-1)


async def test_grain_window_returns_incoherent() -> None:
    ingest = _ingest()
    t0 = time.monotonic() - 2.5
    for i in range(10):
        _push(ingest, _grain(i), t0 + i * 0.25)
    check = await ingest._verify_wake()
    assert check is not None and check.coherent is False


async def test_coherent_motion_returns_coherent() -> None:
    ingest = _ingest()
    t0 = time.monotonic() - 2.5
    for i in range(8):
        _push(ingest, _grain(i), t0 + i * 0.25)
    _push(ingest, _with_blob(_grain(9)), t0 + 8 * 0.25)
    check = await ingest._verify_wake()
    assert check is not None and check.coherent is True


async def test_assess_raising_fails_open() -> None:
    ingest = _ingest()
    t0 = time.monotonic() - 2.5
    for i in range(10):
        _push(ingest, _grain(i), t0 + i * 0.25)

    def _boom(window):
        raise RuntimeError("verifier bug")

    ingest._block_motion.assess = _boom
    assert await ingest._verify_wake() is None      # caller trusts the hint


async def test_cold_buffer_fails_open() -> None:
    ingest = _ingest()                              # no frames at all
    assert await ingest._verify_wake() is None
    _push(ingest, _grain(0), time.monotonic())      # 1 frame < 3
    assert await ingest._verify_wake() is None
