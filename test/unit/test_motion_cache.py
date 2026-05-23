"""Unit tests for `MotionCache`.

The cache is one in-process dict guarded by a TTL — three tests
cover the meaningful behavior: a fresh write reads back, a stale
read returns False, and `drop` forgets the entry. Plus a couple of
guard tests for the constructor and the "missing camera = False"
default the tick scheduler relies on.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.detection.motion_cache import MotionCache


def test_init_rejects_nonpositive_ttl() -> None:
    with pytest.raises(ValueError):
        MotionCache(ttl_seconds=0)


def test_unknown_camera_returns_false() -> None:
    """The tick reads `cache.get(camera_id)` once per cycle. A
    camera that's never published motion_state must read as
    `in_motion=False` so the closed-state short-circuit still works.
    """
    cache = MotionCache(ttl_seconds=15)
    assert cache.get(uuid4(), now=100.0) is False


def test_put_then_get_within_ttl_returns_state() -> None:
    cache = MotionCache(ttl_seconds=15)
    cam = uuid4()
    cache.put(cam, in_motion=True, received_at=100.0)
    # 10s later, still inside the 15s TTL.
    assert cache.get(cam, now=110.0) is True


def test_get_after_ttl_returns_false() -> None:
    """A packet older than ttl_seconds is treated as missing. This is
    the "phone went quiet" semantic — assume idle, save LLM cost.
    """
    cache = MotionCache(ttl_seconds=15)
    cam = uuid4()
    cache.put(cam, in_motion=True, received_at=100.0)
    # 16s later — past the TTL.
    assert cache.get(cam, now=116.0) is False


def test_put_overwrites_previous_state_and_resets_ttl() -> None:
    cache = MotionCache(ttl_seconds=15)
    cam = uuid4()
    cache.put(cam, in_motion=True, received_at=100.0)
    # 10s later the phone reports no motion. Overwrite.
    cache.put(cam, in_motion=False, received_at=110.0)
    # Read 5s after the overwrite — still inside TTL of the new entry.
    assert cache.get(cam, now=115.0) is False
    # Read 20s after the overwrite — past the new entry's TTL.
    assert cache.get(cam, now=130.0) is False  # stale-hit → False


def test_drop_forgets_camera() -> None:
    cache = MotionCache(ttl_seconds=15)
    cam = uuid4()
    cache.put(cam, in_motion=True, received_at=100.0)
    assert cam in cache
    cache.drop(cam)
    assert cam not in cache
    assert cache.get(cam, now=100.0) is False


def test_drop_unknown_camera_is_noop() -> None:
    cache = MotionCache(ttl_seconds=15)
    cache.drop(uuid4())   # must not raise


def test_separate_cameras_have_independent_state() -> None:
    cache = MotionCache(ttl_seconds=15)
    cam_a, cam_b = uuid4(), uuid4()
    cache.put(cam_a, in_motion=True, received_at=100.0)
    cache.put(cam_b, in_motion=False, received_at=100.0)
    assert cache.get(cam_a, now=105.0) is True
    assert cache.get(cam_b, now=105.0) is False
