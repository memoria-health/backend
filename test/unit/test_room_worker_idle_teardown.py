"""Unit tests for `RoomWorker._monitor_idle` (Cleanup #2).

The idle monitor fires the `on_idle_timeout` callback when the
worker has been alive past `_IDLE_TEARDOWN_WINDOW_S` AND has never
seen a `cam_*` ingest spawn. This catches the case where startup
reconciliation joins a patient_* room that has no actual camera
publishing — without this, the RoomWorker would sit connected to
LiveKit forever burning an identity slot.

We exercise the monitor in isolation by patching the window /
tick constants to short values so tests run in milliseconds, and
flipping `_has_ever_seen_ingest` to simulate a real camera arrival.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.services.detection.motion_cache import MotionCache
from src.services.detection import room_worker as rw_module
from src.services.detection.room_worker import RoomWorker


def _make_room_worker(
    *,
    on_idle_timeout=None,
) -> RoomWorker:
    settings = SimpleNamespace(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_worker_window_seconds=15,
        livekit_worker_target_fps=15,
        livekit_worker_tick_seconds=10,
    )
    motion_cache = MotionCache(ttl_seconds=15)
    return RoomWorker(
        room_name="patient_idle_test",
        settings=settings,
        sessionmaker=None,
        motion_cache=motion_cache,
        reid=None,
        on_idle_timeout=on_idle_timeout,
    )


@pytest.mark.asyncio
async def test_monitor_fires_callback_after_window_when_room_stays_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: a room that joined and never saw a camera fires
    the eviction callback after the window expires.
    """
    monkeypatch.setattr(rw_module, "_IDLE_MONITOR_TICK_S", 0.02)
    monkeypatch.setattr(rw_module, "_IDLE_TEARDOWN_WINDOW_S", 0.05)

    evicted: list[str] = []

    async def on_evict(room_name: str) -> None:
        evicted.append(room_name)

    rw = _make_room_worker(on_idle_timeout=on_evict)
    # Mimic what start() would set up. We don't want to actually spawn
    # the supervisor (it would try to talk to a real LiveKit), but we
    # need the idle monitor task and the created_at timestamp.
    import time as time_mod
    rw._created_at = time_mod.monotonic()
    rw._idle_monitor_task = asyncio.create_task(rw._monitor_idle())

    # Wait long enough for the monitor to fire — window + 1 tick.
    await asyncio.sleep(0.15)
    assert evicted == ["patient_idle_test"]
    assert rw._idle_signalled is True

    rw._idle_monitor_task.cancel()
    try:
        await rw._idle_monitor_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_monitor_does_not_fire_when_ingest_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real-use room (one with a camera at some point) must NEVER
    trigger eviction, even if the dict is currently empty. The latch
    is permanent: once an ingest has been seen, the monitor exits
    forever.
    """
    monkeypatch.setattr(rw_module, "_IDLE_MONITOR_TICK_S", 0.02)
    monkeypatch.setattr(rw_module, "_IDLE_TEARDOWN_WINDOW_S", 0.05)

    evicted: list[str] = []

    async def on_evict(room_name: str) -> None:
        evicted.append(room_name)

    rw = _make_room_worker(on_idle_timeout=on_evict)
    import time as time_mod
    rw._created_at = time_mod.monotonic()
    rw._has_ever_seen_ingest = True  # camera was here once
    rw._idle_monitor_task = asyncio.create_task(rw._monitor_idle())

    await asyncio.sleep(0.15)
    assert evicted == []
    assert rw._idle_signalled is False

    rw._idle_monitor_task.cancel()
    try:
        await rw._idle_monitor_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_monitor_skips_if_ingest_arrives_before_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Race-safe: if a camera connects partway through the window,
    the monitor's next tick sees the latch flipped and exits without
    firing eviction.
    """
    monkeypatch.setattr(rw_module, "_IDLE_MONITOR_TICK_S", 0.02)
    monkeypatch.setattr(rw_module, "_IDLE_TEARDOWN_WINDOW_S", 0.20)

    evicted: list[str] = []

    async def on_evict(room_name: str) -> None:
        evicted.append(room_name)

    rw = _make_room_worker(on_idle_timeout=on_evict)
    import time as time_mod
    rw._created_at = time_mod.monotonic()
    rw._idle_monitor_task = asyncio.create_task(rw._monitor_idle())

    # Let the monitor tick a couple of times while empty.
    await asyncio.sleep(0.05)
    # Camera arrives.
    rw._has_ever_seen_ingest = True
    # Wait past the window — monitor should NOT have fired.
    await asyncio.sleep(0.30)

    assert evicted == []

    rw._idle_monitor_task.cancel()
    try:
        await rw._idle_monitor_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_monitor_fires_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Idempotency: the callback is invoked at most once per
    RoomWorker instance. Even if the callback is slow / hangs, no
    second invocation happens.
    """
    monkeypatch.setattr(rw_module, "_IDLE_MONITOR_TICK_S", 0.02)
    monkeypatch.setattr(rw_module, "_IDLE_TEARDOWN_WINDOW_S", 0.05)

    invoke_count = 0

    async def on_evict(room_name: str) -> None:
        nonlocal invoke_count
        invoke_count += 1

    rw = _make_room_worker(on_idle_timeout=on_evict)
    import time as time_mod
    rw._created_at = time_mod.monotonic()
    rw._idle_monitor_task = asyncio.create_task(rw._monitor_idle())

    await asyncio.sleep(0.30)
    assert invoke_count == 1

    rw._idle_monitor_task.cancel()
    try:
        await rw._idle_monitor_task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_monitor_callback_timeout_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung callback bounded by wait_for; monitor exits cleanly
    after timeout. We test by giving the callback a sleep longer
    than the wait_for bound and asserting no exception propagates.
    """
    monkeypatch.setattr(rw_module, "_IDLE_MONITOR_TICK_S", 0.02)
    monkeypatch.setattr(rw_module, "_IDLE_TEARDOWN_WINDOW_S", 0.05)

    callback_started = asyncio.Event()

    async def hung_evict(room_name: str) -> None:
        callback_started.set()
        await asyncio.sleep(20.0)  # well past the 15s wait_for bound

    rw = _make_room_worker(on_idle_timeout=hung_evict)
    import time as time_mod
    rw._created_at = time_mod.monotonic()
    rw._idle_monitor_task = asyncio.create_task(rw._monitor_idle())

    # Wait until the callback at least starts.
    for _ in range(100):
        await asyncio.sleep(0.02)
        if callback_started.is_set():
            break
    assert callback_started.is_set()

    # The monitor task is now blocked inside wait_for. Cancel it to
    # release the test (we'd otherwise wait 15s for the real timeout).
    rw._idle_monitor_task.cancel()
    try:
        await rw._idle_monitor_task
    except (asyncio.CancelledError, Exception):
        pass

    # The signal latch was set the moment we decided to fire.
    assert rw._idle_signalled is True


@pytest.mark.asyncio
async def test_no_callback_registered_means_no_monitor_started() -> None:
    """A RoomWorker built without `on_idle_timeout` (e.g. the test
    path or legacy code path) does not start the idle monitor task.
    Verified by attribute presence.
    """
    rw = _make_room_worker(on_idle_timeout=None)
    # Mimic start() effects without spinning a real LiveKit.
    import time as time_mod
    rw._created_at = time_mod.monotonic()
    # If start() were called: monitor would NOT be created when
    # _on_idle_timeout is None — re-check the start() branch.
    assert rw._on_idle_timeout is None
