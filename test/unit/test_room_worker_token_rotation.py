"""Unit tests for RoomWorker's token-rotation safety: the periodic
recycle and the stuck-reconnect watchdog.

Both exist because the LiveKit Python SDK reuses the connect-time
JWT for the entire lifetime of the WebSocket. A long-lived backend
will eventually hit a network blip that forces an SDK resume, and
if the resume happens past TTL the SDK retries with the expired
token for ~18 min before giving up (observed 2026-05-29). These
loops prevent that:

  - `_recycle_loop` proactively disconnects every
    `_TOKEN_RECYCLE_INTERVAL_S`, which exits the supervisor's
    `_connect_and_wait` and forces a fresh-token reconnect.
  - `_reconnect_watchdog_loop` reactively disconnects if
    `connection_state_changed` reports `CONN_RECONNECTING` longer
    than `_RECONNECT_WATCHDOG_TIMEOUT_S`.

Tests exercise the loops in isolation against a fake room so we can
verify cancellation, error tolerance, and the watchdog's recovery /
re-arm semantics without touching real LiveKit.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from src.services.detection.motion_cache import MotionCache
from src.services.detection import room_worker as rw_module
from src.services.detection.room_worker import RoomWorker


class _FakeRoom:
    """Just enough of `rtc.Room` for the loops: a disconnect() that
    records every call. Optional `raises` simulates an SDK that
    throws (we must keep going)."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.disconnect_calls = 0
        self._raises = raises

    async def disconnect(self) -> None:
        self.disconnect_calls += 1
        if self._raises is not None:
            raise self._raises


def _make_room_worker() -> RoomWorker:
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
        room_name="patient_rotation_test",
        settings=settings,
        sessionmaker=None,
        motion_cache=motion_cache,
        reid=None,
        on_idle_timeout=None,
    )


# ── _recycle_loop ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recycle_loop_calls_disconnect_after_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recycle is the JWT-rotation primitive: after the configured
    interval elapses, it MUST fire disconnect exactly once so the
    supervisor's `_connect_and_wait` exits and mints a fresh token."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 0.05)
    rw = _make_room_worker()
    fake = _FakeRoom()

    task = asyncio.create_task(rw._recycle_loop(fake))
    await asyncio.sleep(0.15)
    # The recycle task is one-shot — after firing disconnect it returns.
    assert task.done()
    assert fake.disconnect_calls == 1


@pytest.mark.asyncio
async def test_recycle_loop_cancellation_before_interval_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the connect cycle ends naturally (disconnect from network or
    natural drop) before the recycle interval expires, the supervisor
    cancels the recycle task. It must exit without calling disconnect
    so we don't double-disconnect a room that's already gone."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 5.0)
    rw = _make_room_worker()
    fake = _FakeRoom()

    task = asyncio.create_task(rw._recycle_loop(fake))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.disconnect_calls == 0


@pytest.mark.asyncio
async def test_recycle_loop_swallows_disconnect_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `room.disconnect()` raises (SDK glitch, race with natural
    drop), the recycle must log and exit — not propagate, since the
    surrounding `_connect_and_wait` is about to clean up anyway."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 0.05)
    rw = _make_room_worker()
    fake = _FakeRoom(raises=RuntimeError("simulated SDK error"))

    task = asyncio.create_task(rw._recycle_loop(fake))
    await asyncio.sleep(0.15)
    assert task.done()
    # The loop returned normally despite the raise — no unhandled
    # exception escaped.
    assert task.exception() is None
    assert fake.disconnect_calls == 1


# ── _reconnect_watchdog_loop ───────────────────────────────────────


@pytest.mark.asyncio
async def test_watchdog_idle_when_not_reconnecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Steady-state: SDK is happily CONNECTED, watchdog ticks
    forever without ever calling disconnect."""
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.05)
    rw = _make_room_worker()
    fake = _FakeRoom()
    state: dict[str, float | None] = {"reconnecting_since": None}

    task = asyncio.create_task(rw._reconnect_watchdog_loop(fake, state))
    await asyncio.sleep(0.15)
    assert not task.done()
    assert fake.disconnect_calls == 0

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A brief network blip puts SDK into RECONNECTING but recovers
    in <threshold. Watchdog must not interfere with normal SDK
    resume — that path is healthier than a full reconnect."""
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.20)
    rw = _make_room_worker()
    fake = _FakeRoom()
    state: dict[str, float | None] = {"reconnecting_since": time.monotonic()}

    task = asyncio.create_task(rw._reconnect_watchdog_loop(fake, state))
    await asyncio.sleep(0.10)
    # Simulate SDK recovering: handler clears the timestamp.
    state["reconnecting_since"] = None
    await asyncio.sleep(0.15)

    assert not task.done()
    assert fake.disconnect_calls == 0

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_watchdog_fires_when_stuck_past_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The actual bug this prevents: SDK enters CONN_RECONNECTING and
    stays there because it's retrying with an expired token. Watchdog
    must force a disconnect so the supervisor can mint a fresh JWT."""
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.05)
    rw = _make_room_worker()
    fake = _FakeRoom()
    state: dict[str, float | None] = {"reconnecting_since": time.monotonic()}

    task = asyncio.create_task(rw._reconnect_watchdog_loop(fake, state))
    await asyncio.sleep(0.20)
    assert task.done()
    assert fake.disconnect_calls == 1


@pytest.mark.asyncio
async def test_watchdog_loop_reacts_to_current_dict_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Locks in the design contract: the loop reads `reconnecting_since`
    on every tick — it does NOT maintain any internal state about
    "have I already seen this timestamp." The handler (in
    `_connect_and_wait`) is the sole authority on timestamp truth.

    This matters because rearm-after-recovery is implemented in the
    HANDLER (it clears to None on `reconnected`, then sets to fresh
    `monotonic()` on the next `reconnecting`). The loop just trusts
    the dict. If we later moved rearm logic into the loop, this test
    would force us to redesign cleanly.

    Test: inject a stale timestamp (as if the handler set it long
    ago); loop should fire because elapsed >> threshold.
    """
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.10)
    rw = _make_room_worker()
    fake = _FakeRoom()
    state: dict[str, float | None] = {"reconnecting_since": time.monotonic() - 999}

    task = asyncio.create_task(rw._reconnect_watchdog_loop(fake, state))
    await asyncio.sleep(0.05)
    assert fake.disconnect_calls == 1
    assert task.done()


@pytest.mark.asyncio
async def test_watchdog_cancellation_is_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When `_connect_and_wait`'s finally block cancels the watchdog
    on natural disconnect, the loop must exit cleanly without firing
    a disconnect (the room is already gone)."""
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 1.0)
    rw = _make_room_worker()
    fake = _FakeRoom()
    state: dict[str, float | None] = {"reconnecting_since": None}

    task = asyncio.create_task(rw._reconnect_watchdog_loop(fake, state))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake.disconnect_calls == 0


@pytest.mark.asyncio
async def test_watchdog_swallows_disconnect_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If `room.disconnect()` raises during the watchdog's fire, the
    loop must log and exit normally — propagation would crash the
    surrounding _connect_and_wait's finally block."""
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.05)
    rw = _make_room_worker()
    fake = _FakeRoom(raises=RuntimeError("simulated SDK error"))
    state: dict[str, float | None] = {"reconnecting_since": time.monotonic()}

    task = asyncio.create_task(rw._reconnect_watchdog_loop(fake, state))
    await asyncio.sleep(0.20)
    assert task.done()
    assert task.exception() is None
    assert fake.disconnect_calls == 1


# ── Constants sanity ───────────────────────────────────────────────


def test_recycle_interval_well_inside_token_ttl() -> None:
    """If someone shortens TTL without also shortening the recycle,
    we'd be back in the 401-loop bug. Lock the invariant: recycle
    must complete at least one full cycle within TTL with comfortable
    headroom for the supervisor's reconnect backoff."""
    ttl_seconds = rw_module._WORKER_TOKEN_TTL.total_seconds()
    recycle = rw_module._TOKEN_RECYCLE_INTERVAL_S
    # Recycle <= TTL/2 means two recycles fit inside TTL, so a failed
    # recycle still has a full interval to retry before the token
    # expires.
    assert recycle <= ttl_seconds / 2, (
        f"recycle {recycle}s must be <= TTL {ttl_seconds}s / 2"
    )


def test_watchdog_timeout_shorter_than_sdk_giveup() -> None:
    """The SDK's internal resume gives up after ~18 minutes. Our
    watchdog must fire well before that or it adds no value."""
    assert rw_module._RECONNECT_WATCHDOG_TIMEOUT_S < 5 * 60
    assert rw_module._RECONNECT_WATCHDOG_TICK_S < rw_module._RECONNECT_WATCHDOG_TIMEOUT_S


# ── Integration: full _connect_and_wait flow with a fake Room ──────
#
# These tests verify that the recycle/watchdog tasks and the
# `reconnecting`/`reconnected` event handlers are actually wired
# correctly inside `_connect_and_wait` — i.e. firing real events
# on a fake Room propagates through to the watchdog's disconnect.
# The isolation tests above can't catch a missing `room.on(...)`
# registration; these can.


class _FakeRoomEmitter:
    """Just enough of `rtc.Room` for `_connect_and_wait`:

      - `.on(event_name)` decorator records handlers
      - `.fire(event_name, *args)` invokes all handlers for that
        event synchronously (matches the SDK's sync delivery)
      - `.connect(url, token)` is a no-op
      - `.disconnect()` fires the `disconnected` event (idempotent —
        only fires the first time, matching the real SDK's
        `if not self.isconnected(): return` early-out)
      - `.remote_participants` is an empty dict (backfill loop is
        a no-op)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self.connect_calls: list[tuple[str, str]] = []
        self.disconnect_calls = 0
        self._is_connected = False
        self.remote_participants: dict = {}

    def on(self, event_name: str):
        def _decorator(fn):
            self._handlers.setdefault(event_name, []).append(fn)
            return fn
        return _decorator

    def fire(self, event_name: str, *args) -> None:
        for handler in self._handlers.get(event_name, []):
            handler(*args)

    async def connect(self, url: str, token: str) -> None:
        self.connect_calls.append((url, token))
        self._is_connected = True

    async def disconnect(self) -> None:
        if not self._is_connected:
            return
        self.disconnect_calls += 1
        self._is_connected = False
        self.fire("disconnected", "client_initiated")


@pytest.fixture
def patched_room(monkeypatch: pytest.MonkeyPatch) -> _FakeRoomEmitter:
    """Monkeypatch `rtc.Room` so `_connect_and_wait` instantiates our
    fake. Returns the single instance the worker will see, so tests
    can fire events on it."""
    fake = _FakeRoomEmitter()
    monkeypatch.setattr(rw_module.rtc, "Room", lambda: fake)
    # `_wire_room_handlers` registers handlers we don't fire in these
    # tests — make sure the fake's .on accepts them all.
    return fake


@pytest.mark.asyncio
async def test_connect_and_wait_recycle_fires_disconnect_and_returns(
    monkeypatch: pytest.MonkeyPatch, patched_room: _FakeRoomEmitter,
) -> None:
    """End-to-end recycle: spawn `_connect_and_wait`, wait past the
    recycle interval, verify the recycle task called disconnect and
    `_connect_and_wait` returned with the SDK's disconnect reason."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 0.05)
    # Long watchdog so it doesn't interfere with the recycle path.
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 60.0)
    rw = _make_room_worker()

    task = asyncio.create_task(rw._connect_and_wait())
    reason = await asyncio.wait_for(task, timeout=1.0)

    assert patched_room.connect_calls, "connect was not called"
    assert patched_room.disconnect_calls == 1, "recycle did not call disconnect"
    assert reason == "client_initiated"


@pytest.mark.asyncio
async def test_connect_and_wait_watchdog_fires_on_stuck_reconnecting(
    monkeypatch: pytest.MonkeyPatch, patched_room: _FakeRoomEmitter,
) -> None:
    """End-to-end watchdog: spawn `_connect_and_wait`, fire the SDK
    `reconnecting` event, wait past the watchdog timeout WITHOUT
    firing `reconnected`. The watchdog must call disconnect and
    `_connect_and_wait` must return."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 60.0)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.05)
    rw = _make_room_worker()

    task = asyncio.create_task(rw._connect_and_wait())
    # Wait for connect to land; then fire the SDK's "I'm trying to
    # reconnect" event. Real SDK fires this when the WebSocket drops
    # and it enters its internal resume loop.
    await asyncio.sleep(0.01)
    patched_room.fire("reconnecting")

    reason = await asyncio.wait_for(task, timeout=1.0)
    assert patched_room.disconnect_calls == 1, "watchdog did not call disconnect"
    assert reason == "client_initiated"


@pytest.mark.asyncio
async def test_connect_and_wait_reconnected_event_clears_watchdog(
    monkeypatch: pytest.MonkeyPatch, patched_room: _FakeRoomEmitter,
) -> None:
    """Normal SDK resume: `reconnecting` fires, then `reconnected`
    fires within the threshold. Watchdog must NOT trigger; the worker
    stays in `_connect_and_wait`."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 60.0)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.20)
    rw = _make_room_worker()

    task = asyncio.create_task(rw._connect_and_wait())
    await asyncio.sleep(0.01)
    patched_room.fire("reconnecting")
    await asyncio.sleep(0.05)  # < threshold
    patched_room.fire("reconnected")
    # Wait past where the watchdog would have fired had we not cleared.
    await asyncio.sleep(0.30)

    assert not task.done(), "watchdog fired despite reconnected event"
    assert patched_room.disconnect_calls == 0

    # Cleanup: tear down the worker so the test exits cleanly.
    await patched_room.disconnect()  # emits `disconnected`
    await asyncio.wait_for(task, timeout=1.0)


@pytest.mark.asyncio
async def test_connect_and_wait_handler_rearms_after_recovery(
    monkeypatch: pytest.MonkeyPatch, patched_room: _FakeRoomEmitter,
) -> None:
    """Flap-then-stick: `reconnecting`/`reconnected`/`reconnecting`
    pattern. The watchdog must measure from the SECOND `reconnecting`,
    not the first — otherwise it would fire instantly. This is the
    REAL test for the handler's rearm behavior."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 60.0)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 0.10)
    rw = _make_room_worker()

    task = asyncio.create_task(rw._connect_and_wait())
    await asyncio.sleep(0.01)

    # First flap, fully recovered.
    patched_room.fire("reconnecting")
    await asyncio.sleep(0.03)
    patched_room.fire("reconnected")
    await asyncio.sleep(0.02)

    # Second entry into reconnecting — handler must SET a fresh
    # timestamp. Without rearm, the watchdog would think we've been
    # reconnecting for ~50ms already and fire on the very next tick.
    patched_room.fire("reconnecting")
    # Wait HALF the threshold — if rearm broken, watchdog already
    # fired by now.
    await asyncio.sleep(0.05)
    assert not task.done(), "watchdog fired too early — rearm broken"
    assert patched_room.disconnect_calls == 0

    # Now wait past the threshold from the SECOND reconnecting.
    await asyncio.sleep(0.10)
    reason = await asyncio.wait_for(task, timeout=1.0)
    assert patched_room.disconnect_calls == 1
    assert reason == "client_initiated"


@pytest.mark.asyncio
async def test_connect_and_wait_natural_disconnect_cancels_recycle_and_watchdog(
    monkeypatch: pytest.MonkeyPatch, patched_room: _FakeRoomEmitter,
) -> None:
    """Cleanup contract: when `_connect_and_wait` exits via a natural
    disconnect (not recycle, not watchdog), the finally block must
    cancel both background tasks so they don't leak across cycles."""
    monkeypatch.setattr(rw_module, "_TOKEN_RECYCLE_INTERVAL_S", 60.0)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TICK_S", 0.01)
    monkeypatch.setattr(rw_module, "_RECONNECT_WATCHDOG_TIMEOUT_S", 60.0)
    rw = _make_room_worker()

    task = asyncio.create_task(rw._connect_and_wait())
    await asyncio.sleep(0.02)
    # Natural disconnect from the SDK side.
    patched_room.fire("disconnected", "network_drop")

    reason = await asyncio.wait_for(task, timeout=1.0)
    assert reason == "network_drop"
    # Recycle never fired its disconnect (task was cancelled before
    # its 60s sleep elapsed).
    assert patched_room.disconnect_calls == 0
    # All asyncio tasks created inside `_connect_and_wait` should be
    # done (cancelled cleanly by the finally block). Give the event
    # loop one tick to settle cancellations.
    await asyncio.sleep(0.05)
    pending = [t for t in asyncio.all_tasks() if not t.done() and t is not asyncio.current_task()]
    leaked = [t.get_name() for t in pending if t.get_name().startswith("room-")]
    assert leaked == [], f"leaked tasks: {leaked}"
