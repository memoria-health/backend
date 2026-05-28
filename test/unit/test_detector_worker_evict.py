"""Unit tests for `DetectorWorker._evict_room` (Cleanup #2 — the
parent-side handler that idle RoomWorkers call back into).

`_evict_room` is symmetric to `ensure_room_left_if_empty` but is
triggered by the worker's idle monitor (an in-process timer) rather
than a LiveKit webhook. The contract is identical: pop from the
dict, stop the worker with a timeout, never raise.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.services.detection.worker import DetectorWorker


class _FakeRoomWorker:
    def __init__(self, *, hang_seconds: float = 0.0, raise_exc: Exception | None = None) -> None:
        self.stop_called = False
        self._hang = hang_seconds
        self._exc = raise_exc

    async def stop(self) -> None:
        if self._hang > 0:
            await asyncio.sleep(self._hang)
        if self._exc is not None:
            raise self._exc
        self.stop_called = True


def _make_worker() -> DetectorWorker:
    settings = SimpleNamespace(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_motion_state_ttl_seconds=15,
    )
    return DetectorWorker(settings=settings, sessionmaker=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_evict_room_pops_and_stops() -> None:
    """Happy path: the room is in the dict, _evict_room pops it and
    awaits stop().
    """
    worker = _make_worker()
    fake = _FakeRoomWorker()
    worker._rooms["patient_idle"] = fake  # type: ignore[assignment]

    await worker._evict_room("patient_idle")

    assert "patient_idle" not in worker._rooms
    assert fake.stop_called is True


@pytest.mark.asyncio
async def test_evict_room_is_idempotent_when_unknown() -> None:
    """Calling _evict_room for a room that isn't in the dict (e.g.
    because participant_left already removed it, or a concurrent
    eviction won) must be a clean no-op.
    """
    worker = _make_worker()
    # Dict is empty.
    await worker._evict_room("patient_nonexistent")
    # Idempotent — second call also fine.
    await worker._evict_room("patient_nonexistent")


@pytest.mark.asyncio
async def test_evict_room_swallows_stop_timeout() -> None:
    """A RoomWorker.stop() that hangs (e.g. LiveKit disconnect
    deadlock) must NOT propagate. _evict_room bounds it via the
    same _STOP_TIMEOUT used elsewhere; on timeout, the dict entry is
    already gone (popped before stop), so the LiveKit participant
    will eventually time out server-side.
    """
    worker = _make_worker()
    # Set _STOP_TIMEOUT-equivalent via short-hang to keep the test fast.
    fake = _FakeRoomWorker(hang_seconds=30.0)
    worker._rooms["patient_hung"] = fake  # type: ignore[assignment]

    # Patch the timeout constant to a fraction of a second so the
    # test runs in milliseconds instead of 10s.
    from src.services.detection import worker as worker_mod
    orig_timeout = worker_mod._STOP_TIMEOUT
    worker_mod._STOP_TIMEOUT = 0.1
    try:
        await worker._evict_room("patient_hung")
    finally:
        worker_mod._STOP_TIMEOUT = orig_timeout

    assert "patient_hung" not in worker._rooms


@pytest.mark.asyncio
async def test_evict_room_swallows_stop_exception() -> None:
    """If RoomWorker.stop() raises, _evict_room logs and continues.
    No exception propagates to the caller (the RoomWorker's idle
    monitor task).
    """
    worker = _make_worker()
    fake = _FakeRoomWorker(raise_exc=RuntimeError("simulated stop failure"))
    worker._rooms["patient_explode"] = fake  # type: ignore[assignment]

    await worker._evict_room("patient_explode")

    assert "patient_explode" not in worker._rooms
