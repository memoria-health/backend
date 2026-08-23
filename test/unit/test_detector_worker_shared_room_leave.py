"""One camera leaving must not blind the others sharing its room.

`ensure_room_left_if_empty` is called from the `participant_left`
webhook for any `cam_*` identity, and it tore the room worker down
unconditionally — it never checked whether the room was, in fact, empty.

That was correct when a room held one camera. Sensors are account-scoped
now (2026-08-22), so `account_<caregiver>` routinely holds several, and
the rule became: unplug one sensor, lose detection on every other sensor
in that home.

Reproduced live on 2026-08-23. Disconnecting a bridge camera stopped the
shared room's worker while a gateway camera was still publishing; the
surviving camera's last analysis was written four seconds later and none
followed. The camera stayed `online` in the app the whole time, so
nothing surfaced it. `stop()` then hung to its 10s timeout and the
reference was dropped, so nothing rebuilt the worker either.

The same teardown is what makes the LiveKit FFI panic: cancelling the
supervisor while `room.connect()` sits between its websocket handshake
and `ReadyForRoomEventRequest` leaves the Rust side waiting on a request
that will never come, and it aborts the process 15s later. Not tested
here — that needs a real SDK — but it is the same call, and not making
it is what closes the window.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.services.detection.worker import DetectorWorker

_ROOM = "account_a30c505e-23b8-4efd-8d41-acc17c6b8879"


class _FakeRoomWorker:
    def __init__(self, publishers: set[str]) -> None:
        self.stop_called = False
        self._publishers = publishers

    def live_camera_identities(self) -> set[str]:
        return set(self._publishers)

    async def stop(self) -> None:
        self.stop_called = True


def _make_worker() -> DetectorWorker:
    settings = SimpleNamespace(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_motion_state_ttl_seconds=15,
        identity_state_url="http://idsvc:1",
        identity_state_token="",
        reid_trace_enabled=False,
    )
    worker = DetectorWorker(settings=settings, sessionmaker=None)  # type: ignore[arg-type]
    worker._started = True
    return worker


@pytest.mark.asyncio
async def test_a_sibling_still_publishing_keeps_the_worker() -> None:
    """THE regression. Two sensors on one account, one leaves."""
    worker = _make_worker()
    fake = _FakeRoomWorker({"cam_650ca94d"})   # the Tapo, still live
    worker._rooms[_ROOM] = fake                # type: ignore[assignment]

    await worker.ensure_room_left_if_empty(_ROOM, departed="cam_c51733e6")

    assert fake.stop_called is False, (
        "tore down the shared room worker while another camera was still "
        "publishing — that camera silently stops being analysed"
    )
    assert _ROOM in worker._rooms, "the surviving camera lost its worker"


@pytest.mark.asyncio
async def test_the_last_camera_leaving_still_stops_the_worker() -> None:
    """The behaviour we must keep: an empty room costs nothing to hold
    open, but holding it forever leaks a LiveKit participant per home."""
    worker = _make_worker()
    fake = _FakeRoomWorker(set())              # nobody left publishing
    worker._rooms[_ROOM] = fake                # type: ignore[assignment]

    await worker.ensure_room_left_if_empty(_ROOM, departed="cam_c51733e6")

    assert fake.stop_called is True
    assert _ROOM not in worker._rooms


@pytest.mark.asyncio
async def test_the_departing_camera_does_not_count_as_a_survivor() -> None:
    """The webhook can arrive before the SDK has pruned the participant,
    so the leaver may still be in the worker's own view. Counting it
    would keep every room open forever — the opposite failure, and a
    quieter one."""
    worker = _make_worker()
    fake = _FakeRoomWorker({"cam_c51733e6"})   # only the one that left
    worker._rooms[_ROOM] = fake                # type: ignore[assignment]

    await worker.ensure_room_left_if_empty(_ROOM, departed="cam_c51733e6")

    assert fake.stop_called is True
    assert _ROOM not in worker._rooms


@pytest.mark.asyncio
async def test_an_unknown_room_is_still_a_no_op() -> None:
    worker = _make_worker()
    await worker.ensure_room_left_if_empty("account_nobody", departed="cam_x")
    assert worker._rooms == {}
