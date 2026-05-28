"""Unit tests for `DetectorWorker.stop_worker_for_camera`.

P2 entry point: the camera DELETE API handler calls this to tear down
the in-process detector worker at the same instant the cameras row is
deleted from the DB. The contract:

  - Find the CameraIngest by camera_id (dict is keyed by room_name,
    so we scan all rooms).
  - Call RoomWorker.remove_ingest outside the lock.
  - Never raise. Convert all failures to False.
  - Bound to `timeout` seconds.

We construct DetectorWorker directly with fake RoomWorkers (no
LiveKit, no DB).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from src.services.detection.worker import DetectorWorker


class _FakeIngest:
    def __init__(self, camera_id: UUID) -> None:
        self.camera_id = camera_id


class _FakeRoomWorker:
    """Stub RoomWorker that exposes the same ingest dict shape and a
    fake remove_ingest. Captures which identity was removed (or hangs
    / raises depending on configuration) so tests can assert behavior.
    """

    def __init__(
        self,
        *,
        ingests: dict[str, _FakeIngest] | None = None,
        hang_seconds: float = 0.0,
        raise_on_remove: Exception | None = None,
    ) -> None:
        self._ingests_by_identity: dict[str, _FakeIngest] = ingests or {}
        self.removed_identities: list[str] = []
        self._hang_seconds = hang_seconds
        self._raise_on_remove = raise_on_remove

    async def remove_ingest(self, identity: str) -> None:
        if self._hang_seconds > 0:
            await asyncio.sleep(self._hang_seconds)
        if self._raise_on_remove is not None:
            raise self._raise_on_remove
        self.removed_identities.append(identity)
        self._ingests_by_identity.pop(identity, None)


def _make_worker() -> DetectorWorker:
    settings = SimpleNamespace(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_motion_state_ttl_seconds=15,
    )
    # ReIDInference is constructed lazily inside DetectorWorker.__init__
    # via the import in `from src.services.reid.inference import ...`,
    # but it doesn't do I/O at construction — only on first call. So
    # passing the real settings (with the sentinel URL) is fine.
    return DetectorWorker(settings=settings, sessionmaker=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_stop_worker_for_camera_found_in_room() -> None:
    """Happy path: ingest with the right camera_id is found, removed,
    and the function returns True.
    """
    worker = _make_worker()
    camera_id = uuid4()
    fake_ingest = _FakeIngest(camera_id=camera_id)
    fake_room = _FakeRoomWorker(ingests={"cam_abcd1234": fake_ingest})
    worker._rooms["patient_xyz"] = fake_room  # type: ignore[assignment]

    result = await worker.stop_worker_for_camera(camera_id)

    assert result is True
    assert fake_room.removed_identities == ["cam_abcd1234"]


@pytest.mark.asyncio
async def test_stop_worker_for_camera_not_found_returns_false() -> None:
    """A camera_id with no matching ingest (paired but offline) must
    return False quickly without raising.
    """
    worker = _make_worker()
    # No rooms at all.
    result = await worker.stop_worker_for_camera(uuid4())
    assert result is False


@pytest.mark.asyncio
async def test_stop_worker_for_camera_multi_room_finds_right_one() -> None:
    """When three rooms each hold a different camera, removing one
    must affect only the matching room's ingest dict.
    """
    worker = _make_worker()
    target_cam = uuid4()
    other_cam_1 = uuid4()
    other_cam_2 = uuid4()

    room_a = _FakeRoomWorker(ingests={"cam_aaaa1111": _FakeIngest(camera_id=other_cam_1)})
    room_b = _FakeRoomWorker(ingests={"cam_bbbb2222": _FakeIngest(camera_id=target_cam)})
    room_c = _FakeRoomWorker(ingests={"cam_cccc3333": _FakeIngest(camera_id=other_cam_2)})
    worker._rooms["patient_a"] = room_a  # type: ignore[assignment]
    worker._rooms["patient_b"] = room_b  # type: ignore[assignment]
    worker._rooms["patient_c"] = room_c  # type: ignore[assignment]

    result = await worker.stop_worker_for_camera(target_cam)

    assert result is True
    assert room_a.removed_identities == []
    assert room_b.removed_identities == ["cam_bbbb2222"]
    assert room_c.removed_identities == []


@pytest.mark.asyncio
async def test_stop_worker_for_camera_timeout_returns_false() -> None:
    """If remove_ingest hangs, stop_worker_for_camera bails after
    `timeout` and returns False — never raises.
    """
    worker = _make_worker()
    camera_id = uuid4()
    fake_room = _FakeRoomWorker(
        ingests={"cam_hanger": _FakeIngest(camera_id=camera_id)},
        hang_seconds=10.0,
    )
    worker._rooms["patient_hang"] = fake_room  # type: ignore[assignment]

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    result = await worker.stop_worker_for_camera(camera_id, timeout=0.3)
    elapsed = loop.time() - t0

    assert result is False
    # Generous slack on the upper bound for scheduling jitter.
    assert elapsed < 1.0, f"took {elapsed:.2f}s; expected <1s"


@pytest.mark.asyncio
async def test_stop_worker_for_camera_swallows_remove_exception() -> None:
    """If remove_ingest raises, return False — don't propagate to the
    DELETE API handler.
    """
    worker = _make_worker()
    camera_id = uuid4()
    fake_room = _FakeRoomWorker(
        ingests={"cam_explode": _FakeIngest(camera_id=camera_id)},
        raise_on_remove=RuntimeError("simulated LiveKit disconnect failure"),
    )
    worker._rooms["patient_boom"] = fake_room  # type: ignore[assignment]

    result = await worker.stop_worker_for_camera(camera_id)

    assert result is False


@pytest.mark.asyncio
async def test_stop_worker_for_camera_empty_worker_returns_false() -> None:
    """Worker with no rooms at all (cold-boot / pre-webhook state) —
    no crash, returns False.
    """
    worker = _make_worker()
    result = await worker.stop_worker_for_camera(uuid4())
    assert result is False
