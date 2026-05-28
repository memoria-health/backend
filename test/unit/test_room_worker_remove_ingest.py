"""Unit tests for `RoomWorker.remove_ingest`.

remove_ingest is the back-channel that lets CameraIngest signal "my
camera was deleted from the DB, please remove me" (P1) AND the entry
point that the camera DELETE API handler uses to explicitly tear
down a paired camera's worker (P2).

These tests pin three invariants:
  - Idempotent removal (called twice or for an unknown identity, no
    crash; the second call is a logged no-op).
  - Symmetric with `_handle_participant_left`: pops from dict,
    cancels video + audio consumer tasks, awaits ingest.stop() with
    a bound.
  - Bounded against a hung ingest.stop() so the caller (self-destruct
    or DELETE handler) is never parked indefinitely.

We don't spin up a real LiveKit connection here — RoomWorker is
constructed directly and its dicts populated with fakes.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.detection.motion_cache import MotionCache
from src.services.detection.room_worker import RoomWorker


class _FakeIngest:
    """Stand-in for CameraIngest. Records whether stop() was called
    so the assertions can verify the lifecycle.
    """

    def __init__(self, *, hang_seconds: float = 0.0) -> None:
        self.camera_id = uuid4()
        self.stopped = False
        self._hang_seconds = hang_seconds

    async def stop(self) -> None:
        if self._hang_seconds > 0:
            await asyncio.sleep(self._hang_seconds)
        self.stopped = True


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
        room_name="patient_test123",
        settings=settings,
        sessionmaker=None,  # never used by remove_ingest path
        motion_cache=motion_cache,
        reid=None,  # never used by remove_ingest path
    )


@pytest.mark.asyncio
async def test_remove_ingest_pops_dict_and_stops_ingest() -> None:
    """The base happy path. After remove_ingest returns:
      - the identity is no longer in _ingests_by_identity
      - ingest.stop() was awaited (assert via the fake's flag)
    """
    rw = _make_room_worker()
    fake = _FakeIngest()
    rw._ingests_by_identity["cam_abcdef12"] = fake

    await rw.remove_ingest("cam_abcdef12")

    assert "cam_abcdef12" not in rw._ingests_by_identity
    assert fake.stopped is True


@pytest.mark.asyncio
async def test_remove_ingest_idempotent_when_identity_unknown() -> None:
    """Calling remove_ingest for an identity that isn't present (e.g.
    because participant_left already removed it) must NOT raise. This
    happens in production whenever the camera physically disconnects
    around the same moment the admin deletes its row.
    """
    rw = _make_room_worker()
    # Dict starts empty — no setup needed.
    # Should complete cleanly with no exception.
    await rw.remove_ingest("cam_nonexistent")
    # And a second call is still safe.
    await rw.remove_ingest("cam_nonexistent")


@pytest.mark.asyncio
async def test_remove_ingest_cancels_video_and_audio_consumer_tasks() -> None:
    """Symmetric with _handle_participant_left: when an ingest is
    removed, its video + audio consumer tasks must be cancelled too
    so they don't keep pushing frames into a stopped ingest's
    buffers.
    """
    rw = _make_room_worker()
    identity = "cam_abcdef12"
    fake = _FakeIngest()
    rw._ingests_by_identity[identity] = fake

    cancelled = {"video": False, "audio": False}

    async def video_consumer() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled["video"] = True
            raise

    async def audio_consumer() -> None:
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled["audio"] = True
            raise

    rw._video_tasks_by_identity[identity] = asyncio.create_task(video_consumer())
    rw._audio_tasks_by_identity[identity] = asyncio.create_task(audio_consumer())
    # Yield so the tasks enter their try block before we cancel them —
    # otherwise the cancel hits a never-started coroutine and the
    # CancelledError handler in the consumer never runs, so the flags
    # don't get set even though the cancel itself is fine.
    await asyncio.sleep(0)

    await rw.remove_ingest(identity)

    assert cancelled["video"] is True
    assert cancelled["audio"] is True
    assert identity not in rw._video_tasks_by_identity
    assert identity not in rw._audio_tasks_by_identity


@pytest.mark.asyncio
async def test_remove_ingest_bounds_stop_at_five_seconds() -> None:
    """If ingest.stop() hangs (e.g. mid-Gemini call that ignores its
    cancel), remove_ingest must return within ~5s rather than parking
    the caller forever.
    """
    rw = _make_room_worker()
    fake = _FakeIngest(hang_seconds=30.0)
    rw._ingests_by_identity["cam_hung"] = fake

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await rw.remove_ingest("cam_hung")
    elapsed = loop.time() - t0

    # 5s timeout in the code; allow 1s slack for scheduling.
    assert elapsed < 6.0, (
        f"remove_ingest took {elapsed:.2f}s; expected <6.0s"
    )
    # Either way, the identity is gone from the dict — the timeout
    # path treats us as best-effort.
    assert "cam_hung" not in rw._ingests_by_identity
