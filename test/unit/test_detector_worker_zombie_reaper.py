"""Unit tests for the startup zombie reaper.

When the backend starts, `_reconcile_active_rooms` lists every active
patient_* room and (per Cleanup #1) kicks any leftover
`detector_worker_*` participants — they can only be zombies from a
prior process that didn't shut down cleanly (SIGKILL, crash, OOM
kill). This is a hygiene fix; the new worker we spawn after reaping
has a fresh random identity suffix and would not collide regardless,
but the zombie burns a LiveKit identity slot until LiveKit's ~5min
server-side timeout fires.

These tests cover the reaper in isolation:
  - reaps every detector_worker_* found in patient_* rooms,
  - leaves cam_* and caregiver_* participants alone,
  - is robust to per-room list_participants failures,
  - is robust to per-participant remove failures,
  - does nothing on a fresh deployment with no rooms.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.services.detection.worker import DetectorWorker


# ── fakes for the LiveKit SDK surface we touch ──────────────────────


class _FakeParticipant:
    def __init__(self, identity: str) -> None:
        self.identity = identity


class _FakeRoom:
    def __init__(self, name: str, num_participants: int = 1) -> None:
        self.name = name
        self.num_participants = num_participants


class _FakeRoomService:
    """Records every remove_participant call so tests can assert
    exactly which identities were kicked.
    """

    def __init__(
        self,
        *,
        rooms: list[_FakeRoom],
        participants_by_room: dict[str, list[_FakeParticipant]],
        list_participants_exc_for: set[str] | None = None,
        remove_exc_for: set[str] | None = None,
    ) -> None:
        self._rooms = rooms
        self._parts = participants_by_room
        self._list_exc_for = list_participants_exc_for or set()
        self._remove_exc_for = remove_exc_for or set()
        self.removed: list[tuple[str, str]] = []

    async def list_rooms(self, req):  # noqa: ANN001
        return SimpleNamespace(rooms=list(self._rooms))

    async def list_participants(self, req):  # noqa: ANN001
        if req.room in self._list_exc_for:
            raise RuntimeError(f"simulated list_participants failure: {req.room}")
        return SimpleNamespace(participants=list(self._parts.get(req.room, [])))

    async def remove_participant(self, req):  # noqa: ANN001
        if req.identity in self._remove_exc_for:
            raise RuntimeError(f"simulated remove failure: {req.identity}")
        self.removed.append((req.room, req.identity))


class _FakeLiveKitAPI:
    def __init__(self, room_service: _FakeRoomService) -> None:
        self.room = room_service

    async def aclose(self) -> None:
        pass


def _install_fake_lkapi(
    monkeypatch: pytest.MonkeyPatch, room_service: _FakeRoomService,
) -> _FakeLiveKitAPI:
    fake = _FakeLiveKitAPI(room_service)
    monkeypatch.setattr(
        "src.services.detection.worker.lkapi.LiveKitAPI",
        lambda url, key, secret: fake,
    )
    return fake


def _make_worker() -> DetectorWorker:
    settings = SimpleNamespace(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_motion_state_ttl_seconds=15,
    )
    return DetectorWorker(settings=settings, sessionmaker=None)  # type: ignore[arg-type]


def _fake_owned(rooms: set[str]):
    """Async stub for `_owned_room_names` so tests don't need a real
    cameras table. The reaper test set always wants the candidate
    rooms in scope.
    """
    async def _owned() -> set[str]:
        return set(rooms)
    return _owned


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reaper_kicks_only_detector_worker_participants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A room with cam_, detector_worker_, caregiver_ participants
    must have ONLY the detector_worker_ identities kicked. The
    publisher and caregiver bots are not zombies — they belong to
    real wall phones / web sessions.
    """
    rooms = [_FakeRoom("patient_abc", num_participants=3)]
    parts = {
        "patient_abc": [
            _FakeParticipant("cam_5ee0eda4"),
            _FakeParticipant("detector_worker_abc_zombie1"),
            _FakeParticipant("caregiver_4e22fd5b_web"),
        ],
    }
    rs = _FakeRoomService(rooms=rooms, participants_by_room=parts)
    _install_fake_lkapi(monkeypatch, rs)

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_abc"})  # type: ignore[assignment]

    async def noop_join(room_name: str) -> None:
        pass
    worker.ensure_room_joined = noop_join  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert rs.removed == [("patient_abc", "detector_worker_abc_zombie1")]


@pytest.mark.asyncio
async def test_reaper_kicks_multiple_zombies_in_one_room(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact symptom we saw in production: a patient_* room with
    two detector_worker_* identities (one from yesterday's backend,
    one from today's). Both must be kicked.
    """
    rooms = [_FakeRoom("patient_601fbbc5", num_participants=2)]
    parts = {
        "patient_601fbbc5": [
            _FakeParticipant("detector_worker_601fbbc5_7a9106"),
            _FakeParticipant("detector_worker_601fbbc5_940954"),
        ],
    }
    rs = _FakeRoomService(rooms=rooms, participants_by_room=parts)
    _install_fake_lkapi(monkeypatch, rs)

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_601fbbc5"})  # type: ignore[assignment]
    worker.ensure_room_joined = (lambda n: asyncio.sleep(0))  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    kicked = {ident for _, ident in rs.removed}
    assert kicked == {
        "detector_worker_601fbbc5_7a9106",
        "detector_worker_601fbbc5_940954",
    }


@pytest.mark.asyncio
async def test_reaper_does_nothing_on_fresh_deploy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-time deployment: zero active rooms. Reaper must run
    cleanly without errors.
    """
    rs = _FakeRoomService(rooms=[], participants_by_room={})
    _install_fake_lkapi(monkeypatch, rs)

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned(set())  # type: ignore[assignment]
    worker.ensure_room_joined = (lambda n: asyncio.sleep(0))  # type: ignore[assignment]

    await worker._reconcile_active_rooms()
    assert rs.removed == []


@pytest.mark.asyncio
async def test_reaper_continues_past_list_participants_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If list_participants fails for one room, the reaper logs and
    continues with the others. Total reap count reflects only the
    rooms that succeeded.
    """
    rooms = [
        _FakeRoom("patient_a", num_participants=1),
        _FakeRoom("patient_b", num_participants=1),
    ]
    parts = {
        "patient_a": [_FakeParticipant("detector_worker_a_ghost")],
        "patient_b": [_FakeParticipant("detector_worker_b_ghost")],
    }
    rs = _FakeRoomService(
        rooms=rooms,
        participants_by_room=parts,
        list_participants_exc_for={"patient_a"},
    )
    _install_fake_lkapi(monkeypatch, rs)

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_a", "patient_b"})  # type: ignore[assignment]
    worker.ensure_room_joined = (lambda n: asyncio.sleep(0))  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert rs.removed == [("patient_b", "detector_worker_b_ghost")]


@pytest.mark.asyncio
async def test_reaper_continues_past_remove_participant_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single remove_participant failure doesn't abort the loop.
    Other zombies in the same room AND other rooms are still kicked.
    """
    rooms = [
        _FakeRoom("patient_a", num_participants=2),
        _FakeRoom("patient_b", num_participants=1),
    ]
    parts = {
        "patient_a": [
            _FakeParticipant("detector_worker_a_stubborn"),
            _FakeParticipant("detector_worker_a_easy"),
        ],
        "patient_b": [_FakeParticipant("detector_worker_b_easy")],
    }
    rs = _FakeRoomService(
        rooms=rooms,
        participants_by_room=parts,
        remove_exc_for={"detector_worker_a_stubborn"},
    )
    _install_fake_lkapi(monkeypatch, rs)

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_a", "patient_b"})  # type: ignore[assignment]
    worker.ensure_room_joined = (lambda n: asyncio.sleep(0))  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert sorted(ident for _, ident in rs.removed) == [
        "detector_worker_a_easy", "detector_worker_b_easy",
    ]


@pytest.mark.asyncio
async def test_reaper_never_kicks_workers_in_foreign_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LiveKit room whose patient UUID is NOT in our cameras table
    belongs to a different backend instance (coworker pointed at the
    same LiveKit project, etc). Kicking workers there would disrupt
    the OTHER backend's perfectly-healthy detector. We must skip
    those rooms entirely — neither join them nor reap them.
    """
    rooms = [
        _FakeRoom("patient_mine_aaaa", num_participants=1),
        _FakeRoom("patient_partners_bbbb", num_participants=1),
    ]
    parts = {
        "patient_mine_aaaa": [_FakeParticipant("detector_worker_mine_ghost")],
        "patient_partners_bbbb": [_FakeParticipant("detector_worker_partners_live")],
    }
    rs = _FakeRoomService(rooms=rooms, participants_by_room=parts)
    _install_fake_lkapi(monkeypatch, rs)

    worker = _make_worker()
    worker._started = True
    # Only patient_mine_aaaa is ours. patient_partners_bbbb is foreign.
    worker._owned_room_names = _fake_owned({"patient_mine_aaaa"})  # type: ignore[assignment]
    worker.ensure_room_joined = (lambda n: asyncio.sleep(0))  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    # The partner's worker MUST be untouched.
    assert ("patient_partners_bbbb", "detector_worker_partners_live") not in rs.removed
    # Our zombie IS reaped.
    assert ("patient_mine_aaaa", "detector_worker_mine_ghost") in rs.removed
