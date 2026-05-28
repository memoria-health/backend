"""Unit tests for `DetectorWorker._reconcile_active_rooms`.

P3: at backend startup, the worker queries LiveKit for active rooms
and spawns workers for any whose participant_joined webhook we missed
(e.g. backend was redeploying while the camera was already connected).

The whole point of this code is "fail safe under any LiveKit failure":
crash, timeout, unreachable, malformed response, partial failures —
none of them should crash backend startup or stop the webhook path
from working for new connects.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.services.detection.worker import DetectorWorker, _RECONCILE_TIMEOUT


# ── Fake LiveKit SDK objects ────────────────────────────────────────


class _FakeRoom:
    def __init__(self, name: str, num_participants: int) -> None:
        self.name = name
        self.num_participants = num_participants


class _FakeRoomService:
    """Stand-in for lkapi.LiveKitAPI().room. Configurable response or
    exception per method call.
    """

    def __init__(
        self,
        *,
        list_rooms_response: list[_FakeRoom] | None = None,
        list_rooms_exc: Exception | None = None,
    ) -> None:
        self._response = list_rooms_response or []
        self._exc = list_rooms_exc
        self.list_rooms_call_count = 0

    async def list_rooms(self, req):  # noqa: ANN001 - opaque SDK type
        self.list_rooms_call_count += 1
        if self._exc is not None:
            raise self._exc
        return SimpleNamespace(rooms=list(self._response))


class _FakeLiveKitAPI:
    """Stand-in for lkapi.LiveKitAPI(). Captures whether aclose() was
    called and can be configured to raise on it.
    """

    def __init__(
        self,
        room_service: _FakeRoomService,
        *,
        aclose_exc: Exception | None = None,
    ) -> None:
        self.room = room_service
        self.aclose_called = False
        self._aclose_exc = aclose_exc

    async def aclose(self) -> None:
        self.aclose_called = True
        if self._aclose_exc is not None:
            raise self._aclose_exc


def _install_fake_lkapi(
    monkeypatch: pytest.MonkeyPatch, room_service: _FakeRoomService,
    *, aclose_exc: Exception | None = None,
) -> _FakeLiveKitAPI:
    """Patch lkapi.LiveKitAPI in the worker module so that constructing
    one returns our fake. Mirrors the pattern at test_talk.py:71.
    """
    fake = _FakeLiveKitAPI(room_service, aclose_exc=aclose_exc)
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
    """Return an async function suitable for monkeypatching
    `worker._owned_room_names`. Lets tests stipulate ownership
    without standing up a real Supabase + cameras row.
    """

    async def _owned() -> set[str]:
        return set(rooms)

    return _owned


# ── tests ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_joins_active_patient_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy path: LiveKit returns several patient_* rooms with
    participants → ensure_room_joined is called for each (provided
    we own them in our cameras table).
    """
    rooms = [
        _FakeRoom("patient_aaaa", num_participants=2),
        _FakeRoom("patient_bbbb", num_participants=1),
    ]
    fake = _install_fake_lkapi(monkeypatch, _FakeRoomService(list_rooms_response=rooms))

    worker = _make_worker()
    worker._started = True  # ensure_room_joined would early-return otherwise
    worker._owned_room_names = _fake_owned({"patient_aaaa", "patient_bbbb"})  # type: ignore[assignment]

    joined: list[str] = []

    async def fake_ensure(room_name: str) -> None:
        joined.append(room_name)

    worker.ensure_room_joined = fake_ensure  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert sorted(joined) == ["patient_aaaa", "patient_bbbb"]
    assert fake.aclose_called is True


@pytest.mark.asyncio
async def test_reconcile_skips_foreign_patient_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ownership filter: a patient_* room whose UUID is NOT in our
    cameras table belongs to a different backend (e.g. a coworker's
    process pointed at the same LiveKit deployment). We MUST NOT
    join those rooms — joining would pollute the foreign room's
    state with our worker bot and we have no DB schema for their
    patient/detector/event data anyway.
    """
    rooms = [
        _FakeRoom("patient_mine_aaaa", num_participants=1),
        _FakeRoom("patient_partners_bbbb", num_participants=1),
        _FakeRoom("patient_partners_cccc", num_participants=2),
    ]
    _install_fake_lkapi(monkeypatch, _FakeRoomService(list_rooms_response=rooms))

    worker = _make_worker()
    worker._started = True
    # Only patient_mine_aaaa is ours; the others belong to a
    # coworker's Supabase.
    worker._owned_room_names = _fake_owned({"patient_mine_aaaa"})  # type: ignore[assignment]

    joined: list[str] = []

    async def fake_ensure(room_name: str) -> None:
        joined.append(room_name)

    worker.ensure_room_joined = fake_ensure  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert joined == ["patient_mine_aaaa"]


@pytest.mark.asyncio
async def test_reconcile_skips_zero_participant_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty rooms are a benign artifact of LiveKit's persistence
    semantics — no camera to detect. Skip them to save a Room.connect
    that wouldn't fire any ticks anyway.
    """
    rooms = [
        _FakeRoom("patient_active", num_participants=1),
        _FakeRoom("patient_empty", num_participants=0),
    ]
    _install_fake_lkapi(monkeypatch, _FakeRoomService(list_rooms_response=rooms))

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_active", "patient_empty"})  # type: ignore[assignment]

    joined: list[str] = []

    async def fake_ensure(room_name: str) -> None:
        joined.append(room_name)

    worker.ensure_room_joined = fake_ensure  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert joined == ["patient_active"]


@pytest.mark.asyncio
async def test_reconcile_skips_non_patient_rooms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Egress / talk / other rooms aren't camera streams. Don't join
    them — RoomWorker would sit there idle but it'd still be one
    more LiveKit connection holding bandwidth and an identity slot.
    """
    rooms = [
        _FakeRoom("patient_real", num_participants=1),
        _FakeRoom("egress_recording_42", num_participants=1),
        _FakeRoom("test-room", num_participants=1),
    ]
    _install_fake_lkapi(monkeypatch, _FakeRoomService(list_rooms_response=rooms))

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_real"})  # type: ignore[assignment]

    joined: list[str] = []

    async def fake_ensure(room_name: str) -> None:
        joined.append(room_name)

    worker.ensure_room_joined = fake_ensure  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert joined == ["patient_real"]


@pytest.mark.asyncio
async def test_reconcile_continues_on_per_room_join_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ensure_room_joined raises on one room (e.g. a stale LiveKit
    identity collision), other rooms must still be attempted. A single
    bad room can't poison the whole reconciliation.
    """
    rooms = [
        _FakeRoom("patient_good_1", num_participants=1),
        _FakeRoom("patient_bad", num_participants=1),
        _FakeRoom("patient_good_2", num_participants=1),
    ]
    _install_fake_lkapi(monkeypatch, _FakeRoomService(list_rooms_response=rooms))

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_good_1", "patient_bad", "patient_good_2"})  # type: ignore[assignment]

    joined: list[str] = []

    async def fake_ensure(room_name: str) -> None:
        if room_name == "patient_bad":
            raise RuntimeError("simulated join failure")
        joined.append(room_name)

    worker.ensure_room_joined = fake_ensure  # type: ignore[assignment]

    await worker._reconcile_active_rooms()

    assert sorted(joined) == ["patient_good_1", "patient_good_2"]


@pytest.mark.asyncio
async def test_reconcile_swallows_list_rooms_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If LiveKit is unreachable / down / returning 500s at startup,
    list_rooms raises. We must log + return, NOT crash. The webhook
    path keeps working for any NEW camera connects after this.
    """
    _install_fake_lkapi(
        monkeypatch,
        _FakeRoomService(list_rooms_exc=RuntimeError("LiveKit network down")),
    )

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned(set())  # type: ignore[assignment]
    # No fake ensure_room_joined needed — list_rooms fails before we
    # get there. Test just asserts no exception escapes.
    await worker._reconcile_active_rooms()


@pytest.mark.asyncio
async def test_reconcile_aclose_failure_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If aclose() on the LiveKit client raises (e.g. socket already
    half-closed), the reconciliation is still successful — log and
    move on.
    """
    rooms = [_FakeRoom("patient_only", num_participants=1)]
    _install_fake_lkapi(
        monkeypatch,
        _FakeRoomService(list_rooms_response=rooms),
        aclose_exc=ConnectionError("simulated socket teardown"),
    )

    worker = _make_worker()
    worker._started = True
    worker._owned_room_names = _fake_owned({"patient_only"})  # type: ignore[assignment]

    joined: list[str] = []

    async def fake_ensure(room_name: str) -> None:
        joined.append(room_name)

    worker.ensure_room_joined = fake_ensure  # type: ignore[assignment]

    # Must not raise.
    await worker._reconcile_active_rooms()
    assert joined == ["patient_only"]


@pytest.mark.asyncio
async def test_start_timeout_on_reconcile_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If reconciliation exceeds _RECONCILE_TIMEOUT, start() catches
    the TimeoutError, logs, and returns. The worker is still _started
    and webhook discovery works.
    """
    # Build a fake list_rooms that sleeps longer than the start()
    # timeout to force the path.
    class _SlowRoomService:
        async def list_rooms(self, req):  # noqa: ANN001
            await asyncio.sleep(_RECONCILE_TIMEOUT + 5.0)
            return SimpleNamespace(rooms=[])

    fake = _FakeLiveKitAPI(_SlowRoomService())  # type: ignore[arg-type]
    monkeypatch.setattr(
        "src.services.detection.worker.lkapi.LiveKitAPI",
        lambda url, key, secret: fake,
    )

    # We can't actually wait 30+ seconds for a unit test — patch the
    # timeout to a small value so the test is fast. This is the
    # rare case where patching a private constant is justified:
    # we're proving the timeout branch is wired correctly.
    monkeypatch.setattr(
        "src.services.detection.worker._RECONCILE_TIMEOUT", 0.3,
    )

    worker = _make_worker()
    worker._owned_room_names = _fake_owned(set())  # type: ignore[assignment]
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    await worker.start()
    elapsed = loop.time() - t0

    # Should return within ~0.3s + small slack, NOT 35s.
    assert elapsed < 1.5, f"start() took {elapsed:.2f}s; expected <1.5s"
    assert worker._started is True


@pytest.mark.asyncio
async def test_start_works_without_livekit_creds() -> None:
    """If LIVEKIT_URL/KEY/SECRET aren't configured, start() must
    short-circuit BEFORE attempting reconciliation. Otherwise we'd
    construct a LiveKitAPI with empty strings and the SDK would
    raise on the first call.
    """
    settings = SimpleNamespace(
        livekit_url="",
        livekit_api_key="",
        livekit_api_secret="",
        livekit_motion_state_ttl_seconds=15,
    )
    worker = DetectorWorker(settings=settings, sessionmaker=None)  # type: ignore[arg-type]
    await worker.start()
    assert worker._started is False
