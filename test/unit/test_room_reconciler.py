"""Unit tests for `LiveKitRoomReconciler`.

Drives `reconcile_once()` directly so we can verify each branch of the
two-signal online rule, the absent-streak debounce, the spawn-via-
backfill path, and webhook signal short-circuit without standing up a
real LiveKit room.

`_FakeRoomService` returns DIFFERENT participant sets across successive
calls — this is the core knob for the debounce tests
([X], [X], [], [] → triggers removal on the fourth call).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable
from uuid import UUID, uuid4

import pytest

from src.services.detection.room_reconciler import (
    LiveKitRoomReconciler,
    ReconcileResult,
)


# ── Fake LiveKit SDK objects ────────────────────────────────────────


class _FakeParticipant:
    def __init__(
        self,
        identity: str,
        *,
        publications: list["_FakePublication"] | None = None,
    ) -> None:
        self.identity = identity
        self.track_publications: dict[str, _FakePublication] = {
            p.sid: p for p in (publications or [])
        }


class _FakePublication:
    def __init__(self, sid: str, *, track: object | None = None) -> None:
        self.sid = sid
        self.track = track
        self.set_subscribed_calls: list[bool] = []

    def set_subscribed(self, value: bool) -> None:
        self.set_subscribed_calls.append(value)


class _FakeRoom:
    def __init__(
        self, remote_participants: dict[str, _FakeParticipant] | None = None,
    ) -> None:
        self.remote_participants = remote_participants or {}
        self.disconnect_called = False

    async def disconnect(self) -> None:
        self.disconnect_called = True


class _FakeRoomService:
    """Stand-in for `lkapi.LiveKitAPI().room`. `participants_by_call`
    is a list of participant lists — index N is returned by the Nth
    `list_participants` call. Last element is repeated indefinitely
    once exhausted. Exceptions can be injected for specific calls.
    """

    def __init__(
        self,
        *,
        participants_by_call: list[list[_FakeParticipant]] | None = None,
        list_participants_exc_on_call: dict[int, Exception] | None = None,
        malformed_response_on_call: set[int] | None = None,
    ) -> None:
        self._participants_by_call = participants_by_call or []
        self._exc_on_call = list_participants_exc_on_call or {}
        self._malformed = malformed_response_on_call or set()
        self.call_count: int = 0
        self.list_participants_room_args: list[str] = []

    async def list_participants(self, req):  # noqa: ANN001 - opaque SDK type
        idx = self.call_count
        self.call_count += 1
        self.list_participants_room_args.append(req.room)
        if idx in self._exc_on_call:
            raise self._exc_on_call[idx]
        if idx in self._malformed:
            # Missing `.participants` attribute.
            return SimpleNamespace()
        if not self._participants_by_call:
            return SimpleNamespace(participants=[])
        seq = self._participants_by_call
        chosen = seq[idx] if idx < len(seq) else seq[-1]
        return SimpleNamespace(participants=list(chosen))


class _FakeLiveKitAPI:
    def __init__(self, room_service: _FakeRoomService) -> None:
        self.room = room_service


# ── Fake CameraIngest + RoomWorker ──────────────────────────────────


@dataclass
class _FakeIngest:
    camera_id: UUID
    _last_frame_at: float = 0.0
    stopped: bool = False

    async def stop(self) -> None:
        self.stopped = True


class _FakeRoomWorker:
    """Records spawn/remove calls and exposes the in-memory state the
    reconciler reaches into. `_backfill_participant_tracks` is a hook
    so tests can simulate a successful spawn by populating the ingest
    dict from within it.
    """

    def __init__(
        self,
        *,
        room: _FakeRoom | None = None,
        backfill_handler: Callable[[_FakeParticipant], None] | None = None,
    ) -> None:
        self._room: _FakeRoom | None = room
        self._ingests_by_identity: dict[str, _FakeIngest] = {}
        self._backfill_handler = backfill_handler
        self.backfill_calls: list[str] = []
        self.remove_calls: list[str] = []

    async def _backfill_participant_tracks(self, participant: _FakeParticipant) -> None:
        self.backfill_calls.append(participant.identity)
        if self._backfill_handler is not None:
            self._backfill_handler(participant)

    async def remove_ingest(self, identity: str) -> None:
        self.remove_calls.append(identity)
        ingest = self._ingests_by_identity.pop(identity, None)
        if ingest is not None:
            await ingest.stop()


# ── Fake sessionmaker ───────────────────────────────────────────────


class _FakeResult:
    """Minimal stand-in for a SQLAlchemy CursorResult — only `.rowcount`,
    which the reconciler reads to tell a real transition from a
    self-healing no-op."""

    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeSession:
    def __init__(self, store: "_FakeDBStore") -> None:
        self._store = store

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def execute(self, stmt, params: dict | None = None) -> "_FakeResult":
        # Emulate the guarded, idempotent UPDATEs `camera_presence` issues:
        # each mutates the row ONLY when it would actually change `online`
        # (the WHERE ... AND online = false / true guard), and reports
        # rowcount so the reconciler can distinguish a real flip from a
        # self-healing no-op. Identify the variant by the SET clause (not
        # the whole SQL — the guarded WHERE now also mentions the opposite
        # boolean). Applied at execute time to mirror real UPDATE rowcount
        # semantics.
        sql = str(stmt)
        params = dict(params or {})
        camera_id = params.get("camera_id")
        if camera_id is None:
            return _FakeResult(0)
        cur = bool((self._store.cameras.get(camera_id) or {}).get("online"))
        now = params.get("now")
        if "SET online = true" in sql and "state = 'open'" in sql:
            if not cur:  # guarded false -> true + fresh-start reset
                self._store.set_online(camera_id, True, fresh_start=True, now=now)
                return _FakeResult(1)
            return _FakeResult(0)
        if "SET online = true" in sql:  # guarded false -> true, keep state
            if not cur:
                self._store.set_online(camera_id, True, fresh_start=False, now=now)
                return _FakeResult(1)
            return _FakeResult(0)
        if "SET online = false" in sql:  # guarded true -> false
            if cur:
                self._store.set_online(camera_id, False, fresh_start=False, now=None)
                return _FakeResult(1)
            return _FakeResult(0)
        return _FakeResult(0)

    async def commit(self) -> None:
        return None


class _FakeDBStore:
    """Captures the cameras.online + state mutations the reconciler
    issues. Tests assert against this rather than spinning up Postgres.
    """

    def __init__(self) -> None:
        # camera_id -> dict of last-written values + count of writes per kind
        self.cameras: dict[UUID, dict] = {}
        self.write_log: list[tuple[UUID, bool, bool]] = []
        # (camera_id, online, fresh_start)

    def set_online(
        self,
        camera_id: UUID,
        online: bool,
        *,
        fresh_start: bool,
        now,
    ) -> None:
        entry = self.cameras.setdefault(
            camera_id,
            {"online": None, "fresh_start_count": 0, "writes": 0},
        )
        entry["online"] = online
        entry["writes"] += 1
        if online and fresh_start:
            entry["fresh_start_count"] += 1
        self.write_log.append((camera_id, online, fresh_start))


def _fake_sessionmaker(store: _FakeDBStore):
    def _factory() -> _FakeSession:
        return _FakeSession(store)

    return _factory


# ── Reconciler factory ─────────────────────────────────────────────


def _make_reconciler(
    *,
    room_worker: _FakeRoomWorker,
    room_service: _FakeRoomService,
    db_store: _FakeDBStore,
    poll_interval_s: float = 10.0,
    miss_threshold: int = 2,
    liveness_grace_s: float = 30.0,
    webhook_debounce_s: float = 1.0,
) -> LiveKitRoomReconciler:
    fake_api = _FakeLiveKitAPI(room_service)
    return LiveKitRoomReconciler(
        room_name="patient_test1234",
        room_worker=room_worker,  # type: ignore[arg-type]
        lk_api=fake_api,  # type: ignore[arg-type]
        sessionmaker=_fake_sessionmaker(db_store),  # type: ignore[arg-type]
        poll_interval_s=poll_interval_s,
        miss_threshold=miss_threshold,
        liveness_grace_s=liveness_grace_s,
        webhook_debounce_s=webhook_debounce_s,
    )


# ── Tests ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconciler_spawns_ingest_for_new_cam_identity() -> None:
    """REST returns one cam_* identity; reconciler triggers the spawn
    path (backfill) and writes online=true once frames are fresh."""
    identity = "cam_aaaaaaaa"
    cam_id = uuid4()
    publication = _FakePublication("pub1", track=object())
    participant = _FakeParticipant(identity, publications=[publication])
    room = _FakeRoom(remote_participants={identity: participant})

    def _spawn_in_backfill(_p: _FakeParticipant) -> None:
        ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
        worker._ingests_by_identity[identity] = ingest

    worker = _FakeRoomWorker(room=room, backfill_handler=_spawn_in_backfill)
    svc = _FakeRoomService(participants_by_call=[[participant]])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    result = await rec.reconcile_once()

    assert identity in result.rest_identities
    assert result.spawned == [identity]
    assert result.online_writes == {identity: True}
    assert worker.backfill_calls == [identity]
    # Spawn-via-backfill path always re-issues set_subscribed.
    assert publication.set_subscribed_calls == [True]
    assert store.cameras[cam_id]["online"] is True
    # First-ever write IS a fresh-start.
    assert store.cameras[cam_id]["fresh_start_count"] == 1


@pytest.mark.asyncio
async def test_reconciler_skips_already_tracked_identity() -> None:
    """Identity already in `_ingests_by_identity` and frames fresh →
    no spawn, write online=true once on the first pass."""
    identity = "cam_bbbbbbbb"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    participant = _FakeParticipant(identity)
    svc = _FakeRoomService(participants_by_call=[[participant], [participant]])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    result1 = await rec.reconcile_once()
    result2 = await rec.reconcile_once()

    assert result1.spawned == []
    assert result2.spawned == []
    assert worker.backfill_calls == []
    # online=true written exactly once (no-op on the second pass).
    assert store.cameras[cam_id]["writes"] == 1
    assert result1.online_writes == {identity: True}
    assert result2.online_writes == {}


@pytest.mark.asyncio
async def test_reconciler_filters_non_cam_identities() -> None:
    """detector_worker_*, caregiver_*, agent_* identities are ignored."""
    cam_id = uuid4()
    cam_ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity["cam_camcam99"] = cam_ingest

    svc = _FakeRoomService(participants_by_call=[[
        _FakeParticipant("cam_camcam99"),
        _FakeParticipant("detector_worker_test1234"),
        _FakeParticipant("caregiver_abc_ios"),
        _FakeParticipant("agent_something"),
    ]])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    result = await rec.reconcile_once()

    assert result.rest_identities == {"cam_camcam99"}
    assert worker.backfill_calls == []
    assert worker.remove_calls == []


@pytest.mark.asyncio
async def test_reconciler_single_missing_poll_does_not_remove() -> None:
    """[X], [] over two polls — absent_streak=1 < miss_threshold=2, so
    ingest stays put."""
    identity = "cam_singlemis"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    participant = _FakeParticipant(identity)
    svc = _FakeRoomService(participants_by_call=[[participant], []])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store, miss_threshold=2)

    await rec.reconcile_once()  # present
    await rec.reconcile_once()  # absent (streak=1)

    assert worker.remove_calls == []
    assert identity in worker._ingests_by_identity


@pytest.mark.asyncio
async def test_reconciler_two_consecutive_misses_removes_ingest() -> None:
    """[X], [X], [], [] — second absent poll triggers removal."""
    identity = "cam_evictme0"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    participant = _FakeParticipant(identity)
    svc = _FakeRoomService(participants_by_call=[
        [participant], [participant], [], [],
    ])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store, miss_threshold=2)

    await rec.reconcile_once()  # present, online=true write
    await rec.reconcile_once()  # present, no-op
    await rec.reconcile_once()  # absent_streak=1
    result4 = await rec.reconcile_once()  # absent_streak=2 → evict

    assert worker.remove_calls == [identity]
    assert result4.removed == [identity]
    assert identity not in worker._ingests_by_identity
    assert store.cameras[cam_id]["online"] is False


@pytest.mark.asyncio
async def test_reconciler_miss_then_present_resets_streak() -> None:
    """[X], [], [X] — present pass resets absent_streak, no eviction
    on a subsequent single absent pass."""
    identity = "cam_flapfla1"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    participant = _FakeParticipant(identity)
    svc = _FakeRoomService(participants_by_call=[
        [participant], [], [participant], [],
    ])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store, miss_threshold=2)

    await rec.reconcile_once()  # present
    await rec.reconcile_once()  # absent (streak=1)
    await rec.reconcile_once()  # present (streak reset)
    await rec.reconcile_once()  # absent (streak=1 again, NOT 2)

    assert worker.remove_calls == []
    assert identity in worker._ingests_by_identity


@pytest.mark.asyncio
async def test_reconciler_rest_timeout_keeps_state() -> None:
    """REST raises TimeoutError → no spawn, no remove, no DB writes.
    State machine and ingest dict untouched."""
    identity = "cam_timeoutc"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    svc = _FakeRoomService(
        participants_by_call=[[_FakeParticipant(identity)]],
        list_participants_exc_on_call={0: asyncio.TimeoutError()},
    )
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    result = await rec.reconcile_once()

    assert result == ReconcileResult()
    assert worker.remove_calls == []
    assert store.write_log == []


@pytest.mark.asyncio
async def test_reconciler_rest_returns_malformed_response() -> None:
    """Response missing `.participants` attribute → treated as failure;
    state untouched."""
    identity = "cam_malforme"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    svc = _FakeRoomService(malformed_response_on_call={0})
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    result = await rec.reconcile_once()

    assert result == ReconcileResult()
    assert store.write_log == []


@pytest.mark.asyncio
async def test_reconciler_rest_raises_does_not_crash_loop() -> None:
    """First poll raises RuntimeError, second succeeds. Loop must
    continue running and process the second poll normally."""
    identity = "cam_raisem01"
    cam_id = uuid4()
    publication = _FakePublication("pub1", track=object())
    participant = _FakeParticipant(identity, publications=[publication])

    def _spawn_in_backfill(_p: _FakeParticipant) -> None:
        worker._ingests_by_identity[identity] = _FakeIngest(
            camera_id=cam_id, _last_frame_at=time.monotonic(),
        )

    worker = _FakeRoomWorker(
        room=_FakeRoom(remote_participants={identity: participant}),
        backfill_handler=_spawn_in_backfill,
    )
    svc = _FakeRoomService(
        participants_by_call=[[participant], [participant]],
        list_participants_exc_on_call={0: RuntimeError("boom")},
    )
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    await rec.reconcile_once()  # raises, swallowed
    result2 = await rec.reconcile_once()

    assert result2.spawned == [identity]
    assert store.cameras[cam_id]["online"] is True


@pytest.mark.asyncio
async def test_reconciler_idempotent_same_identity_repeated() -> None:
    """Same cam_* identity present in 5 calls → exactly ONE online=true
    write (the first); no duplicate spawns."""
    identity = "cam_iddempot"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    p = _FakeParticipant(identity)
    svc = _FakeRoomService(participants_by_call=[[p]] * 5)
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    for _ in range(5):
        await rec.reconcile_once()

    assert store.cameras[cam_id]["writes"] == 1
    assert worker.backfill_calls == []


@pytest.mark.asyncio
async def test_reconciler_webhook_signal_short_circuits_sleep() -> None:
    """signal_tick wakes the loop within debounce window even if the
    periodic interval has not elapsed.

    Verified by starting the loop with a long poll interval, firing
    signal_tick after 0.2s, and asserting an extra poll fires within
    1.0s wall-clock.
    """
    identity = "cam_sigwakeu"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    svc = _FakeRoomService(participants_by_call=[[_FakeParticipant(identity)]] * 10)
    store = _FakeDBStore()
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, db_store=store,
        poll_interval_s=30.0, webhook_debounce_s=0.05,
    )
    await rec.start()
    try:
        # Let the initial poll-or-wait state settle (no tick yet).
        await asyncio.sleep(0.1)
        before = svc.call_count
        await rec.signal_tick()
        # Allow the loop to pick up the event.
        await asyncio.sleep(0.5)
        after = svc.call_count
    finally:
        await rec.stop()

    assert after >= before + 1, (
        f"signal_tick should have triggered an immediate poll; "
        f"calls before={before} after={after}"
    )


@pytest.mark.asyncio
async def test_reconciler_webhook_debounce_coalesces_multiple_signals() -> None:
    """Three signal_tick calls in <100ms → at most ONE extra poll."""
    identity = "cam_debounc1"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    svc = _FakeRoomService(participants_by_call=[[_FakeParticipant(identity)]] * 10)
    store = _FakeDBStore()
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, db_store=store,
        poll_interval_s=30.0, webhook_debounce_s=1.0,
    )
    await rec.start()
    try:
        await asyncio.sleep(0.05)
        before = svc.call_count
        await rec.signal_tick()
        await rec.signal_tick()  # debounced
        await rec.signal_tick()  # debounced
        await asyncio.sleep(0.5)
        after = svc.call_count
    finally:
        await rec.stop()

    # Exactly one signal-triggered poll (or zero if scheduling was
    # unfortunate; allow both but never more than 1 extra).
    assert after - before <= 1


@pytest.mark.asyncio
async def test_reconciler_cancellation_exits_cleanly() -> None:
    """stop() cancels the loop and awaits it; no warnings about un-
    awaited tasks."""
    worker = _FakeRoomWorker(room=_FakeRoom())
    svc = _FakeRoomService(participants_by_call=[[]])
    store = _FakeDBStore()
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, db_store=store,
        poll_interval_s=30.0,
    )
    await rec.start()
    await asyncio.sleep(0.05)
    await rec.stop()
    # Idempotent: second stop is a no-op.
    await rec.stop()
    assert rec._loop_task is None


@pytest.mark.asyncio
async def test_reconciler_orphan_identity_no_db_camera_logs_once_skips() -> None:
    """REST returns a cam_* identity but no participant in the SDK dict
    AND no ingest can be spawned → log once, no DB writes."""
    identity = "cam_orphan99"
    # Empty SDK dict — the reconciler will see participant=None,
    # log the desync warning, and call room.disconnect (heavy hammer).
    room = _FakeRoom(remote_participants={})
    worker = _FakeRoomWorker(room=room)
    svc = _FakeRoomService(participants_by_call=[
        [_FakeParticipant(identity)],
        [_FakeParticipant(identity)],
    ])
    store = _FakeDBStore()
    rec = _make_reconciler(room_worker=worker, room_service=svc, db_store=store)

    await rec.reconcile_once()
    await rec.reconcile_once()

    # No spawn happened; no DB writes; reconciler should have asked
    # the room to disconnect at least once during the desync handling.
    assert worker.backfill_calls == []
    assert store.write_log == []
    assert room.disconnect_called is True


@pytest.mark.asyncio
async def test_reconciler_two_signal_present_in_rest_stale_frames_writes_offline() -> None:
    """REST has identity, but `_last_frame_at` is older than
    `liveness_grace_s` → online=false despite presence in REST."""
    identity = "cam_stale011"
    cam_id = uuid4()
    # `_last_frame_at` set to 60s ago, grace window is 30s.
    stale_ts = time.monotonic() - 60.0
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=stale_ts)
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    svc = _FakeRoomService(participants_by_call=[[_FakeParticipant(identity)]])
    store = _FakeDBStore()
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, db_store=store,
        liveness_grace_s=30.0,
    )

    # Seed the row online=true so the stale-frame pass has a real true->false
    # transition to make. (The guarded offline UPDATE is a no-op on an
    # already-offline row — the correct, echo-free behavior — so without a
    # prior true there would be nothing to flip.)
    store.cameras[cam_id] = {"online": True, "fresh_start_count": 0, "writes": 0}

    result = await rec.reconcile_once()

    assert result.online_writes == {identity: False}
    assert store.cameras[cam_id]["online"] is False


@pytest.mark.asyncio
async def test_reconciler_two_signal_present_in_rest_fresh_frames_writes_online() -> None:
    identity = "cam_freshlol"
    cam_id = uuid4()
    ingest = _FakeIngest(camera_id=cam_id, _last_frame_at=time.monotonic())
    worker = _FakeRoomWorker(room=_FakeRoom())
    worker._ingests_by_identity[identity] = ingest

    svc = _FakeRoomService(participants_by_call=[[_FakeParticipant(identity)]])
    store = _FakeDBStore()
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, db_store=store,
        liveness_grace_s=30.0,
    )

    result = await rec.reconcile_once()

    assert result.online_writes == {identity: True}
    assert store.cameras[cam_id]["online"] is True


@pytest.mark.asyncio
async def test_reconciler_online_true_fresh_start_resets_state_machine() -> None:
    """First online=true write for an identity uses the fresh-start
    SQL which resets state='open', no_people_count=0. Subsequent
    writes (after a re-spawn) trigger a NEW fresh-start."""
    identity = "cam_freshrst"
    cam_id = uuid4()

    # Pass 1: spawn + fresh-start write.
    publication = _FakePublication("pub1", track=object())
    participant = _FakeParticipant(identity, publications=[publication])
    room = _FakeRoom(remote_participants={identity: participant})

    def _spawn_in_backfill(_p: _FakeParticipant) -> None:
        worker._ingests_by_identity[identity] = _FakeIngest(
            camera_id=cam_id, _last_frame_at=time.monotonic(),
        )

    worker = _FakeRoomWorker(room=room, backfill_handler=_spawn_in_backfill)
    svc = _FakeRoomService(participants_by_call=[
        [participant],     # present + fresh → online=true fresh_start #1
        [participant],     # still present → no-op
        [],                # absent_streak=1
        [],                # absent_streak=2 → remove + online=false
        [participant],     # present again → online=true fresh_start #2
    ])
    store = _FakeDBStore()
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, db_store=store, miss_threshold=2,
    )

    await rec.reconcile_once()
    await rec.reconcile_once()
    await rec.reconcile_once()
    await rec.reconcile_once()
    # Simulate the next spawn re-populating the ingest dict on backfill.
    await rec.reconcile_once()

    assert store.cameras[cam_id]["fresh_start_count"] == 2


@pytest.mark.asyncio
async def test_reconciler_no_lkapi_skips_silently() -> None:
    """`lk_api=None` (config missing) returns a no-op pass."""
    worker = _FakeRoomWorker(room=_FakeRoom())
    store = _FakeDBStore()
    rec = LiveKitRoomReconciler(
        room_name="patient_unconfig",
        room_worker=worker,  # type: ignore[arg-type]
        lk_api=None,
        sessionmaker=_fake_sessionmaker(store),  # type: ignore[arg-type]
    )
    result = await rec.reconcile_once()
    assert result == ReconcileResult()
