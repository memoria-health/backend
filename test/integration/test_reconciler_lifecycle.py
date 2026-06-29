"""Integration tests for `LiveKitRoomReconciler` against real Postgres.

Unit tests in `test/unit/test_room_reconciler.py` cover branch logic
with a fake DB; this file pins the *DB-side* contract end-to-end:

  - The atomic-UPDATE SQL strings the reconciler issues actually mutate
    `cameras.online` (and the fresh-start state-machine reset) in real
    Postgres.
  - The fresh-start reset on closed→online sets `state='open'`,
    `no_people_count=0`, `state_changed_at=now` in one statement —
    a regression here would mean a freshly-resumed camera stays
    counted-down-toward-closed even after frames arrive.
  - An identity present in the LiveKit REST response but missing from
    `cameras` (orphaned by a DELETE) is a no-op — the reconciler
    silently skips it, no FK violation, no stuck task. Logs only once.
  - A backend restart (= fresh `LiveKitRoomReconciler` instance over
    the same DB row) picks up the existing online publisher on its
    first poll and writes `online=true`.

LiveKit is mocked via the same fake-SDK pattern from
`test/unit/test_room_reconciler.py` so we don't need LiveKit Cloud or
network reachability. The reconciler is constructed directly with the
real `get_sessionmaker()` (real Postgres) + a fake `lkapi.LiveKitAPI`.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Callable
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services.detection.room_reconciler import LiveKitRoomReconciler


# ── Fake LiveKit SDK objects (mirrors test/unit/test_room_reconciler.py)
# These are intentionally duplicated here rather than imported from the
# unit test module to keep integration / unit boundaries clean. The two
# files evolve independently if the fake shape diverges.


class _FakePublication:
    def __init__(self, sid: str, *, track: object | None = None) -> None:
        self.sid = sid
        self.track = track
        self.set_subscribed_calls: list[bool] = []

    def set_subscribed(self, value: bool) -> None:
        self.set_subscribed_calls.append(value)


class _FakeParticipant:
    def __init__(
        self,
        identity: str,
        *,
        publications: list[_FakePublication] | None = None,
    ) -> None:
        self.identity = identity
        self.track_publications: dict[str, _FakePublication] = {
            p.sid: p for p in (publications or [])
        }


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
    is a list of participant lists; index N returns on the Nth
    `list_participants` call. Last element repeats indefinitely.
    """

    def __init__(
        self,
        *,
        participants_by_call: list[list[_FakeParticipant]] | None = None,
    ) -> None:
        self._participants_by_call = participants_by_call or []
        self.call_count: int = 0
        self.list_participants_room_args: list[str] = []

    async def list_participants(self, req):  # noqa: ANN001 - opaque SDK type
        idx = self.call_count
        self.call_count += 1
        self.list_participants_room_args.append(req.room)
        if not self._participants_by_call:
            return SimpleNamespace(participants=[])
        seq = self._participants_by_call
        chosen = seq[idx] if idx < len(seq) else seq[-1]
        return SimpleNamespace(participants=list(chosen))


class _FakeLiveKitAPI:
    def __init__(self, room_service: _FakeRoomService) -> None:
        self.room = room_service


# ── Fake CameraIngest + RoomWorker ────────────────────────────────


@dataclass
class _FakeIngest:
    camera_id: UUID
    _last_frame_at: float = 0.0
    stopped: bool = False

    async def stop(self) -> None:
        self.stopped = True


class _FakeRoomWorker:
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


# ── DB seeding helper ────────────────────────────────────────────


async def _seed_caregiver_patient_camera(
    session: AsyncSession,
    auth_user_factory,
    *,
    livekit_identity: str,
    online: bool = False,
    state: str = "closed",
    no_people_count: int = 3,
) -> tuple[UUID, UUID, UUID]:
    """Seed a caregiver + patient + one camera. Camera defaults are
    chosen so the fresh-start reset is observable: state='closed',
    no_people_count=3 → after a successful online transition the row
    must read state='open', no_people_count=0.
    """
    uid, email = await auth_user_factory()
    now = datetime.now(timezone.utc)
    session.add(
        Caregiver(
            id=uid, display_name="Reconciler Tester", email=email,
            created_at=now, updated_at=now,
        )
    )
    patient_id = uuid4()
    session.add(
        Patient(
            id=patient_id, caregiver_id=uid,
            full_name="Margaret O'Brien", preferred_name="Maggie",
            relation="parent", state="calm", status_line="OK",
            cameras_paused=False, created_at=now, updated_at=now,
        )
    )
    await session.flush()
    camera_id = uuid4()
    session.add(
        Camera(
            id=camera_id, patient_id=patient_id, display_id=1,
            room="Bedroom", icon_name="bed.double", online=online,
            api_key_hash=None, livekit_identity=livekit_identity,
            state=state,
            no_people_count=no_people_count, close_threshold=6,
            created_at=now,
        )
    )
    await session.commit()
    return uid, patient_id, camera_id


def _make_reconciler(
    *,
    room_worker: _FakeRoomWorker,
    room_service: _FakeRoomService,
    room_name: str,
    miss_threshold: int = 2,
    liveness_grace_s: float = 30.0,
    webhook_debounce_s: float = 1.0,
    poll_interval_s: float = 10.0,
) -> LiveKitRoomReconciler:
    fake_api = _FakeLiveKitAPI(room_service)
    return LiveKitRoomReconciler(
        room_name=room_name,
        room_worker=room_worker,  # type: ignore[arg-type]
        lk_api=fake_api,  # type: ignore[arg-type]
        sessionmaker=get_sessionmaker(),
        poll_interval_s=poll_interval_s,
        miss_threshold=miss_threshold,
        liveness_grace_s=liveness_grace_s,
        webhook_debounce_s=webhook_debounce_s,
    )


# ── Tests ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_lifecycle_paired_camera_via_reconciler(
    auth_user_factory,
) -> None:
    """End-to-end against real Postgres:

      1. Camera row starts online=false, state='closed', count=3.
      2. REST returns [cam_X] for two polls (present → present), frames
         fresh: reconciler writes online=true AND fresh-start reset
         (state='open', no_people_count=0, state_changed_at=now) on the
         first pass.
      3. REST returns [] for two polls (miss_threshold=2): second
         absent pass evicts the ingest and writes online=false.
    """
    identity = "cam_lifecyc1"
    room_name = "patient_lifecycle"

    sm = get_sessionmaker()
    async with sm() as session:
        _, _, camera_id = await _seed_caregiver_patient_camera(
            session, auth_user_factory,
            livekit_identity=identity,
            online=False, state="closed", no_people_count=3,
        )

    publication = _FakePublication("pub-vid", track=object())
    participant = _FakeParticipant(identity, publications=[publication])
    room = _FakeRoom(remote_participants={identity: participant})

    def _spawn_in_backfill(_p: _FakeParticipant) -> None:
        ingest = _FakeIngest(
            camera_id=camera_id, _last_frame_at=time.monotonic(),
        )
        worker._ingests_by_identity[identity] = ingest

    worker = _FakeRoomWorker(room=room, backfill_handler=_spawn_in_backfill)
    svc = _FakeRoomService(participants_by_call=[
        [participant], [participant],   # present, present
        [], [],                          # absent, absent
    ])
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, room_name=room_name,
        miss_threshold=2,
    )

    # ── pass 1: spawn + online=true fresh-start ──────────────────
    r1 = await rec.reconcile_once()
    assert r1.spawned == [identity]
    assert worker.backfill_calls == [identity]
    assert publication.set_subscribed_calls == [True]
    assert r1.online_writes == {identity: True}

    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is True
        # Fresh-start reset must have fired: counter zeroed + state opened.
        assert cam.state == "open"
        assert cam.no_people_count == 0
        assert cam.state_changed_at is not None

    state_changed_at_after_first = cam.state_changed_at

    # ── pass 2: idempotent (still present + already online) ──────
    r2 = await rec.reconcile_once()
    assert r2.spawned == []
    assert r2.online_writes == {}    # no-op, already online
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is True
        assert cam.state == "open"
        # state_changed_at unchanged on a no-op pass — pins that the
        # SQL guard against re-writing on every poll is in place.
        assert cam.state_changed_at == state_changed_at_after_first

    # ── pass 3: absent_streak=1, still online ────────────────────
    r3 = await rec.reconcile_once()
    assert r3.removed == []
    assert identity in worker._ingests_by_identity
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is True   # debounce hasn't tripped yet

    # ── pass 4: absent_streak=2 → evict + online=false write ─────
    r4 = await rec.reconcile_once()
    assert r4.removed == [identity]
    assert worker.remove_calls == [identity]
    assert identity not in worker._ingests_by_identity
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is False


@pytest.mark.asyncio
async def test_subsequent_passes_do_not_re_fire_fresh_start(
    auth_user_factory,
) -> None:
    """The fresh-start SQL (`_SET_ONLINE_TRUE_FRESH_START_SQL`) zeroes
    `no_people_count` + stamps `state_changed_at`. Once the reconciler
    has written online=true for an identity, subsequent in-session
    passes must take the KEEP_STATE branch — otherwise every 10s poll
    would zero a legitimately-advancing close countdown.

    NOTE on backend-restart behavior: the in-memory
    `_last_online_written` cache is empty on a fresh reconciler
    instance, so the FIRST write after restart IS a fresh-start by
    design (mirrors the deleted `livekit_webhooks._set_camera_online`
    semantic — see room_reconciler.py:124-130). This test pins the
    in-session no-op contract, not the restart-fresh-start contract.
    """
    identity = "cam_keepst01"
    room_name = "patient_keepst"
    sm = get_sessionmaker()
    async with sm() as session:
        _, _, camera_id = await _seed_caregiver_patient_camera(
            session, auth_user_factory,
            livekit_identity=identity,
            online=False, state="closed", no_people_count=3,
        )

    publication = _FakePublication("pub-vid", track=object())
    participant = _FakeParticipant(identity, publications=[publication])
    room = _FakeRoom(remote_participants={identity: participant})

    def _spawn_in_backfill(_p: _FakeParticipant) -> None:
        worker._ingests_by_identity[identity] = _FakeIngest(
            camera_id=camera_id, _last_frame_at=time.monotonic(),
        )

    worker = _FakeRoomWorker(room=room, backfill_handler=_spawn_in_backfill)
    svc = _FakeRoomService(participants_by_call=[
        [participant], [participant], [participant],
    ])
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, room_name=room_name,
    )

    # ── pass 1: fresh-start (false → true). State machine reset. ─
    await rec.reconcile_once()
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is True
        assert cam.state == "open"
        assert cam.no_people_count == 0
        first_changed_at = cam.state_changed_at
        assert first_changed_at is not None

    # Simulate the detector ticking and advancing the close counter
    # mid-session (the publisher stays connected the whole time).
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        cam.no_people_count = 4
        await session.commit()

    # ── pass 2: subsequent online=true with publisher still present.
    # MUST take the KEEP_STATE branch — counter stays at 4. ────
    r2 = await rec.reconcile_once()
    assert r2.online_writes == {}   # no-op write (already online cached)
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is True
        assert cam.state == "open"
        # KEEP_STATE branch leaves the counter alone. If the reconciler
        # ever regressed to fresh-start on every poll, this would be 0.
        assert cam.no_people_count == 4
        # state_changed_at not updated on a no-op pass.
        assert cam.state_changed_at == first_changed_at

    # ── pass 3: same — confirms the cache stays True. ─────────────
    r3 = await rec.reconcile_once()
    assert r3.online_writes == {}
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.no_people_count == 4


@pytest.mark.asyncio
async def test_orphan_identity_no_db_row_is_no_op_clean(
    auth_user_factory,
) -> None:
    """LiveKit Cloud holds a phantom cam_* participant whose `cameras`
    row was deleted (DELETE handler ran between webhook delivery and
    our REST poll). Reconciler must skip cleanly: no spawn, no FK
    violation, no stuck task. Re-polling repeatedly must not re-log.
    """
    orphan_identity = "cam_orphan01"
    room_name = "patient_orphanr"

    # Seed only the caregiver + patient + room (no camera row for the
    # orphan identity).
    sm = get_sessionmaker()
    async with sm() as session:
        await _seed_caregiver_patient_camera(
            session, auth_user_factory,
            livekit_identity="cam_realreal",  # different camera, present
            online=False,
        )

    participant = _FakeParticipant(orphan_identity)
    room = _FakeRoom(remote_participants={orphan_identity: participant})
    worker = _FakeRoomWorker(room=room)
    svc = _FakeRoomService(participants_by_call=[
        [participant], [participant], [participant],
    ])
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, room_name=room_name,
    )

    # Three polls of the orphan: the reconciler will attempt spawn each
    # pass (it can't tell from REST alone that the DB row is missing),
    # so backfill IS called. But no DB UPDATE is ever issued (nothing
    # to write online=true for — no `_ingests_by_identity` entry =>
    # no `camera_id`) and no ingest gets evicted. This is the safety
    # property: phantom identities never trigger writes against a
    # phantom camera_id.
    for _ in range(3):
        result = await rec.reconcile_once()
        assert result.spawned == []
        assert result.removed == []
        assert result.online_writes == {}

    # Spawn attempted on every pass (backfill is the reconciler's only
    # path to get an ingest in place), but it produced no ingest so no
    # downstream DB write fired.
    assert worker.backfill_calls == [orphan_identity] * 3
    assert worker._ingests_by_identity == {}
    assert worker.remove_calls == []
    # The orphan warning is logged exactly once (state captured in
    # `_logged_orphan_identities` so per-poll noise is suppressed).
    assert rec._logged_orphan_identities == {orphan_identity}


@pytest.mark.asyncio
async def test_backend_restart_respawns_via_first_reconciler_poll(
    auth_user_factory,
) -> None:
    """Simulate the backend-restart-with-live-publisher case. A
    fresh `LiveKitRoomReconciler` (no in-process memory of past
    state) over the SAME `cameras` row: on its first poll it picks
    up the existing identity in REST, calls backfill (which spawns
    the ingest), and writes online=true.
    """
    identity = "cam_postrest"
    room_name = "patient_postrest"
    sm = get_sessionmaker()
    async with sm() as session:
        _, _, camera_id = await _seed_caregiver_patient_camera(
            session, auth_user_factory,
            livekit_identity=identity,
            online=False, state="closed", no_people_count=2,
        )

    publication = _FakePublication("pub-vid", track=object())
    participant = _FakeParticipant(identity, publications=[publication])
    room = _FakeRoom(remote_participants={identity: participant})

    def _spawn_in_backfill(_p: _FakeParticipant) -> None:
        worker._ingests_by_identity[identity] = _FakeIngest(
            camera_id=camera_id, _last_frame_at=time.monotonic(),
        )

    worker = _FakeRoomWorker(room=room, backfill_handler=_spawn_in_backfill)
    svc = _FakeRoomService(participants_by_call=[[participant]])
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, room_name=room_name,
    )

    started = time.monotonic()
    r = await rec.reconcile_once()
    elapsed = time.monotonic() - started

    assert r.spawned == [identity]
    assert r.online_writes == {identity: True}
    # First poll respawn must be fast — well under the 12s wall-clock
    # bound from the plan. Tightened to 5s for early-warning here; if
    # this ever flakes, the production behavior degraded too.
    assert elapsed < 5.0

    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.online is True
        # Fresh-start: state machine reset since the prior session was
        # also closed → online transition.
        assert cam.state == "open"
        assert cam.no_people_count == 0


@pytest.mark.asyncio
async def test_signal_tick_short_circuits_periodic_sleep(
    auth_user_factory,
) -> None:
    """Webhook fast-path. The reconciler's `start()` loop blocks on a
    long periodic sleep; `signal_tick()` must release it so the next
    `reconcile_once`-equivalent pass runs immediately.

    We assert the contract on the `_tick_event` rather than starting
    the full loop (which would race wall-clock timers and flake under
    load). `signal_tick` setting the event is the load-bearing piece
    that webhook → fast-path-reconcile depends on.
    """
    identity = "cam_signalti"
    room_name = "patient_signalt"
    sm = get_sessionmaker()
    async with sm() as session:
        await _seed_caregiver_patient_camera(
            session, auth_user_factory,
            livekit_identity=identity, online=False,
        )

    worker = _FakeRoomWorker(room=_FakeRoom())
    svc = _FakeRoomService(participants_by_call=[[]])
    rec = _make_reconciler(
        room_worker=worker, room_service=svc, room_name=room_name,
        webhook_debounce_s=0.001,  # don't debounce in this test
    )

    # Event starts clear.
    assert not rec._tick_event.is_set()

    # Webhook signal → event set.
    started = time.monotonic()
    await rec.signal_tick()
    elapsed = time.monotonic() - started

    assert rec._tick_event.is_set()
    # signal_tick is the synchronous control path — must complete in
    # well under 1s (the production budget the webhook handler hands
    # back to LiveKit Cloud).
    assert elapsed < 0.5
