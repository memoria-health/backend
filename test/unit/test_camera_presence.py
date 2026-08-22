"""Unit tests for `camera_presence` — the single owner of `cameras.online`.

These exercise the webhook/startup trigger-point helpers against a fake
session (no Postgres), pinning:

  - `mark_offline_by_identity` issues the identity-scoped offline UPDATE
    and reports whether a row actually flipped.
  - `mark_room_offline` issues the room-scoped offline UPDATE.
  - `sweep_stale_online` forces offline only the online cameras whose
    room is NOT in the active set, leaving active-room cameras alone.
  - All three swallow DB errors and never raise (a webhook ack / startup
    must not be blocked on a transient DB failure).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from src.services.detection import camera_presence


class _FakeResult:
    def __init__(self, *, rows=None, rowcount=0) -> None:
        self._rows = rows or []
        self.rowcount = rowcount

    def all(self):
        return self._rows


class _FakeSession:
    """Models just enough of the cameras table to answer the handful of
    statements `camera_presence` issues. `store["cameras"]` is a list of
    dicts: {id, identity, room, online}.
    """

    def __init__(self, store: dict) -> None:
        self._store = store

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def execute(self, stmt, params: dict | None = None):
        sql = str(stmt)
        params = params or {}
        self._store["executed"].append((sql, params))
        cams = self._store["cameras"]
        if "SELECT id, livekit_room" in sql:
            rows = [(c["id"], c["room"]) for c in cams if c["online"]]
            return _FakeResult(rows=rows)
        # Checked BEFORE the offline branch: the re-arm UPDATE is also
        # identity-scoped, and would otherwise be mistaken for one.
        if "setup_test_passed_at = NULL" in sql:
            ident = params.get("identity")
            n = 0
            for c in cams:
                if c["identity"] == ident and c.get("setup_test_passed_at"):
                    c["setup_test_passed_at"] = None
                    n += 1
            return _FakeResult(rowcount=n)
        if "livekit_identity = :identity" in sql:
            ident = params.get("identity")
            n = 0
            for c in cams:
                if c["identity"] == ident and c["online"]:
                    c["online"] = False
                    n += 1
            return _FakeResult(rowcount=n)
        if "livekit_room = :room" in sql:
            room = params.get("room")
            n = 0
            for c in cams:
                if c["room"] == room and c["online"]:
                    c["online"] = False
                    n += 1
            return _FakeResult(rowcount=n)
        if "online = false" in sql and ":camera_id" in sql:
            cid = params.get("camera_id")
            for c in cams:
                if c["id"] == cid:
                    c["online"] = False
            return _FakeResult(rowcount=1)
        return _FakeResult()

    async def commit(self) -> None:
        self._store["commits"] += 1


def _sessionmaker(store: dict):
    def _factory() -> _FakeSession:
        return _FakeSession(store)

    return _factory


def _store(cameras: list[dict]) -> dict:
    return {"cameras": cameras, "executed": [], "commits": 0}


# ── mark_offline_by_identity ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_offline_by_identity_flips_online_camera() -> None:
    cid = uuid4()
    store = _store([
        {"id": cid, "identity": "cam_aaaa", "room": "patient_x", "online": True},
    ])
    flipped = await camera_presence.mark_offline_by_identity(
        _sessionmaker(store), "cam_aaaa",
    )
    assert flipped is True
    assert store["cameras"][0]["online"] is False
    assert store["commits"] == 1


@pytest.mark.asyncio
async def test_mark_offline_by_identity_noop_when_already_offline() -> None:
    cid = uuid4()
    store = _store([
        {"id": cid, "identity": "cam_aaaa", "room": "patient_x", "online": False},
    ])
    flipped = await camera_presence.mark_offline_by_identity(
        _sessionmaker(store), "cam_aaaa",
    )
    # Idempotent: nothing flipped → False (so we don't emit a no-op
    # Realtime echo).
    assert flipped is False


@pytest.mark.asyncio
async def test_mark_offline_by_identity_empty_identity_is_noop() -> None:
    store = _store([])
    flipped = await camera_presence.mark_offline_by_identity(
        _sessionmaker(store), "",
    )
    assert flipped is False
    assert store["executed"] == []


# ── mark_room_offline ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_room_offline_flips_every_camera_in_room() -> None:
    a, b, c = uuid4(), uuid4(), uuid4()
    store = _store([
        {"id": a, "identity": "cam_a", "room": "patient_x", "online": True},
        {"id": b, "identity": "cam_b", "room": "patient_x", "online": True},
        {"id": c, "identity": "cam_c", "room": "patient_y", "online": True},
    ])
    n = await camera_presence.mark_room_offline(_sessionmaker(store), "patient_x")
    assert n == 2
    assert store["cameras"][0]["online"] is False
    assert store["cameras"][1]["online"] is False
    # Camera in a different room is untouched.
    assert store["cameras"][2]["online"] is True


# ── sweep_stale_online ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sweep_stale_online_offlines_only_inactive_rooms() -> None:
    live = uuid4()       # room IS active → leave alone
    stale = uuid4()      # room not active → force offline
    orphan = uuid4()     # no room at all → force offline
    store = _store([
        {"id": live, "identity": "cam_live", "room": "patient_active", "online": True},
        {"id": stale, "identity": "cam_stale", "room": "patient_gone", "online": True},
        {"id": orphan, "identity": "cam_orphan", "room": None, "online": True},
    ])
    flipped = await camera_presence.sweep_stale_online(
        _sessionmaker(store), {"patient_active"},
    )
    assert flipped == 2
    assert store["cameras"][0]["online"] is True   # active room left alone
    assert store["cameras"][1]["online"] is False  # gone room swept
    assert store["cameras"][2]["online"] is False  # no-room swept


@pytest.mark.asyncio
async def test_sweep_stale_online_noop_when_all_active() -> None:
    a = uuid4()
    store = _store([
        {"id": a, "identity": "cam_a", "room": "patient_active", "online": True},
    ])
    flipped = await camera_presence.sweep_stale_online(
        _sessionmaker(store), {"patient_active"},
    )
    assert flipped == 0
    assert store["cameras"][0]["online"] is True


# ── error tolerance ───────────────────────────────────────────────────


class _BoomSession:
    async def __aenter__(self):
        raise RuntimeError("db down")

    async def __aexit__(self, *args):
        return None


def _boom_sessionmaker():
    def _factory():
        return _BoomSession()

    return _factory


@pytest.mark.asyncio
async def test_helpers_never_raise_on_db_failure() -> None:
    sm = _boom_sessionmaker()
    assert await camera_presence.mark_offline_by_identity(sm, "cam_x") is False
    assert await camera_presence.mark_room_offline(sm, "patient_x") == 0
    assert await camera_presence.sweep_stale_online(sm, set()) == 0
    assert await camera_presence.rearm_setup_test_by_identity(sm, "cam_x") is False


# ── rearm_setup_test_by_identity ──────────────────────────────────────
# A camera restart re-arms its setup self-test, and only its own: the
# whole point of moving fire-once off the shared per-patient row.


@pytest.mark.asyncio
async def test_rearm_clears_the_stamp_on_that_camera_only() -> None:
    store = _store([
        {"id": uuid4(), "identity": "cam_aaaa", "room": "patient_x",
         "online": True, "setup_test_passed_at": "2026-08-22T08:02:17Z"},
        {"id": uuid4(), "identity": "cam_bbbb", "room": "patient_x",
         "online": True, "setup_test_passed_at": "2026-08-22T08:02:17Z"},
    ])
    rearmed = await camera_presence.rearm_setup_test_by_identity(
        _sessionmaker(store), "cam_aaaa",
    )
    assert rearmed is True
    assert store["cameras"][0]["setup_test_passed_at"] is None
    assert store["cameras"][1]["setup_test_passed_at"] is not None
    assert store["commits"] == 1


@pytest.mark.asyncio
async def test_rearm_leaves_online_alone() -> None:
    """The re-arm is identity-scoped like the offline write, so a typo that
    let it fall through to that branch would silently mark a rejoining
    camera offline."""
    store = _store([
        {"id": uuid4(), "identity": "cam_aaaa", "room": "patient_x",
         "online": True, "setup_test_passed_at": "2026-08-22T08:02:17Z"},
    ])
    await camera_presence.rearm_setup_test_by_identity(
        _sessionmaker(store), "cam_aaaa",
    )
    assert store["cameras"][0]["online"] is True


@pytest.mark.asyncio
async def test_rearm_on_camera_that_never_passed_is_a_no_op() -> None:
    """No stamp means no Realtime echo — clients re-render off every
    cameras UPDATE, and a camera reconnects far more often than it tests."""
    store = _store([
        {"id": uuid4(), "identity": "cam_aaaa", "room": "patient_x",
         "online": True, "setup_test_passed_at": None},
    ])
    rearmed = await camera_presence.rearm_setup_test_by_identity(
        _sessionmaker(store), "cam_aaaa",
    )
    assert rearmed is False
