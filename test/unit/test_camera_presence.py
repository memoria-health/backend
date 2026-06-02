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
