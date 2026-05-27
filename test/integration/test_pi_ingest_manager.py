"""Integration tests for `PiIngestManager`.

Uses the real local Supabase (per the project's "integration tests
hit a real DB" convention in conftest.py) so the SQLAlchemy session
factory, ORM models, and DB-side CHECK constraints are all
exercised exactly as they will be in production. The LiveKit
publisher itself is replaced with a fake coroutine factory that
records calls; the manager's lifecycle, reconciliation, and backoff
logic is what's under test, not the publisher's internals.

Tests verify temporal ordering, not just final state: we drive the
manager's poll loop explicitly via `reconcile_once` and observe the
relay-start event sequence in real time.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_sessionmaker
from src.services.ingest.pi.manager import PiIngestManager
from src.services.ingest.pi.relay_publisher import RelayConfig
from src.settings import Settings
from test.conftest import mint_test_jwt

# Silence the manager's INFO chatter during test runs.
logging.getLogger("src.services.ingest.pi.manager").setLevel(logging.WARNING)


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


# ─── helpers ─────────────────────────────────────────────────────


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _add_pi_camera(
    client: AsyncClient, headers: dict[str, str], patient_id: str, *,
    room: str, pi_stream_url: str,
) -> str:
    """Provision a Pi camera via the public endpoint and return its id."""
    r = await client.post(
        f"/v1/patients/{patient_id}/cameras/pi-provision",
        headers=headers,
        json={"room": room, "pi_stream_url": pi_stream_url},
    )
    assert r.status_code == 201, r.text
    return r.json()["camera_id"]


class _FakeRelay:
    """Records each relay invocation. Each call blocks on a per-call
    event so tests can simulate the relay running for as long as
    they need, then either let it finish cleanly (event set) or
    raise to exercise the manager's failure path.
    """

    def __init__(self) -> None:
        self.started: list[RelayConfig] = []
        # One asyncio.Event per started relay so tests can release
        # them individually.
        self._release_events: list[asyncio.Event] = []
        self._fail_next: bool = False

    def fail_next(self) -> None:
        self._fail_next = True

    def release_all(self) -> None:
        for ev in self._release_events:
            ev.set()

    async def __call__(self, config: RelayConfig, settings: Settings) -> None:
        self.started.append(config)
        release = asyncio.Event()
        self._release_events.append(release)
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError("simulated relay failure")
        await release.wait()


@pytest_asyncio.fixture
async def fake_relay() -> _FakeRelay:
    return _FakeRelay()


@pytest_asyncio.fixture
async def manager(fake_relay: _FakeRelay) -> AsyncIterator[PiIngestManager]:
    """Spin up a manager with the fake relay and the real session
    factory. Yields started; cleans up on exit.
    """
    from src.settings import get_settings
    settings = get_settings()
    session_factory = get_sessionmaker()
    m = PiIngestManager(
        settings,
        session_factory,
        relay_coroutine_factory=fake_relay,
        # Long poll interval so the test drives reconciliation by
        # calling reconcile_once explicitly; the loop should not fire
        # during the test window.
        db_poll_seconds=3600.0,
    )
    yield m
    await m.stop()


# ─── tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_manager_starts_with_no_relays_when_db_has_no_pi_cameras(
    client: AsyncClient, auth_user, manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    """Empty DB → manager.start() returns and active_camera_ids is empty."""
    await manager.start()
    # Give event loop a chance to schedule any spurious spawns.
    await asyncio.sleep(0.05)
    assert manager.active_camera_ids() == set()
    assert fake_relay.started == []


@pytest.mark.asyncio
async def test_manager_starts_one_relay_per_pi_camera(
    client: AsyncClient, auth_user, manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid_a = await _add_pi_camera(
        client, headers, pid, room="Living", pi_stream_url="http://1.1.1.1:8/s.mjpg",
    )
    cid_b = await _add_pi_camera(
        client, headers, pid, room="Kitchen", pi_stream_url="http://2.2.2.2:8/s.mjpg",
    )

    await manager.start()
    # `start()` does an immediate reconcile so relays are spawned
    # synchronously; one event-loop tick is enough for the tasks to
    # be created and visible in active_camera_ids.
    await asyncio.sleep(0.05)

    active = manager.active_camera_ids()
    assert active == {_uuid(cid_a), _uuid(cid_b)}
    assert len(fake_relay.started) == 2
    urls = {c.pi_stream_url for c in fake_relay.started}
    assert urls == {"http://1.1.1.1:8/s.mjpg", "http://2.2.2.2:8/s.mjpg"}


@pytest.mark.asyncio
async def test_manager_ignores_iphone_cameras(
    client: AsyncClient, auth_user, session: AsyncSession,
    manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    """Inserting an iPhone-source camera must not spawn a relay."""
    headers, pid = await _setup_patient(client, auth_user)
    # Mint an iPhone pairing-code row (the existing pairing endpoint).
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers,
        json={"room": "iPhone Cam", "icon_name": "tv"},
    )
    assert r.status_code == 200

    await manager.start()
    await asyncio.sleep(0.05)
    assert manager.active_camera_ids() == set()
    assert fake_relay.started == []


@pytest.mark.asyncio
async def test_manager_picks_up_new_pi_camera_on_reconcile(
    client: AsyncClient, auth_user, manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    """A Pi camera added to the DB AFTER start() must be picked up
    on the next reconciliation cycle.
    """
    headers, pid = await _setup_patient(client, auth_user)
    await manager.start()
    await asyncio.sleep(0.05)
    assert manager.active_camera_ids() == set()

    cid = await _add_pi_camera(
        client, headers, pid, room="Bedroom",
        pi_stream_url="http://3.3.3.3:8/s.mjpg",
    )
    # Drive reconciliation explicitly (avoids racing the poll timer).
    await manager.reconcile_once()
    await asyncio.sleep(0.05)

    assert manager.active_camera_ids() == {_uuid(cid)}
    assert len(fake_relay.started) == 1
    assert fake_relay.started[0].pi_stream_url == "http://3.3.3.3:8/s.mjpg"


@pytest.mark.asyncio
async def test_manager_cancels_relay_for_deleted_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
    manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    """Deleting a Pi camera row must cancel its relay task on the
    next reconciliation.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid = await _add_pi_camera(
        client, headers, pid, room="Living", pi_stream_url="http://4.4.4.4:8/s.mjpg",
    )
    await manager.start()
    await asyncio.sleep(0.05)
    assert manager.active_camera_ids() == {_uuid(cid)}

    # Hard delete: the DELETE endpoint, then reconcile.
    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    assert r.status_code == 204
    await manager.reconcile_once()
    # Cancellation propagates within one event-loop turn, but the
    # done-callback that removes the entry from `_relays` runs on
    # the next turn. A small sleep covers both.
    await asyncio.sleep(0.1)

    assert manager.active_camera_ids() == set()


@pytest.mark.asyncio
async def test_manager_retries_failed_relay_with_backoff(
    client: AsyncClient, auth_user, manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    """When a relay raises, the manager backs off and starts a fresh
    one. We override the initial backoff to keep the test fast.
    """
    from src.services.ingest.pi import manager as manager_module

    headers, pid = await _setup_patient(client, auth_user)
    cid = await _add_pi_camera(
        client, headers, pid, room="Living", pi_stream_url="http://5.5.5.5:8/s.mjpg",
    )
    fake_relay.fail_next()

    # Patch the backoff floor to 50 ms for the test; the supervisor
    # loop reads this constant on each iteration.
    original_floor = manager_module._RELAY_BACKOFF_INITIAL_S
    manager_module._RELAY_BACKOFF_INITIAL_S = 0.05
    try:
        await manager.start()
        # The first relay raises immediately; the supervisor sleeps
        # ~50 ms, then starts a second relay. We wait long enough
        # for that to be visible.
        await asyncio.sleep(0.2)
        assert len(fake_relay.started) >= 2, (
            f"Manager did not retry after relay failure; "
            f"started={len(fake_relay.started)}"
        )
        # The retried relay is for the same camera.
        assert fake_relay.started[0].camera_id == _uuid(cid)
        assert fake_relay.started[1].camera_id == _uuid(cid)
    finally:
        manager_module._RELAY_BACKOFF_INITIAL_S = original_floor


@pytest.mark.asyncio
async def test_manager_stop_cancels_all_relays(
    client: AsyncClient, auth_user, manager: PiIngestManager, fake_relay: _FakeRelay,
) -> None:
    """stop() must cancel every per-camera relay AND the poll task,
    leaving no orphaned tasks behind.
    """
    headers, pid = await _setup_patient(client, auth_user)
    cid_a = await _add_pi_camera(
        client, headers, pid, room="A", pi_stream_url="http://6.6.6.6:8/s.mjpg",
    )
    cid_b = await _add_pi_camera(
        client, headers, pid, room="B", pi_stream_url="http://7.7.7.7:8/s.mjpg",
    )
    await manager.start()
    await asyncio.sleep(0.05)
    assert manager.active_camera_ids() == {_uuid(cid_a), _uuid(cid_b)}

    await manager.stop()
    # After stop, no tasks should still be running.
    assert manager.active_camera_ids() == set()


@pytest.mark.asyncio
async def test_manager_start_is_not_reentrant(
    manager: PiIngestManager,
) -> None:
    """Calling start() twice must raise rather than silently leaking
    a second poll task.
    """
    await manager.start()
    with pytest.raises(RuntimeError, match="already started"):
        await manager.start()


# ─── small helper for UUID comparison ───────────────────────────


def _uuid(s: str) -> object:
    """The cameras API returns UUIDs as strings; the manager stores
    them as UUID objects. Convert for set comparison.
    """
    from uuid import UUID
    return UUID(s)
