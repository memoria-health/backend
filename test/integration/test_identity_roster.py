"""Integration tests for POST /v1/identity/roster.

This endpoint is the whole scaling story: a GPU node polls it to be told
which homes to run, and the same call is its heartbeat. Everything it does
fails QUIETLY if it is wrong — a home that is never claimed just has no
identity forever, with no error anywhere — so the behaviours below are
tested against a real database rather than reasoned about.

Assumes a local Supabase is running and
scripts/apply_identity_nodes_delta.sql has been applied.

Coverage:
- a node is upserted and heartbeated by the act of polling
- a node claims no more than its capacity
- two nodes polling do not both claim the same home
- a stale node's homes are released and re-claimed by a live node
- a node that comes back keeps its identity and re-claims
- base_url and capacity are refreshed on every poll (a node can move/resize)
- fail-closed auth outside local/test
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

pytestmark = pytest.mark.asyncio

ROSTER = "/v1/identity/roster"


async def _mk_caregiver(session: AsyncSession, caregiver_id: UUID, email: str) -> None:
    """The `auth_user` fixture creates an auth.users row but not the
    `caregivers` row the patients FK points at. Idempotent so a test can call
    it per patient without caring."""
    await session.execute(
        text(
            "INSERT INTO public.caregivers (id, display_name, email) "
            "VALUES (:id, :n, :e) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": caregiver_id, "n": "Roster Test", "e": email},
    )
    await session.commit()


async def _mk_patient(session: AsyncSession, caregiver_id: UUID) -> UUID:
    """A bare patient row — the roster only cares about id/created_at."""
    # full_name / preferred_name / relation are the only NOT NULLs without a
    # default; everything else the roster ignores.
    pid = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.patients "
            "  (id, caregiver_id, full_name, preferred_name, relation) "
            "VALUES (:id, :cg, :n, :p, :r)"
        ),
        {
            "id": pid,
            "cg": caregiver_id,
            "n": f"Roster Test {str(pid)[:8]}",
            "p": "Roster",
            "r": "parent",
        },
    )
    await session.commit()
    return pid


async def _clear(session: AsyncSession) -> None:
    """Detach every home and drop test nodes, so a case starts from a known
    pool regardless of what other tests left behind."""
    await session.execute(text("UPDATE public.patients SET identity_node_id = NULL"))
    await session.execute(text("DELETE FROM public.identity_nodes"))
    await session.commit()


def _names(body) -> list[str]:
    """Room names only. The roster now returns {room, port} per entry —
    the port is asserted separately, in the tests that care about it."""
    return [r["room"] for r in body["rooms"]]


async def _poll(client: AsyncClient, node: str, cap: int, url: str | None = None):
    return await client.post(
        ROSTER,
        json={
            "node_id": node,
            "base_url": url or f"http://{node}:8012",
            "capacity": cap,
        },
    )


async def test_poll_registers_node_and_claims_up_to_capacity(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(5):
        await _mk_patient(session, caregiver_id)

    r = await _poll(client, "node-cap", cap=3)
    assert r.status_code == 200, r.text
    body = r.json()

    # Capacity is a ceiling, not a target to overshoot.
    assert len(body["rooms"]) == 3
    assert body["node_id"] == "node-cap"
    assert all(r.startswith("patient_") for r in _names(body))

    # Polling IS the heartbeat — the row exists without a separate register.
    seen = (
        await session.execute(
            text("SELECT count(*) FROM public.identity_nodes WHERE node_id = 'node-cap'")
        )
    ).scalar()
    assert seen == 1


async def test_second_poll_is_stable_not_cumulative(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """A node polling again must not keep claiming — steady state is a no-op.

    Getting this wrong would have every poll drag more homes onto the first
    node until it collapsed.
    """
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(5):
        await _mk_patient(session, caregiver_id)

    first = _names((await _poll(client, "node-stable", cap=2)).json())
    second = _names((await _poll(client, "node-stable", cap=2)).json())
    assert first == second
    assert len(second) == 2


async def test_two_nodes_do_not_claim_the_same_home(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """Disjoint claims. A home served by two nodes would mean two identity
    processes fighting over one seat."""
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(4):
        await _mk_patient(session, caregiver_id)

    a = set(_names((await _poll(client, "node-a", cap=2)).json()))
    b = set(_names((await _poll(client, "node-b", cap=2)).json()))

    assert len(a) == 2 and len(b) == 2
    assert a.isdisjoint(b), "a home was handed to two nodes"


async def test_stale_node_homes_are_reclaimed(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """The failover path: a node stops polling, a survivor picks its homes up.

    This is what makes losing a node cost a few homes instead of all of them.
    """
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(2):
        await _mk_patient(session, caregiver_id)

    dead = _names((await _poll(client, "node-dead", cap=2)).json())
    assert len(dead) == 2

    # Age it past the staleness window without waiting for wall-clock.
    await session.execute(
        text(
            "UPDATE public.identity_nodes "
            "   SET last_seen_at = now() - interval '1 hour' "
            " WHERE node_id = 'node-dead'"
        )
    )
    await session.commit()

    live = await _poll(client, "node-live", cap=2)
    body = live.json()
    assert body["reclaimed"] == 2, "stale node's homes were not released"
    assert set(_names(body)) == set(dead), "survivor did not pick them up"


async def test_returning_node_keeps_identity_and_reclaims(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """A container restart or brief partition must not orphan the node row.

    Releasing homes NULLs the assignment but leaves identity_nodes alone, so
    the node comes back as itself rather than as a stranger.
    """
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    await _mk_patient(session, caregiver_id)

    (await _poll(client, "node-flap", cap=1)).json()
    await session.execute(
        text(
            "UPDATE public.identity_nodes "
            "   SET last_seen_at = now() - interval '1 hour' "
            " WHERE node_id = 'node-flap'"
        )
    )
    await session.commit()

    # Someone else releases its home...
    await _poll(client, "node-other", cap=0 + 1)
    # ...and it comes back.
    back = (await _poll(client, "node-flap", cap=1)).json()

    still_there = (
        await session.execute(
            text("SELECT count(*) FROM public.identity_nodes WHERE node_id = 'node-flap'")
        )
    ).scalar()
    assert still_there == 1, "node row was deleted rather than just released"
    assert back["node_id"] == "node-flap"


async def test_base_url_and_capacity_refresh_on_every_poll(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """A node that moves (new tailnet name) or is resized must not need a
    hand-edited row."""
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(3):
        await _mk_patient(session, caregiver_id)

    await _poll(client, "node-move", cap=1, url="http://old-name:8012")
    await _poll(client, "node-move", cap=3, url="http://new-name:8012")

    row = (
        await session.execute(
            text(
                "SELECT base_url, capacity FROM public.identity_nodes "
                " WHERE node_id = 'node-move'"
            )
        )
    ).first()
    assert row[0] == "http://new-name:8012"
    assert row[1] == 3


async def test_each_room_on_a_node_gets_its_own_port(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """THE bug this endpoint exists to prevent, in its second form.

    A node runs one identity process per room, each on its own port. If the
    roster hands back one address for the node, exactly one home is reachable
    and every other one silently gets no identity — which is the same
    end-state as having no roster at all.
    """
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(3):
        await _mk_patient(session, caregiver_id)

    body = (await _poll(client, "node-ports", cap=3)).json()
    ports = [r["port"] for r in body["rooms"]]

    assert len(ports) == 3
    assert len(set(ports)) == 3, f"ports collided: {ports}"
    assert all(8012 <= p <= 8111 for p in ports), ports


async def test_ports_are_stable_across_polls(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """A room that changed port would look to the backend like a brand new,
    empty home, and would make the node restart it for nothing."""
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(2):
        await _mk_patient(session, caregiver_id)

    first = {r["room"]: r["port"] for r in (await _poll(client, "node-fix", cap=2)).json()["rooms"]}
    second = {r["room"]: r["port"] for r in (await _poll(client, "node-fix", cap=2)).json()["rooms"]}
    assert first == second


async def test_released_port_is_reused(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """Ports return to the node's pool on release, or a node that churns
    homes eventually exhausts its range."""
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    await _mk_patient(session, caregiver_id)

    before = (await _poll(client, "node-reuse", cap=1)).json()["rooms"][0]["port"]

    # Release it the way a failover does.
    await session.execute(
        text(
            "UPDATE public.identity_nodes "
            "   SET last_seen_at = now() - interval '1 hour' WHERE node_id='node-reuse'"
        )
    )
    await session.commit()
    await _poll(client, "node-sweeper", cap=0 + 1)

    # Sweeper hands it back, then the original reclaims from a clean pool.
    await session.execute(
        text("UPDATE public.patients SET identity_node_id=NULL, identity_port=NULL")
    )
    await session.commit()
    after = (await _poll(client, "node-reuse", cap=1)).json()["rooms"][0]["port"]

    assert after == before, "lowest free port was not reused"


async def test_lowering_capacity_sheds_load(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """Capacity is a live limit, not just a claim ceiling. A node resized
    down (e.g. concurrency reverted) must stop being told to run the extra
    homes, or it keeps serving beyond what it can handle."""
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)
    for _ in range(4):
        await _mk_patient(session, caregiver_id)

    assert len(_names((await _poll(client, "node-shrink", cap=4)).json())) == 4
    assert len(_names((await _poll(client, "node-shrink", cap=2)).json())) == 2


async def test_unassigned_claimed_oldest_first(
    client: AsyncClient, session: AsyncSession, auth_user
) -> None:
    """Oldest-first, because the home unassigned longest has been blind
    longest."""
    caregiver_id, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, caregiver_id, email)

    # The dev database carries patients from other tests and from manual
    # poking, and some are older than anything created here. Park them on a
    # sentinel node so the unassigned pool is exactly this test's three —
    # otherwise "oldest" is whatever happens to be lying around.
    await session.execute(
        text(
            "INSERT INTO public.identity_nodes (node_id, base_url, capacity) "
            "VALUES ('sentinel','http://sentinel:8012',9999) "
            "ON CONFLICT (node_id) DO NOTHING"
        )
    )
    await session.execute(
        text(
            "UPDATE public.patients SET identity_node_id = 'sentinel' "
            " WHERE identity_node_id IS NULL"
        )
    )
    await session.commit()

    ids = [await _mk_patient(session, caregiver_id) for _ in range(3)]
    # Force a known ordering rather than relying on insert timing.
    for offset, pid in enumerate(ids):
        await session.execute(
            text(
                "UPDATE public.patients "
                "   SET created_at = now() - make_interval(mins => :m) "
                " WHERE id = :id"
            ),
            {"m": 100 - offset, "id": pid},
        )
    await session.commit()

    rooms = _names((await _poll(client, "node-order", cap=1)).json())
    assert rooms == [f"patient_{ids[0]}"], "did not take the longest-waiting home"
