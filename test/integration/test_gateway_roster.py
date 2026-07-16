"""Integration tests for gateway camera -> gateway ownership.

The roster used to fan every one of a caregiver's gateway cameras to every
box. Now each gateway camera is owned by a specific box (cameras.gateway_id),
so a caregiver with more than one gateway gets only its own cameras. These
tests lock in that isolation plus the transitional NULL-owner fallback.

Requires the local DB to have `scripts/apply_gateway_delta.sql` AND
`scripts/apply_gateway_ownership_delta.sql` applied (the gateways table +
cameras.gateway_id column), same as the other delta-gated suites.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.ingest.pi.auth import hash_ingest_token
from test.conftest import mint_test_jwt

_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(
    client: AsyncClient, auth_user,
) -> tuple[dict[str, str], str, str]:
    """Create the caregiver + one patient. Returns (headers, email, patient_id)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, email, p["id"]


async def _provision(client: AsyncClient, email: str, token: str, room: str) -> dict:
    """Provision a gateway camera as the box holding `token`."""
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": email, "room": room},
    )
    assert r.status_code == 201, r.text
    return r.json()


async def _roster_identities(client: AsyncClient, token: str) -> set[str]:
    r = await client.get(
        "/v1/gateway/roster",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    return {c["livekit_identity"] for c in r.json()["cameras"]}


@pytest.mark.asyncio
async def test_provision_stamps_owning_gateway(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A provisioned camera records the box that provisioned it."""
    _, email, _ = await _setup_patient(client, auth_user)
    resp = await _provision(client, email, "gw-token-A", room="Living Room")
    camera_id = resp["cameras"][0]["camera_id"]

    gw_id = (await session.execute(
        text("SELECT id FROM public.gateways WHERE token_hash = :h"),
        {"h": hash_ingest_token("gw-token-A")},
    )).scalar_one()
    stamped = (await session.execute(
        text("SELECT gateway_id FROM public.cameras WHERE id = :id"),
        {"id": camera_id},
    )).scalar_one()
    assert str(stamped) == str(gw_id)


@pytest.mark.asyncio
async def test_roster_isolates_cameras_by_gateway(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Two boxes for the same caregiver each see only their own camera."""
    _, email, _ = await _setup_patient(client, auth_user)
    a = await _provision(client, email, "gw-token-A", room="Room A")
    b = await _provision(client, email, "gw-token-B", room="Room B")
    a_ident = a["cameras"][0]["livekit_identity"]
    b_ident = b["cameras"][0]["livekit_identity"]

    roster_a = await _roster_identities(client, "gw-token-A")
    roster_b = await _roster_identities(client, "gw-token-B")

    assert a_ident in roster_a and b_ident not in roster_a
    assert b_ident in roster_b and a_ident not in roster_b


@pytest.mark.asyncio
async def test_roster_includes_unowned_null_cameras(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A gateway camera with no owner (gateway_id NULL) still reaches every
    box — the transitional fallback so un-backfilled rows keep working."""
    _, email, pid = await _setup_patient(client, auth_user)
    await _provision(client, email, "gw-token-A", room="Owned")

    orphan = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.cameras "
            "(id, patient_id, display_id, room, icon_name, online, created_at, "
            " livekit_identity, livekit_room, source_type, ingest_token_hash, gateway_id) "
            "VALUES (:id, :pid, 99, 'Orphan', 'tv', false, now(), "
            " 'cam_orphan', :room, 'gateway', :tok, NULL)"
        ),
        {"id": str(orphan), "pid": pid, "room": f"patient_{pid}", "tok": "x" * 64},
    )
    await session.commit()

    assert "cam_orphan" in await _roster_identities(client, "gw-token-A")


@pytest.mark.asyncio
async def test_roster_assigns_stable_unique_tunnel_ports(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """First roster poll assigns a unique reverse-SSH port (never the
    reserved 22001); later polls return the SAME port; two boxes never
    collide."""
    _, email, _ = await _setup_patient(client, auth_user)
    await _provision(client, email, "gw-token-A", room="Room A")
    await _provision(client, email, "gw-token-B", room="Room B")

    async def _port(token: str) -> int:
        r = await client.get(
            "/v1/gateway/roster",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        return r.json()["tunnel_port"]

    port_a = await _port("gw-token-A")
    port_b = await _port("gw-token-B")
    assert port_a >= 22002  # 22001 is reserved for the legacy box-01 unit
    assert port_b >= 22002
    assert port_a != port_b
    assert await _port("gw-token-A") == port_a  # stable across polls
