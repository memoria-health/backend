"""Integration tests for the gateway roster's tunnel_port assignment.

Each gateway box gets a UNIQUE reverse-SSH relay port, assigned lazily by
GET /v1/gateway/roster on the box's first poll (folded into the liveness
UPDATE). Requires `scripts/apply_gateway_delta.sql` AND
`scripts/apply_gateway_tunnel_port_delta.sql` applied to the local DB,
same as the other delta-gated suites.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

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
