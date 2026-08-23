"""Provisioning must honour `service_kind`, or multi-tenancy is dead code.

The backend has everything it needs to scope a hosted service correctly:
`_BRIDGE_CAM_WHERE`, the `multi_tenant` flag, and the rule that a service
binding is never rewritten. All of it is reached only when the gateways
row has `multi_tenant = true`, which is set only when the caller declares
`service_kind: "camera-bridge"`.

`provision-camera` accepted that field -- `GatewayProvisionRequest`
inherits it from `GatewayEnrollRequest` -- and then dropped it on the
floor, calling `_enroll_gateway_token` without it. So a bridge that had
never called `/gateway/enroll` first was enrolled as a single-tenant BOX,
and every subsequent provision for a different customer REBOUND the whole
bridge to that customer, taking the previous customer's sensors off the
roster silently.

The existing multi-tenancy tests all passed because they POST
`/gateway/enroll` with the service_kind first and provision second. That
is the one order the bridge never used.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_gateway_code, mint_test_jwt

_PATIENT = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
            "relation": "parent"}
_TOKEN = "gw-service-kind-bridge"


async def _account(client: AsyncClient, auth_user, name: str):
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": name})
    return headers, email, uid


async def _provision(client, email, room, *, service_kind=None, headers=None):
    # Binding takes a code the caregiver generated (2026-08-23).
    body = {"room": room,
            "pairing_code": await mint_gateway_code(client, headers)}
    if service_kind is not None:
        body["service_kind"] = service_kind
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {_TOKEN}"}, json=body)
    assert r.status_code == 201, r.text
    return r.json()["cameras"][0]


@pytest.mark.asyncio
async def test_provisioning_as_a_bridge_marks_the_row_multi_tenant(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Declared at provision, not only at enrol.

    A bridge's FIRST contact with the backend is a provision — an
    installer typing an email into its page — so if that path drops the
    declaration the row is born wrong and only a later enrol can repair
    it.
    """
    headers, email, _uid = await _account(client, auth_user, "Bridge Co")
    await _provision(client, email, "Living Room", headers=headers, service_kind="camera-bridge")

    multi = (await session.execute(
        text("SELECT multi_tenant FROM public.gateways "
             " WHERE token_hash = encode(digest(:t, 'sha256'), 'hex')"),
        {"t": _TOKEN},
    )).scalar_one()
    assert multi is True


@pytest.mark.asyncio
async def test_a_box_provisioning_stays_single_tenant(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Declaring nothing must keep today's behaviour exactly.

    A mini-PC in one home relies on being rebound when it is moved to a
    different house, and on the `gateway_id IS NULL` fallback that 4 of
    the 5 gateway cameras in production still depend on.
    """
    headers, email, _uid = await _account(client, auth_user, "Box Co")
    await _provision(client, email, "Living Room", headers=headers)

    multi = (await session.execute(
        text("SELECT multi_tenant FROM public.gateways "
             " WHERE token_hash = encode(digest(:t, 'sha256'), 'hex')"),
        {"t": _TOKEN},
    )).scalar_one()
    assert multi is False
