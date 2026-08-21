"""A bridge serves MANY customers. A gateway box serves one.

The mini-PC model is one box, one home, one account, and `gateways` binds
a token to a single `caregiver_id` to match. The Warner bridge is our own
hosted service: it fronts the vendor cloud for every customer at once and
replicates as demand grows. It is not scoped to anybody.

Run it on the box model and provisioning a camera for a second customer
REBINDS the whole bridge to them — the first customer's cameras vanish
from the roster and stop publishing, with no error anywhere.

So a service declares its kind at enrolment. A `camera-bridge` is scoped
strictly by the cameras it owns (`cameras.gateway_id`), across every
account, and its caregiver binding is never rewritten. That scoping is
NARROWER than a box's, which is why letting the client declare it is
safe: it can only ever see cameras its own provisioning created.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.ingest.pi.auth import hash_ingest_token
from test.conftest import mint_test_jwt

_BODY = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
         "relation": "parent"}
BRIDGE = "bridge-token-MULTITENANT"


async def _caregiver(client: AsyncClient, uid, email) -> tuple[dict, str]:
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    p = (await client.post("/v1/patients", headers=headers, json=_BODY)).json()
    return headers, p["id"]


async def _enroll(client: AsyncClient, email: str, kind: str | None = None):
    body = {"email": email}
    if kind:
        body["service_kind"] = kind
    return await client.post("/v1/gateway/enroll",
                             headers={"Authorization": f"Bearer {BRIDGE}"},
                             json=body)


async def _provision(client: AsyncClient, email: str, headers, patient_id,
                     room: str) -> str:
    r = await client.post("/v1/gateway/provision-camera",
                          headers={"Authorization": f"Bearer {BRIDGE}"},
                          json={"email": email, "room": room})
    assert r.status_code == 201, r.text
    camera_id = r.json()["cameras"][0]["camera_id"]
    a = await client.post(f"/v1/cameras/{camera_id}/coverage", headers=headers,
                          json={"primary_patient_id": patient_id})
    assert a.status_code == 200, a.text
    return camera_id


async def _roster_ids(client: AsyncClient) -> set[str]:
    r = await client.get("/v1/gateway/roster",
                         headers={"Authorization": f"Bearer {BRIDGE}"})
    assert r.status_code == 200, r.text
    return {c["camera_id"] for c in r.json()["cameras"]}


@pytest.mark.asyncio
async def test_a_bridge_keeps_cameras_from_every_customer(
    client: AsyncClient, auth_user_factory, session: AsyncSession,
) -> None:
    """The whole point. Provisioning for customer B must not evict A."""
    a = await auth_user_factory()
    b = await auth_user_factory()
    h_a, p_a = await _caregiver(client, *a)
    h_b, p_b = await _caregiver(client, *b)
    await _enroll(client, a[1], kind="camera-bridge")

    cam_a = await _provision(client, a[1], h_a, p_a, "A Living Room")
    assert cam_a in await _roster_ids(client)

    cam_b = await _provision(client, b[1], h_b, p_b, "B Bedroom")
    ids = await _roster_ids(client)
    assert cam_b in ids, "the new customer's camera must appear"
    assert cam_a in ids, (
        "the FIRST customer's camera vanished — the bridge was rebound to "
        "whoever was provisioned last"
    )


@pytest.mark.asyncio
async def test_a_bridge_is_not_rebound_by_provisioning(
    client: AsyncClient, auth_user_factory, session: AsyncSession,
) -> None:
    a = await auth_user_factory()
    b = await auth_user_factory()
    h_a, p_a = await _caregiver(client, *a)
    h_b, p_b = await _caregiver(client, *b)
    await _enroll(client, a[1], kind="camera-bridge")
    await _provision(client, a[1], h_a, p_a, "A Room")
    before = (await session.execute(
        text("SELECT caregiver_id FROM public.gateways WHERE token_hash = :h"),
        {"h": hash_ingest_token(BRIDGE)})).scalar()

    await _provision(client, b[1], h_b, p_b, "B Room")
    after = (await session.execute(
        text("SELECT caregiver_id FROM public.gateways WHERE token_hash = :h"),
        {"h": hash_ingest_token(BRIDGE)})).scalar()
    assert before == after, "a bridge's account binding must never be rewritten"


@pytest.mark.asyncio
async def test_a_bridge_sees_only_cameras_it_provisioned(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Narrower than a box, not wider. A bridge gets the cameras carrying
    its own gateway_id and nothing else — including none of the older
    rows that predate ownership and have gateway_id NULL."""
    headers, patient_id = await _caregiver(client, *auth_user)
    await _enroll(client, auth_user[1], kind="camera-bridge")
    mine = await _provision(client, auth_user[1], headers, patient_id, "Mine")

    orphan = (await session.execute(
        text("SELECT id FROM public.cameras WHERE source_type = 'gateway' "
             "  AND gateway_id IS NULL LIMIT 1"))).scalar()
    if orphan is not None:
        assert str(orphan) not in await _roster_ids(client)
    assert mine in await _roster_ids(client)


@pytest.mark.asyncio
async def test_a_plain_box_still_gets_the_unowned_fallback(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Four of five gateway cameras in production still have gateway_id
    NULL. Removing that fallback for everyone would take them dark, so a
    service that does NOT declare itself a bridge keeps the old rule."""
    headers, patient_id = await _caregiver(client, *auth_user)
    r = await client.post("/v1/gateway/enroll",
                          headers={"Authorization": "Bearer plain-box-token"},
                          json={"email": auth_user[1]})
    assert r.status_code == 200, r.text
    cam = (await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": "Bearer plain-box-token"},
        json={"email": auth_user[1], "room": "Box Room"})).json()["cameras"][0]
    await client.post(f"/v1/cameras/{cam['camera_id']}/coverage",
                      headers=headers, json={"primary_patient_id": patient_id})
    await session.execute(
        text("UPDATE public.cameras SET gateway_id = NULL WHERE id = :c"),
        {"c": cam["camera_id"]})
    await session.commit()

    r = await client.get("/v1/gateway/roster",
                         headers={"Authorization": "Bearer plain-box-token"})
    assert cam["camera_id"] in {c["camera_id"] for c in r.json()["cameras"]}
