"""An installer binds a service and mounts a sensor in ONE step, with a code.

A bridge proved which account it belonged to by presenting a bearer token
of its own invention plus the caregiver's EMAIL. An email is not a secret
— it is on the invoice and in the signature — so anyone who knew one
could bind a token and poll that account's roster: camera ids, room
names, LiveKit rooms, privacy and audio flags. Metadata, not video, but
it is the customer's home either way.

There were also TWO doors into that room. `enroll` bound a token, and
`provision-camera` bound one too, so gating only the first would have
left the second wide open — a fix that looked complete and did nothing.

There is one door now, and it matches what actually happens on site: the
caregiver generates a 5-minute code in the app, the installer types it in
alongside the camera they just picked, and the same call binds the
service and creates the sensor. `/gateway/enroll` is gone; a bridge
re-declares its `service_kind` on the fleet poll it was already making.

The email is optional and is a CONFIRMATION only. The code already names
the account, so treating the email as a second source of truth would just
be a way to bind the right token to the wrong customer.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt

_TTL_MAX = 5 * 60


async def _account(client: AsyncClient, auth_user):
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    return headers, email, uid


async def _mint_code(client: AsyncClient, headers) -> str:
    r = await client.post("/v1/gateway/pairing-code", headers=headers, json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 < body["expires_in_seconds"] <= _TTL_MAX, "code must be short-lived"
    assert body["pairing_code"].isdigit() and len(body["pairing_code"]) == 6
    return body["pairing_code"]


async def _provision(client, token: str, *, code=None, email=None,
                     room="Living Room"):
    body: dict = {"room": room, "service_kind": "camera-bridge"}
    if code is not None:
        body["pairing_code"] = code
    if email is not None:
        body["email"] = email
    return await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {token}"}, json=body)


@pytest.mark.asyncio
async def test_one_call_binds_the_service_and_creates_the_sensor(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The installer's whole interaction: a code and a camera."""
    headers, _email, uid = await _account(client, auth_user)
    code = await _mint_code(client, headers)

    r = await _provision(client, "gw-onestep", code=code)
    assert r.status_code == 201, r.text
    cam = r.json()["cameras"][0]
    assert cam["livekit_room"] == f"account_{str(uid).lower()}"

    bound = (await session.execute(
        text("SELECT caregiver_id, multi_tenant FROM public.gateways "
             " WHERE token_hash = encode(digest(:t,'sha256'),'hex')"),
        {"t": "gw-onestep"},
    )).first()
    assert str(bound[0]) == str(uid)
    assert bound[1] is True, "service_kind declared in the same call"


@pytest.mark.asyncio
async def test_no_code_binds_nothing(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Knowing the customer is not proof of anything."""
    headers, email, _uid = await _account(client, auth_user)
    await _mint_code(client, headers)          # exists, not presented

    r = await _provision(client, "gw-nocode", email=email)
    assert r.status_code in (401, 403), r.text
    rows = (await session.execute(
        text("SELECT 1 FROM public.gateways "
             " WHERE token_hash = encode(digest(:t,'sha256'),'hex')"),
        {"t": "gw-nocode"},
    )).all()
    assert not rows, "a token was bound with no code"


@pytest.mark.asyncio
async def test_a_code_is_single_use(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A code read aloud once must not bind services forever."""
    headers, _email, _uid = await _account(client, auth_user)
    code = await _mint_code(client, headers)

    assert (await _provision(client, "gw-1", code=code,
                             room="Room A")).status_code == 201
    again = await _provision(client, "gw-2", code=code, room="Room B")
    assert again.status_code in (401, 403), again.text


@pytest.mark.asyncio
async def test_an_expired_code_is_refused(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, _email, _uid = await _account(client, auth_user)
    code = await _mint_code(client, headers)
    await session.execute(
        text("UPDATE public.gateway_pairing_codes "
             "   SET expires_at = now() - interval '1 minute'"))
    await session.commit()

    r = await _provision(client, "gw-expired", code=code)
    assert r.status_code in (401, 403), r.text


@pytest.mark.asyncio
async def test_a_bound_service_provisions_again_without_a_code(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A bridge mounts a second sensor in the same home minutes later.

    Making the installer walk back to the caregiver for a second code
    would be the security control becoming the annoyance that gets
    designed around.
    """
    headers, _email, _uid = await _account(client, auth_user)
    code = await _mint_code(client, headers)
    assert (await _provision(client, "gw-two-cams", code=code,
                             room="Living Room")).status_code == 201

    second = await _provision(client, "gw-two-cams", room="Kitchen")
    assert second.status_code == 201, second.text


@pytest.mark.asyncio
async def test_a_mismatched_email_is_refused(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The email is a typo check, never a second source of truth.

    If it disagrees with the code, something is wrong with the install —
    the wrong code was read out, or the wrong customer is on the form —
    and binding to either answer would be a guess.
    """
    headers, _email, _uid = await _account(client, auth_user)
    code = await _mint_code(client, headers)

    r = await _provision(client, "gw-mismatch", code=code,
                         email="someone.else@example.com")
    assert r.status_code in (400, 403), r.text


@pytest.mark.asyncio
async def test_the_enroll_endpoint_is_gone(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """One door. A second bind path is how the first gets bypassed."""
    r = await client.post(
        "/v1/gateway/enroll",
        headers={"Authorization": "Bearer gw-anything"},
        json={"email": "x@example.com", "service_kind": "camera-bridge"})
    assert r.status_code == 404, r.text
