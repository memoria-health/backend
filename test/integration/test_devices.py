"""Integration tests for `POST /v1/me/devices`.

Hits the FastAPI app via ASGI transport and reaches a real Supabase +
Postgres. The endpoint is RLS-protected by the caregiver JWT; cross-
caregiver writes are blocked at the API layer via `CaregiverAuth`'s
identity-bind, not at the SQL layer (every POST writes its own
caller's UID into `caregiver_id`).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver_device import CaregiverDevice
from test.conftest import mint_test_jwt


async def _provision(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    r = await client.post(
        "/v1/me", headers=headers, json={"display_name": "Pusher"},
    )
    assert r.status_code == 200
    return headers, str(uid)


@pytest.mark.asyncio
async def test_register_device_returns_canonical_row(
    client: AsyncClient,
    auth_user,
    session: AsyncSession,
) -> None:
    headers, caregiver_id = await _provision(client, auth_user)
    r = await client.post(
        "/v1/me/devices",
        headers=headers,
        json={
            "apns_token": "a" * 64,
            "bundle_id": "com.elephantlabs.nemara.caregiver",
            "environment": "sandbox",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["caregiver_id"] == caregiver_id
    assert body["apns_token"] == "a" * 64
    assert body["bundle_id"] == "com.elephantlabs.nemara.caregiver"
    assert body["environment"] == "sandbox"
    assert body["id"]
    assert body["created_at"]
    assert body["updated_at"]

    # Exactly one row landed in the DB.
    rows = (
        await session.execute(
            select(CaregiverDevice).where(CaregiverDevice.apns_token == "a" * 64)
        )
    ).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert str(row.id) == body["id"]
    assert str(row.caregiver_id) == caregiver_id
    assert row.apns_token == "a" * 64
    assert row.bundle_id == "com.elephantlabs.nemara.caregiver"
    assert row.environment == "sandbox"


@pytest.mark.asyncio
async def test_register_device_is_idempotent_on_token(
    client: AsyncClient,
    auth_user,
    session: AsyncSession,
) -> None:
    """A second POST with the same `apns_token` updates the existing row's
    `updated_at` rather than inserting a duplicate. Critical: the caregiver
    app POSTs on every launch, and a fresh row per launch would blow up the
    table."""
    headers, _ = await _provision(client, auth_user)
    token = "b" * 64
    body = {
        "apns_token": token,
        "bundle_id": "com.elephantlabs.nemara.caregiver",
        "environment": "sandbox",
    }
    r1 = await client.post("/v1/me/devices", headers=headers, json=body)
    assert r1.status_code == 200
    first_id = r1.json()["id"]

    r2 = await client.post("/v1/me/devices", headers=headers, json=body)
    assert r2.status_code == 200
    assert r2.json()["id"] == first_id

    rows = (
        await session.execute(
            select(CaregiverDevice).where(CaregiverDevice.apns_token == token)
        )
    ).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_register_device_updates_environment_and_bundle(
    client: AsyncClient,
    auth_user,
) -> None:
    """A device that switches from sandbox to production (TestFlight ->
    App Store) re-POSTs with the new environment. The row must reflect
    the latest values, not the original ones."""
    headers, _ = await _provision(client, auth_user)
    token = "c" * 64
    await client.post(
        "/v1/me/devices",
        headers=headers,
        json={
            "apns_token": token,
            "bundle_id": "com.elephantlabs.nemara.caregiver",
            "environment": "sandbox",
        },
    )
    r2 = await client.post(
        "/v1/me/devices",
        headers=headers,
        json={
            "apns_token": token,
            "bundle_id": "com.elephantlabs.nemara.caregiver.prod",
            "environment": "production",
        },
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["bundle_id"] == "com.elephantlabs.nemara.caregiver.prod"
    assert body["environment"] == "production"


@pytest.mark.asyncio
async def test_token_ownership_transfers_between_caregivers(
    client: AsyncClient,
    auth_user_factory,
    session: AsyncSession,
) -> None:
    """If a device signs out as caregiver A and back in as caregiver B
    on the same iOS install, the same `apns_token` re-registers — and
    must end up owned by B. Sending future pushes to A's events would
    leak patient data."""
    cg_a = await auth_user_factory()
    cg_b = await auth_user_factory()

    for cg in (cg_a, cg_b):
        uid, email = cg
        headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
        await client.post("/v1/me", headers=headers, json={"display_name": "x"})

    token = "d" * 64
    body = {
        "apns_token": token,
        "bundle_id": "com.elephantlabs.nemara.caregiver",
        "environment": "sandbox",
    }
    h_a = {"Authorization": f"Bearer {mint_test_jwt(cg_a[0], cg_a[1])}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(cg_b[0], cg_b[1])}"}

    r_a = await client.post("/v1/me/devices", headers=h_a, json=body)
    assert r_a.status_code == 200
    assert r_a.json()["caregiver_id"] == str(cg_a[0])

    r_b = await client.post("/v1/me/devices", headers=h_b, json=body)
    assert r_b.status_code == 200
    assert r_b.json()["caregiver_id"] == str(cg_b[0])

    rows = (
        await session.execute(
            select(CaregiverDevice).where(CaregiverDevice.apns_token == token)
        )
    ).scalars().all()
    assert len(rows) == 1
    assert str(rows[0].caregiver_id) == str(cg_b[0])


@pytest.mark.asyncio
async def test_register_device_requires_auth(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/me/devices",
        json={
            "apns_token": "x" * 64,
            "bundle_id": "com.elephantlabs.nemara.caregiver",
            "environment": "sandbox",
        },
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_register_device_rejects_unknown_environment(
    client: AsyncClient, auth_user
) -> None:
    headers, _ = await _provision(client, auth_user)
    r = await client.post(
        "/v1/me/devices",
        headers=headers,
        json={
            "apns_token": "e" * 64,
            "bundle_id": "com.elephantlabs.nemara.caregiver",
            "environment": "neither",  # not in Literal['sandbox','production']
        },
    )
    assert r.status_code == 422
