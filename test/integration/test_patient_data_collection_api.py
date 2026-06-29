"""Task 2 — expose data_collection on the patient API.

The PATCH endpoint accepts ``data_collection_enabled`` and STAMPS
``data_collection_decided_at`` whenever the flag is present in the
payload — both "Share" (true) and "Not now" (false) record a decision,
so the first-open consent banner never re-nags. GET/list responses
surface both fields. ``decided_at`` itself is server-managed: a client
cannot set it directly (``extra="forbid"`` on the update model).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from test.integration.test_patients import _FULL_BODY, _provision


async def _create_patient(client: AsyncClient, headers: dict[str, str]) -> dict:
    r = await client.post("/v1/patients", headers=headers, json=_FULL_BODY)
    assert r.status_code == 201
    return r.json()


@pytest.mark.asyncio
async def test_get_exposes_data_collection_fields_defaulting_off(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    r = await client.get(f"/v1/patients/{created['id']}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["data_collection_enabled"] is False
    assert body["data_collection_decided_at"] is None


@pytest.mark.asyncio
async def test_patch_enable_stamps_decided_at(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    r = await client.patch(
        f"/v1/patients/{created['id']}",
        headers=headers,
        json={"data_collection_enabled": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data_collection_enabled"] is True
    assert body["data_collection_decided_at"] is not None


@pytest.mark.asyncio
async def test_patch_not_now_records_decision(
    client: AsyncClient, auth_user
) -> None:
    """"Not now" (enabled=false) still stamps decided_at so the banner
    won't re-nag the next time the patient is opened."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    r = await client.patch(
        f"/v1/patients/{created['id']}",
        headers=headers,
        json={"data_collection_enabled": False},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["data_collection_enabled"] is False
    assert body["data_collection_decided_at"] is not None


@pytest.mark.asyncio
async def test_patch_cannot_set_decided_at_directly(
    client: AsyncClient, auth_user
) -> None:
    """decided_at is server-stamped, not client-writable — extra keys
    are rejected by the update model."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    r = await client.patch(
        f"/v1/patients/{created['id']}",
        headers=headers,
        json={"data_collection_decided_at": "2020-01-01T00:00:00Z"},
    )
    assert r.status_code == 422
