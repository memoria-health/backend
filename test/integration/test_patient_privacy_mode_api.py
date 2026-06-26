"""Privacy mode — expose privacy_mode_enabled on the patient API.

The PATCH endpoint accepts ``privacy_mode_enabled`` (true OR false).
Unlike data-collection consent, flipping privacy does NOT stamp a
decision timestamp and — critically — does NOT mutate
``data_collection_enabled`` or ``data_collection_decided_at``: privacy
SUSPENDS data collection, it never resets the stored consent choice.
GET/list responses surface the flag, defaulting off.
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
async def test_get_exposes_privacy_mode_defaulting_off(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    r = await client.get(f"/v1/patients/{created['id']}", headers=headers)
    assert r.status_code == 200
    assert r.json()["privacy_mode_enabled"] is False


@pytest.mark.asyncio
async def test_patch_privacy_mode_toggles_and_does_not_touch_data_collection(
    client: AsyncClient, auth_user
) -> None:
    """Privacy is a live override: turning it on must leave the stored
    data-collection consent (flag + decided_at) exactly as it was."""
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    # Establish a real consent decision first (stamps decided_at).
    r = await client.patch(
        f"/v1/patients/{created['id']}",
        headers=headers,
        json={"data_collection_enabled": True},
    )
    assert r.status_code == 200
    decided_at = r.json()["data_collection_decided_at"]
    assert decided_at is not None

    # Now flip privacy ON. The consent decision must be untouched.
    r = await client.patch(
        f"/v1/patients/{created['id']}",
        headers=headers,
        json={"privacy_mode_enabled": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["privacy_mode_enabled"] is True
    assert body["data_collection_enabled"] is True  # suspended, not reset
    assert body["data_collection_decided_at"] == decided_at  # unchanged


@pytest.mark.asyncio
async def test_patch_privacy_mode_off(client: AsyncClient, auth_user) -> None:
    headers = await _provision(client, auth_user)
    created = await _create_patient(client, headers)

    for value in (True, False):
        r = await client.patch(
            f"/v1/patients/{created['id']}",
            headers=headers,
            json={"privacy_mode_enabled": value},
        )
        assert r.status_code == 200
        assert r.json()["privacy_mode_enabled"] is value
