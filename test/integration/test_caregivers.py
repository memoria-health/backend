"""Integration tests for /v1/me.

Hits the FastAPI app via ASGI transport (no network) and asserts
against the local Supabase + Postgres. Requires `supabase start` and
`make db-reset` first.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from test.conftest import mint_test_jwt


@pytest.mark.asyncio
async def test_get_me_without_auth_returns_401(client: AsyncClient) -> None:
    r = await client.get("/v1/me")
    assert r.status_code == 401
    assert r.json() == {"detail": "invalid_auth"}


@pytest.mark.asyncio
async def test_get_me_with_bad_token_returns_401(client: AsyncClient) -> None:
    r = await client.get("/v1/me", headers={"Authorization": "Bearer notreallyajwt"})
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_post_me_provisions_idempotently(client: AsyncClient) -> None:
    """POST /v1/me twice returns the same caregiver row."""
    from uuid import uuid4

    cid = uuid4()
    token = mint_test_jwt(caregiver_id=cid, email=f"prov-{cid}@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    # First call — provisions.
    r1 = await client.post(
        "/v1/me", headers=headers, json={"display_name": "Test User"},
    )
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["id"] == str(cid)
    assert body1["display_name"] == "Test User"

    # Second call — returns existing row, ignores new display_name.
    r2 = await client.post(
        "/v1/me", headers=headers, json={"display_name": "Different Name"},
    )
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["display_name"] == "Test User"        # not "Different Name"
    assert body2["id"] == body1["id"]
    assert body2["created_at"] == body1["created_at"]


@pytest.mark.asyncio
async def test_patch_me_updates_phone(client: AsyncClient) -> None:
    from uuid import uuid4

    cid = uuid4()
    token = mint_test_jwt(caregiver_id=cid, email=f"patch-{cid}@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    await client.post(
        "/v1/me", headers=headers, json={"display_name": "Patch Tester"},
    )

    r = await client.patch(
        "/v1/me", headers=headers, json={"phone": "+14155551234"},
    )
    assert r.status_code == 200
    assert r.json()["phone"] == "+14155551234"


@pytest.mark.asyncio
async def test_patch_me_rejects_unknown_fields(client: AsyncClient) -> None:
    """`extra="forbid"` on UpdateCaregiverRequest should 422 on email."""
    from uuid import uuid4

    cid = uuid4()
    token = mint_test_jwt(caregiver_id=cid, email=f"forbid-{cid}@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    await client.post(
        "/v1/me", headers=headers, json={"display_name": "Forbid Tester"},
    )

    r = await client.patch(
        "/v1/me", headers=headers, json={"email": "new@example.com"},
    )
    assert r.status_code == 422
