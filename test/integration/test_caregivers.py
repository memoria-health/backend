"""Integration tests for /v1/me.

Hits the FastAPI app via ASGI transport (no network) and asserts
against the local Supabase + Postgres. Requires `supabase start` and
`make db-reset` first.

Every test that exercises a `CaregiverAuth`-protected route uses the
`auth_user_factory` fixture to create a real `auth.users` row first —
without it, the FK on `caregivers.id` rejects the insert.
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
    r = await client.get(
        "/v1/me", headers={"Authorization": "Bearer notreallyajwt"}
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_get_me_returns_404_when_not_provisioned(
    client: AsyncClient, auth_user
) -> None:
    """Brand-new auth user exists but no caregiver row yet → 404."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    r = await client.get("/v1/me", headers=headers)
    assert r.status_code == 404
    assert r.json() == {"detail": "caregiver_not_found"}


@pytest.mark.asyncio
async def test_post_me_provisions_idempotently(
    client: AsyncClient, auth_user
) -> None:
    """POST /v1/me twice for the same auth user returns the same row,
    ignoring the second body."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}

    r1 = await client.post(
        "/v1/me", headers=headers, json={"display_name": "Test User"},
    )
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["id"] == str(uid)
    assert b1["display_name"] == "Test User"
    assert b1["email"] == email
    assert b1["phone"] is None

    r2 = await client.post(
        "/v1/me", headers=headers, json={"display_name": "Different Name"},
    )
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["display_name"] == "Test User"        # not "Different Name"
    assert b2["id"] == b1["id"]
    assert b2["created_at"] == b1["created_at"]


@pytest.mark.asyncio
async def test_get_me_returns_full_row(client: AsyncClient, auth_user) -> None:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post(
        "/v1/me", headers=headers,
        json={"display_name": "GetMeTest", "phone": "+14155550001"},
    )

    r = await client.get("/v1/me", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == str(uid)
    assert body["display_name"] == "GetMeTest"
    assert body["email"] == email
    assert body["phone"] == "+14155550001"
    assert "created_at" in body
    assert "updated_at" in body


@pytest.mark.asyncio
async def test_patch_me_updates_display_name_and_phone(
    client: AsyncClient, auth_user
) -> None:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post(
        "/v1/me", headers=headers, json={"display_name": "PatchTester"},
    )

    r = await client.patch(
        "/v1/me", headers=headers,
        json={"display_name": "Patched", "phone": "+14155551234"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["display_name"] == "Patched"
    assert body["phone"] == "+14155551234"


@pytest.mark.asyncio
async def test_patch_me_partial_leaves_other_fields_untouched(
    client: AsyncClient, auth_user
) -> None:
    """PATCH with only `phone` leaves `display_name` alone."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post(
        "/v1/me", headers=headers, json={"display_name": "Untouched"},
    )

    r = await client.patch(
        "/v1/me", headers=headers, json={"phone": "+14155550002"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["display_name"] == "Untouched"      # not overwritten
    assert body["phone"] == "+14155550002"


@pytest.mark.asyncio
async def test_patch_me_explicit_null_clears_phone(
    client: AsyncClient, auth_user
) -> None:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post(
        "/v1/me", headers=headers,
        json={"display_name": "ClearTest", "phone": "+14155559999"},
    )

    r = await client.patch("/v1/me", headers=headers, json={"phone": None})
    assert r.status_code == 200
    assert r.json()["phone"] is None


@pytest.mark.asyncio
async def test_patch_me_rejects_unknown_fields(
    client: AsyncClient, auth_user
) -> None:
    """`extra="forbid"` on UpdateCaregiverRequest → 422 on email."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post(
        "/v1/me", headers=headers, json={"display_name": "ForbidTester"},
    )

    r = await client.patch(
        "/v1/me", headers=headers, json={"email": "new@example.com"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_me_rejects_invalid_phone_format(
    client: AsyncClient, auth_user
) -> None:
    """Phone must be E.164 (`^\\+[1-9]\\d{1,14}$`)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post(
        "/v1/me", headers=headers, json={"display_name": "PhoneTester"},
    )

    r = await client.patch(
        "/v1/me", headers=headers, json={"phone": "415-555-1234"},     # not E.164
    )
    assert r.status_code == 422
