"""Integration tests for POST /v1/feedback.

Exercises the same path the iOS Settings → Send feedback sheet hits:
authenticated caregiver, persist the row (source of truth), then a
best-effort founder email that must NOT fail the request.

Mirrors test/integration/test_events.py for fixture usage (auth_user /
auth_user_factory create real auth.users rows; mint_test_jwt signs a
local HS256 token). Assumes a local Supabase is running and the schema
delta has been applied (scripts/apply_feedback_delta.sql).

Coverage:
- a valid submit returns 200 / status="received" and persists a row
- a wholly-empty submission is rejected with 422 (schema validator)
- rating out of 1..5 is rejected with 422
- a FORCED email failure still returns success and still persists the
  row (best-effort email — differs from /v1/support)
- the email no-op path (no Resend key) still returns success
- another caregiver's auth cannot see this caregiver's feedback (RLS
  is enforced by the owner check; cross-caregiver INSERT scoping)
"""

from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services import email
from test.conftest import mint_test_jwt


async def _provision(client: AsyncClient, auth_user) -> tuple[dict[str, str], UUID]:
    uid, addr = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, addr)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test CG"})
    return headers, uid


async def _count_feedback(session: AsyncSession, caregiver_id: UUID) -> int:
    row = await session.execute(
        text("select count(*) from feedback where caregiver_id = :cid"),
        {"cid": str(caregiver_id)},
    )
    return int(row.scalar_one())


# ─── Happy path ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_feedback_persists_and_returns_received(
    client: AsyncClient, auth_user, session
) -> None:
    headers, uid = await _provision(client, auth_user)

    r = await client.post(
        "/v1/feedback",
        headers=headers,
        json={
            "rating": 5,
            "whats_working": "Alerts arrive fast.",
            "whats_frustrating": "Wish I could snooze.",
            "category": "alerts",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "received"
    # id is a valid UUID echoed back
    fid = UUID(body["id"])

    # Row is actually persisted and matches what we sent.
    assert await _count_feedback(session, uid) == 1
    row = (
        await session.execute(
            text(
                "select rating, whats_working, whats_frustrating, category, caregiver_id"
                " from feedback where id = :id"
            ),
            {"id": str(fid)},
        )
    ).one()
    assert row.rating == 5
    assert row.whats_working == "Alerts arrive fast."
    assert row.whats_frustrating == "Wish I could snooze."
    assert row.category == "alerts"
    assert UUID(str(row.caregiver_id)) == uid


@pytest.mark.asyncio
async def test_submit_text_only_no_rating_persists(
    client: AsyncClient, auth_user, session
) -> None:
    headers, uid = await _provision(client, auth_user)
    r = await client.post(
        "/v1/feedback",
        headers=headers,
        json={"whats_frustrating": "It crashed once."},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "received"
    assert await _count_feedback(session, uid) == 1


# ─── Validation ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_empty_rejected_422(
    client: AsyncClient, auth_user, session
) -> None:
    headers, uid = await _provision(client, auth_user)
    r = await client.post("/v1/feedback", headers=headers, json={})
    assert r.status_code == 422
    # Nothing persisted.
    assert await _count_feedback(session, uid) == 0


@pytest.mark.asyncio
async def test_submit_rating_out_of_range_rejected_422(
    client: AsyncClient, auth_user, session
) -> None:
    headers, uid = await _provision(client, auth_user)
    r = await client.post(
        "/v1/feedback", headers=headers, json={"rating": 9, "whats_working": "x"}
    )
    assert r.status_code == 422
    assert await _count_feedback(session, uid) == 0


@pytest.mark.asyncio
async def test_submit_requires_auth(client: AsyncClient) -> None:
    r = await client.post("/v1/feedback", json={"rating": 4})
    assert r.status_code == 401


# ─── Best-effort email ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_email_failure_still_succeeds_and_persists(
    client: AsyncClient, auth_user, session, monkeypatch
) -> None:
    """The differentiator from /v1/support: if the founder-notification
    email raises, the request still returns 200 and the row stays."""
    headers, uid = await _provision(client, auth_user)

    async def _boom(**_kwargs):
        raise RuntimeError("resend exploded")

    monkeypatch.setattr(email, "send_feedback_email", _boom)

    r = await client.post(
        "/v1/feedback",
        headers=headers,
        json={"rating": 2, "whats_frustrating": "Too many false alarms."},
    )
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "received"
    # The row committed BEFORE the email attempt, so it survives the failure.
    assert await _count_feedback(session, uid) == 1


@pytest.mark.asyncio
async def test_email_noop_path_still_succeeds(
    client: AsyncClient, auth_user, session, monkeypatch
) -> None:
    """When Resend isn't configured, send_feedback_email no-ops (returns
    a status string) rather than raising — the request still succeeds."""
    headers, uid = await _provision(client, auth_user)

    async def _noop(**_kwargs):
        return "Email isn't configured on the server yet (no Resend key)."

    monkeypatch.setattr(email, "send_feedback_email", _noop)

    r = await client.post("/v1/feedback", headers=headers, json={"rating": 4})
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "received"
    assert await _count_feedback(session, uid) == 1


# ─── Cross-caregiver isolation ────────────────────────────────────


@pytest.mark.asyncio
async def test_other_caregiver_does_not_see_my_feedback(
    client: AsyncClient, auth_user_factory, session
) -> None:
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})

    r = await client.post("/v1/feedback", headers=h_a, json={"rating": 5})
    assert r.status_code == 200

    # A has one row; B has none. (Each row is scoped by caregiver_id.)
    assert await _count_feedback(session, a_uid) == 1
    assert await _count_feedback(session, b_uid) == 0
