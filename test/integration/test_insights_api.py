"""Integration: /v1/insights — the caregiver dashboard's read API.

These tests exist for one property above all: TENANT ISOLATION IS
SERVER-DERIVED. The predecessor (web app → nemara_readonly with
BYPASSRLS) enforced scope with a hand-written `= any($tenant)` clause
per query and a client-assembled tenant array; the failure mode was a
silent cross-tenant read. Here the patient set comes from
`list_for_caregiver` per request, and a foreign patient_id must 404
without disclosing existence.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from test.conftest import mint_test_jwt

_PATIENT_BODY = {
    "full_name": "Insight Test Patient",
    "preferred_name": "Insight",
    "relation": "parent",
}


async def _caregiver_with_patient(client, auth_user_factory):
    uid, email = await auth_user_factory()
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    r = await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)
    assert r.status_code == 201
    return headers, r.json()["id"]


@pytest.mark.asyncio
async def test_insights_require_auth(client: AsyncClient) -> None:
    for path in (
        "/v1/insights/patients",
        "/v1/insights/trend-facts",
        "/v1/insights/activity-trend?patient_id=x",
    ):
        r = await client.get(path)
        assert r.status_code in (401, 403), path


@pytest.mark.asyncio
async def test_scope_is_server_derived_not_client_supplied(
    client: AsyncClient, auth_user_factory,
) -> None:
    h_a, pid_a = await _caregiver_with_patient(client, auth_user_factory)
    h_b, pid_b = await _caregiver_with_patient(client, auth_user_factory)

    # Each caregiver's /patients lists exactly their own.
    for h, own, foreign in ((h_a, pid_a, pid_b), (h_b, pid_b, pid_a)):
        ids = {p["id"] for p in (await client.get(
            "/v1/insights/patients", headers=h)).json()}
        assert own in ids and foreign not in ids

    # A foreign patient_id 404s on every patient-scoped endpoint —
    # not 200-empty (silent cross-tenant read) and not 403 (existence
    # disclosure).
    for path in (
        f"/v1/insights/activity-trend?patient_id={pid_b}",
        f"/v1/insights/detection-counts?patient_id={pid_b}",
        f"/v1/insights/uptime?patient_id={pid_b}",
        f"/v1/insights/event-timeline?patient_id={pid_b}",
        f"/v1/insights/event-leaderboard?patient_id={pid_b}",
        f"/v1/insights/trend-facts?patient_id={pid_b}",
        f"/v1/insights/offline-cameras?patient_id={pid_b}",
    ):
        r = await client.get(path, headers=h_a)
        assert r.status_code == 404, f"{path} -> {r.status_code}"

    # Garbage patient_id is a 404 too, never a 500.
    r = await client.get(
        "/v1/insights/activity-trend?patient_id=not-a-uuid", headers=h_a)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_shapes_match_web_data_layer(
    client: AsyncClient, auth_user_factory,
) -> None:
    """Empty-data shapes — the web db.ts contract the UI renders."""
    h, pid = await _caregiver_with_patient(client, auth_user_factory)

    r = (await client.get(
        f"/v1/insights/detection-counts?patient_id={pid}", headers=h)).json()
    assert r == {"types": [], "rows": []}

    r = (await client.get("/v1/insights/trend-facts", headers=h)).json()
    assert set(r) == {"events", "nightEvents", "activity", "offlineCameras"}
    assert set(r["events"]) == {"cur", "prev"}

    assert (await client.get(
        f"/v1/insights/activity-trend?patient_id={pid}", headers=h)).json() == []
    assert (await client.get(
        f"/v1/insights/uptime?patient_id={pid}", headers=h)).json() == []
    lb = (await client.get(
        "/v1/insights/event-leaderboard", headers=h)).json()
    assert [e["id"] for e in lb] == [pid]
    assert lb[0]["events"] == 0 and lb[0]["byCategory"] == {}
    assert (await client.get(
        "/v1/insights/busiest-patient", headers=h)).json() is None

    # days clamp: absurd values must not 500.
    assert (await client.get(
        f"/v1/insights/activity-trend?patient_id={pid}&days=100000",
        headers=h)).status_code == 200
