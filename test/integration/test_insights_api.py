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


# ── widget engine + home-data (the BYPASSRLS-retirement surface) ─────

@pytest.mark.asyncio
async def test_widget_data_scope_and_shapes(
    client: AsyncClient, auth_user_factory,
) -> None:
    h_a, pid_a = await _caregiver_with_patient(client, auth_user_factory)
    h_b, pid_b = await _caregiver_with_patient(client, auth_user_factory)

    # Foreign patient in `source.patient` AND in `compareWith` → 404.
    for source in (
        {"metric": "activity", "patient": pid_b},
        {"patient": pid_a, "compareWith": pid_b, "days": 7},
    ):
        r = await client.post("/v1/insights/widget-data",
                              headers=h_a, json={"source": source})
        assert r.status_code == 404, source

    # Default events series: zero-filled to the full window.
    r = (await client.post("/v1/insights/widget-data", headers=h_a,
                           json={"source": {"days": 7}})).json()
    assert r["kind"] == "series" and r["terms"] == ["events"]
    assert len(r["rows"]) == 7
    assert all(row["events"] == 0 for row in r["rows"])
    assert r["statValue"] == "0"

    # byHour: exactly 24 buckets.
    r = (await client.post("/v1/insights/widget-data", headers=h_a,
                           json={"source": {"byHour": True}})).json()
    assert len(r["rows"]) == 24 and r["rows"][0]["bucket"] == "00:00"

    # byDow: labeled Mon..Sun.
    r = (await client.post("/v1/insights/widget-data", headers=h_a,
                           json={"source": {"byDow": True}})).json()
    assert [x["bucket"] for x in r["rows"]] == [
        "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # daysSince with no matches: the honest unavailable answer.
    r = (await client.post("/v1/insights/widget-data", headers=h_a,
                           json={"source": {"metric": "daysSince",
                                            "terms": ["fall"]}})).json()
    assert r["statValue"] is None and "unavailable" in r

    # activity byHour: 24 buckets, percent unit.
    r = (await client.post("/v1/insights/widget-data", headers=h_a,
                           json={"source": {"metric": "activity",
                                            "byHour": True,
                                            "patient": pid_a}})).json()
    assert r["unit"] == "%" and len(r["rows"]) == 24

    # Term sanitiser: quotes stripped, no SQL error, empty result.
    r = await client.post(
        "/v1/insights/widget-data", headers=h_a,
        json={"source": {"terms": ["fall'; drop table events; --"],
                         "days": 7}})
    assert r.status_code == 200
    assert r.json()["statValue"] == "0"


@pytest.mark.asyncio
async def test_home_data_scope_and_shape(
    client: AsyncClient, auth_user_factory,
) -> None:
    h_a, pid_a = await _caregiver_with_patient(client, auth_user_factory)
    h_b, pid_b = await _caregiver_with_patient(client, auth_user_factory)

    r = await client.get(
        f"/v1/insights/home-data?patient_id={pid_b}", headers=h_a)
    assert r.status_code == 404

    r = (await client.get("/v1/insights/home-data", headers=h_a))
    assert r.status_code == 200
    d = r.json()
    assert d["patient"]["id"] == pid_a          # default pick = own patient
    assert set(d) == {"patient", "cameras", "recent", "last_fall_at",
                      "event_nights", "feed", "wakes"}
    assert d["cameras"] == {"total": 0, "online": 0}
    assert d["recent"] == [] and d["feed"] == [] and d["wakes"] == []
    assert d["last_fall_at"] is None
