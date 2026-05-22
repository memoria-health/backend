"""Integration tests for /v1/patients/*.

Covers the full CRUD surface plus the writable / non-writable field
split, the cross-account RLS guarantee, and the cascade-on-delete
behavior. Every test that touches a `CaregiverAuth`-protected route
provisions a real `auth.users` + `caregivers` row first.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from test.conftest import mint_test_jwt


async def _provision(client: AsyncClient, auth_user) -> dict[str, str]:
    """Helper: turn an auth_user fixture into a working bearer-header
    + a provisioned caregiver row. Most patient tests need this prelude.
    """
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    r = await client.post(
        "/v1/me", headers=headers, json={"display_name": "Test Caregiver"},
    )
    assert r.status_code == 200
    return headers


_FULL_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "date_of_birth": "1951-03-18",
    "gender": "Female",
    "relation": "parent",
    "home_address": "84 Birch Lane, Cambridge MA",
}


# ─── auth + ownership ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_patients_without_auth_returns_401(client: AsyncClient) -> None:
    r = await client.get("/v1/patients")
    assert r.status_code == 401


# Note: not testing "POST /v1/patients without prior POST /v1/me" because
# the iOS app always calls /v1/me on signup before any patient UI. The
# server-side FK violation that would surface in that case is an unhandled
# IntegrityError today; we'll add a friendlier 409 mapping if/when we
# decide that's a real surface to defend.


# ─── CRUD happy path ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_then_get_then_delete_roundtrip(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)

    # Create
    r = await client.post("/v1/patients", headers=headers, json=_FULL_BODY)
    assert r.status_code == 201
    p = r.json()
    pid = p["id"]
    assert p["full_name"] == "Margaret O'Brien"
    assert p["preferred_name"] == "Maggie"
    assert p["date_of_birth"] == "1951-03-18"
    assert p["gender"] == "Female"
    assert p["relation"] == "parent"
    assert p["home_address"] == "84 Birch Lane, Cambridge MA"
    # Server-managed fields default safely.
    assert p["state"] == "calm"
    assert p["status_line"] == "All quiet"
    assert p["last_event_at"] is None
    assert p["battery_percent"] is None
    assert p["cameras_paused"] is False

    # List
    r = await client.get("/v1/patients", headers=headers)
    assert r.status_code == 200
    listed = r.json()
    assert len(listed) == 1
    assert listed[0]["id"] == pid

    # Get one
    r = await client.get(f"/v1/patients/{pid}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == pid

    # Delete
    r = await client.delete(f"/v1/patients/{pid}", headers=headers)
    assert r.status_code == 204

    # Get one → 404
    r = await client.get(f"/v1/patients/{pid}", headers=headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_create_minimal_body(client: AsyncClient, auth_user) -> None:
    """Only the three required fields — date_of_birth, gender,
    home_address are optional."""
    headers = await _provision(client, auth_user)
    r = await client.post(
        "/v1/patients",
        headers=headers,
        json={
            "full_name": "Henry Gold",
            "preferred_name": "Henry",
            "relation": "other",
        },
    )
    assert r.status_code == 201
    p = r.json()
    assert p["full_name"] == "Henry Gold"
    assert p["date_of_birth"] is None
    assert p["gender"] is None
    assert p["home_address"] is None


# ─── PATCH semantics ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_writable_fields(client: AsyncClient, auth_user) -> None:
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    r = await client.patch(
        f"/v1/patients/{pid}",
        headers=headers,
        json={
            "preferred_name": "Margie",
            "home_address": "9 New St, Cambridge MA",
            "cameras_paused": True,
            "relation": "spouse",
        },
    )
    assert r.status_code == 200
    p = r.json()
    assert p["preferred_name"] == "Margie"
    assert p["home_address"] == "9 New St, Cambridge MA"
    assert p["cameras_paused"] is True
    assert p["relation"] == "spouse"
    # Unspecified fields preserved.
    assert p["full_name"] == "Margaret O'Brien"
    assert p["date_of_birth"] == "1951-03-18"


@pytest.mark.asyncio
async def test_patch_rejects_server_managed_state_field(
    client: AsyncClient, auth_user
) -> None:
    """`state` is server/wall-phone-managed; the Pydantic model doesn't
    expose it and `extra="forbid"` rejects it as 422."""
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers, json={"state": "critical"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_rejects_status_line_field(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers,
        json={"status_line": "Manual override"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_rejects_battery_percent_field(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers, json={"battery_percent": 50},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_patch_404_when_patient_missing(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    r = await client.patch(
        "/v1/patients/00000000-0000-0000-0000-000000000000",
        headers=headers,
        json={"preferred_name": "Nope"},
    )
    assert r.status_code == 404


# ─── Cross-caregiver isolation (RLS-equivalent at the app layer) ──


@pytest.mark.asyncio
async def test_other_caregiver_cannot_see_my_patient(
    client: AsyncClient, auth_user_factory
) -> None:
    """Caregiver A creates a patient. Caregiver B's GET returns 404."""
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()

    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}

    # Provision both caregivers.
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})

    # A creates a patient.
    pid = (await client.post("/v1/patients", headers=h_a, json=_FULL_BODY)).json()["id"]

    # B's list omits A's patient.
    r = await client.get("/v1/patients", headers=h_b)
    assert r.status_code == 200
    assert r.json() == []

    # B's GET returns 404 (not 403 — no existence leak).
    r = await client.get(f"/v1/patients/{pid}", headers=h_b)
    assert r.status_code == 404

    # B's PATCH returns 404.
    r = await client.patch(
        f"/v1/patients/{pid}", headers=h_b, json={"preferred_name": "X"},
    )
    assert r.status_code == 404

    # B's DELETE returns 404.
    r = await client.delete(f"/v1/patients/{pid}", headers=h_b)
    assert r.status_code == 404


# ─── Auto-seed of preset detectors (Step 7) ───────────────────────


@pytest.mark.asyncio
async def test_create_patient_auto_seeds_four_preset_detectors(
    client: AsyncClient, auth_user
) -> None:
    """POST /v1/patients should seed Falls / Agitation / Wandering /
    Medication adherence — same shape MockData.swift uses on iOS.
    """
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    r = await client.get(f"/v1/patients/{pid}/detectors", headers=headers)
    assert r.status_code == 200
    detectors = r.json()
    assert len(detectors) == 4

    names = [d["name"] for d in detectors]
    assert names == ["Falls", "Agitation", "Wandering", "Medication adherence"]
    for d in detectors:
        assert d["source"] == "preset"
        assert d["sensitivity"] == "standard"
        assert d["is_on"] is True
        assert d["custom_prompt"] is None
