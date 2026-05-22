"""Integration tests for the detector API.

Covers custom-detector creation, sensitivity tuning (the iOS Patient
Settings UI's main affordance), enum validation, the source=preset
immutability, RLS, and the cascade-on-patient-delete behavior.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}

_CUSTOM_DETECTOR_BODY = {
    "name": "Late kitchen visits",
    "icon_name": "moon.stars",
    "description": "Kitchen activity between 11 PM and 6 AM.",
    "category": "safety",
    "shape": "point_in_time",
    "where_scope": "KITCHEN",
    "when_scope": "11 PM – 6 AM",
    "custom_prompt": "Trigger when motion is detected in the kitchen between 11pm and 6am.",
}


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    """Provision caregiver + patient. Returns (headers, patient_id)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


# ─── List + auto-seed ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_detectors_returns_four_presets_after_patient_create(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.get(f"/v1/patients/{pid}/detectors", headers=headers)
    assert r.status_code == 200
    dets = r.json()
    assert [d["name"] for d in dets] == [
        "Falls", "Agitation", "Wandering", "Medication adherence",
    ]
    # Field-by-field check on the first preset.
    falls = dets[0]
    assert falls["icon_name"] == "figure.fall"
    assert falls["description"] == "Sudden fall events across all rooms."
    assert falls["category"] == "safety"
    assert falls["source"] == "preset"
    assert falls["shape"] == "point_in_time"
    assert falls["where_scope"] == "ALL ROOMS"
    assert falls["when_scope"] == "ALWAYS"
    assert falls["sensitivity"] == "standard"
    assert falls["is_on"] is True
    assert falls["custom_prompt"] is None


@pytest.mark.asyncio
async def test_list_unauthenticated_returns_401(client: AsyncClient) -> None:
    r = await client.get(
        "/v1/patients/00000000-0000-0000-0000-000000000000/detectors"
    )
    assert r.status_code == 401


# ─── Create custom ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_custom_detector_full_field_assertion(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        f"/v1/patients/{pid}/detectors",
        headers=headers,
        json=_CUSTOM_DETECTOR_BODY,
    )
    assert r.status_code == 201
    d = r.json()
    assert d["name"] == _CUSTOM_DETECTOR_BODY["name"]
    assert d["icon_name"] == _CUSTOM_DETECTOR_BODY["icon_name"]
    assert d["description"] == _CUSTOM_DETECTOR_BODY["description"]
    assert d["category"] == "safety"
    assert d["source"] == "custom"                          # server-forced
    assert d["shape"] == "point_in_time"
    assert d["where_scope"] == "KITCHEN"
    assert d["when_scope"] == "11 PM – 6 AM"
    assert d["sensitivity"] == "standard"                   # default
    assert d["is_on"] is True
    assert d["custom_prompt"] == _CUSTOM_DETECTOR_BODY["custom_prompt"]
    assert d["patient_id"] == pid
    assert "id" in d


@pytest.mark.asyncio
async def test_create_custom_detector_with_explicit_sensitivity(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    body = {**_CUSTOM_DETECTOR_BODY, "sensitivity": "high"}
    r = await client.post(
        f"/v1/patients/{pid}/detectors", headers=headers, json=body
    )
    assert r.status_code == 201
    assert r.json()["sensitivity"] == "high"


@pytest.mark.asyncio
async def test_create_custom_detector_rejects_unknown_field(
    client: AsyncClient, auth_user
) -> None:
    """`extra="forbid"` rejects `source` from the client — server forces it."""
    headers, pid = await _setup_patient(client, auth_user)
    body = {**_CUSTOM_DETECTOR_BODY, "source": "preset"}
    r = await client.post(
        f"/v1/patients/{pid}/detectors", headers=headers, json=body
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_custom_detector_rejects_invalid_enum(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    body = {**_CUSTOM_DETECTOR_BODY, "category": "not-a-real-category"}
    r = await client.post(
        f"/v1/patients/{pid}/detectors", headers=headers, json=body
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_create_custom_detector_on_other_caregivers_patient_404(
    client: AsyncClient, auth_user_factory
) -> None:
    """Caregiver B can't create a detector on caregiver A's patient."""
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]

    r = await client.post(
        f"/v1/patients/{pid}/detectors", headers=h_b, json=_CUSTOM_DETECTOR_BODY
    )
    assert r.status_code == 404


# ─── PATCH (sensitivity tuning, on/off toggle, …) ─────────────────


@pytest.mark.asyncio
async def test_patch_sensitivity_tune(client: AsyncClient, auth_user) -> None:
    """The iOS UI's "Tune" affordance hits this — toggles Low/Standard/High."""
    headers, pid = await _setup_patient(client, auth_user)
    dets = (await client.get(f"/v1/patients/{pid}/detectors", headers=headers)).json()
    falls_id = dets[0]["id"]

    for level in ("low", "standard", "high"):
        r = await client.patch(
            f"/v1/detectors/{falls_id}",
            headers=headers,
            json={"sensitivity": level},
        )
        assert r.status_code == 200
        assert r.json()["sensitivity"] == level


@pytest.mark.asyncio
async def test_patch_toggle_is_on(client: AsyncClient, auth_user) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    dets = (await client.get(f"/v1/patients/{pid}/detectors", headers=headers)).json()
    falls_id = dets[0]["id"]

    r = await client.patch(
        f"/v1/detectors/{falls_id}", headers=headers, json={"is_on": False},
    )
    assert r.status_code == 200
    assert r.json()["is_on"] is False

    r = await client.patch(
        f"/v1/detectors/{falls_id}", headers=headers, json={"is_on": True},
    )
    assert r.status_code == 200
    assert r.json()["is_on"] is True


@pytest.mark.asyncio
async def test_patch_partial_leaves_unspecified_fields_unchanged(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    dets = (await client.get(f"/v1/patients/{pid}/detectors", headers=headers)).json()
    falls = dets[0]

    r = await client.patch(
        f"/v1/detectors/{falls['id']}",
        headers=headers,
        json={"sensitivity": "high"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["sensitivity"] == "high"
    # Everything else is the seeded preset value.
    assert d["name"] == "Falls"
    assert d["category"] == "safety"
    assert d["source"] == "preset"
    assert d["shape"] == "point_in_time"
    assert d["is_on"] is True


@pytest.mark.asyncio
async def test_patch_404_when_detector_missing(
    client: AsyncClient, auth_user
) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    r = await client.patch(
        "/v1/detectors/00000000-0000-0000-0000-000000000000",
        headers=headers,
        json={"sensitivity": "low"},
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_patch_other_caregivers_detector_404(
    client: AsyncClient, auth_user_factory
) -> None:
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]
    det_id = (await client.get(f"/v1/patients/{pid}/detectors", headers=h_a)).json()[0]["id"]

    r = await client.patch(
        f"/v1/detectors/{det_id}", headers=h_b, json={"sensitivity": "high"},
    )
    assert r.status_code == 404


# ─── DELETE ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_custom_detector(client: AsyncClient, auth_user) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    created = await client.post(
        f"/v1/patients/{pid}/detectors", headers=headers, json=_CUSTOM_DETECTOR_BODY,
    )
    did = created.json()["id"]

    r = await client.delete(f"/v1/detectors/{did}", headers=headers)
    assert r.status_code == 204

    # Re-list — preset 4 remain, custom is gone.
    dets = (await client.get(f"/v1/patients/{pid}/detectors", headers=headers)).json()
    assert len(dets) == 4
    assert all(d["source"] == "preset" for d in dets)


@pytest.mark.asyncio
async def test_delete_404_when_detector_missing(
    client: AsyncClient, auth_user
) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    r = await client.delete(
        "/v1/detectors/00000000-0000-0000-0000-000000000000", headers=headers,
    )
    assert r.status_code == 404


# ─── Cascade ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_patient_cascades_to_detectors(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    # 4 preset detectors exist; pre-confirm.
    pre = await client.get(f"/v1/patients/{pid}/detectors", headers=headers)
    assert len(pre.json()) == 4

    r = await client.delete(f"/v1/patients/{pid}", headers=headers)
    assert r.status_code == 204

    # Listing the (now-deleted) patient's detectors returns empty —
    # joining through patients filters them out cleanly.
    post = await client.get(f"/v1/patients/{pid}/detectors", headers=headers)
    assert post.status_code == 200
    assert post.json() == []
