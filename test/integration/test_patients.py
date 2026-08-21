"""Integration tests for /v1/patients/*.

Covers the full CRUD surface plus the writable / non-writable field
split, the cross-account RLS guarantee, and the cascade-on-delete
behavior. Every test that touches a `CaregiverAuth`-protected route
provisions a real `auth.users` + `caregivers` row first.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.db.camera import Camera
from src.db.session import get_sessionmaker
from test.conftest import mint_test_jwt


async def _insert_camera(
    *,
    patient_id: UUID,
    display_id: int,
    room: str,
    source_type: str = "iphone",
    desired_state: str = "connected",
) -> UUID:
    """Insert a minimum-viable camera row. Caller owns FK validity
    (patient_id must already exist). Returns the new camera's id."""
    cam_id = uuid4()
    sm = get_sessionmaker()
    async with sm() as s:
        s.add(
            Camera(
                id=cam_id,
                patient_id=patient_id,
                display_id=display_id,
                room=room,
                icon_name="bed.double",
                online=False,
                state="closed",
                no_people_count=0,
                close_threshold=2,
                source_type=source_type,
                desired_state=desired_state,
                ingest_token_hash=("a" * 64) if source_type == "pi" else None,
                created_at=datetime.now(timezone.utc),
            ),
        )
        await s.commit()
    return cam_id


async def _camera_desired_state(camera_id: UUID) -> str:
    sm = get_sessionmaker()
    async with sm() as s:
        result = await s.execute(
            select(Camera.desired_state).where(Camera.id == camera_id),
        )
        return result.scalar_one()


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


# ─── cameras_paused cascade ───────────────────────────────────────


@pytest.mark.asyncio
async def test_patch_cameras_paused_false_to_true_cascades_to_disconnected(
    client: AsyncClient, auth_user,
) -> None:
    """Flipping `cameras_paused` false to true MUST set every camera
    for the patient to `desired_state='disconnected'` in the same
    transaction. This is the hard-kill behavior the caregiver expects
    when toggling Monitoring off: wall phones leave LiveKit entirely,
    no LiveKit minutes burned while paused."""
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    iphone_id = await _insert_camera(
        patient_id=UUID(pid), display_id=1, room="Living Room",
        source_type="iphone", desired_state="connected",
    )
    pi_id = await _insert_camera(
        patient_id=UUID(pid), display_id=2, room="Bedroom",
        source_type="pi", desired_state="connected",
    )

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers,
        json={"cameras_paused": True},
    )

    assert r.status_code == 200
    assert r.json()["cameras_paused"] is True

    assert await _camera_desired_state(iphone_id) == "disconnected"
    assert await _camera_desired_state(pi_id) == "disconnected"


@pytest.mark.asyncio
async def test_patch_cameras_paused_true_to_false_cascades_to_connected(
    client: AsyncClient, auth_user,
) -> None:
    """Unpausing MUST set every camera back to `desired_state='connected'`.
    Naive restore: caregiver-explicit per-camera disconnects are
    overwritten on unpause (by design, no snapshot column)."""
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    iphone_id = await _insert_camera(
        patient_id=UUID(pid), display_id=1, room="Living Room",
        source_type="iphone", desired_state="disconnected",
    )
    pi_id = await _insert_camera(
        patient_id=UUID(pid), display_id=2, room="Bedroom",
        source_type="pi", desired_state="disconnected",
    )

    await client.patch(
        f"/v1/patients/{pid}", headers=headers, json={"cameras_paused": True},
    )

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers,
        json={"cameras_paused": False},
    )

    assert r.status_code == 200
    assert r.json()["cameras_paused"] is False

    assert await _camera_desired_state(iphone_id) == "connected"
    assert await _camera_desired_state(pi_id) == "connected"


@pytest.mark.asyncio
async def test_patch_cameras_paused_same_value_does_not_cascade(
    client: AsyncClient, auth_user,
) -> None:
    """If the body sets `cameras_paused` to the value it already has,
    no cascade fires and cameras keep their current `desired_state`.
    Prevents churn from the iOS app PATCHing the full mutable surface
    on every minor edit."""
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    cam_id = await _insert_camera(
        patient_id=UUID(pid), display_id=1, room="Living Room",
        source_type="iphone", desired_state="disconnected",
    )

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers,
        json={"cameras_paused": False},
    )

    assert r.status_code == 200
    assert await _camera_desired_state(cam_id) == "disconnected"


@pytest.mark.asyncio
async def test_patch_without_cameras_paused_does_not_touch_cameras(
    client: AsyncClient, auth_user,
) -> None:
    """Editing any OTHER field on the patient must not perturb
    `desired_state` on the cameras. The cascade is gated on
    `cameras_paused` being in the body."""
    headers = await _provision(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers, json=_FULL_BODY)).json()["id"]

    cam_id = await _insert_camera(
        patient_id=UUID(pid), display_id=1, room="Living Room",
        source_type="iphone", desired_state="disconnected",
    )

    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers,
        json={"preferred_name": "Margie"},
    )

    assert r.status_code == 200
    assert await _camera_desired_state(cam_id) == "disconnected"


@pytest.mark.asyncio
async def test_patch_cameras_paused_does_not_cascade_to_other_caregivers_cameras(
    client: AsyncClient, auth_user_factory,
) -> None:
    """Caregiver A pausing their patient's cameras must not touch any
    camera belonging to caregiver B. `camera_repo.list_for_patient`
    enforces the caregiver scope; this test pins that contract."""
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}

    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})

    a_pid = (await client.post("/v1/patients", headers=h_a, json=_FULL_BODY)).json()["id"]
    b_pid = (await client.post("/v1/patients", headers=h_b, json=_FULL_BODY)).json()["id"]

    a_cam = await _insert_camera(
        patient_id=UUID(a_pid), display_id=1, room="A-Bedroom",
    )
    b_cam = await _insert_camera(
        patient_id=UUID(b_pid), display_id=1, room="B-Bedroom",
    )

    r = await client.patch(
        f"/v1/patients/{a_pid}", headers=h_a, json={"cameras_paused": True},
    )
    assert r.status_code == 200

    assert await _camera_desired_state(a_cam) == "disconnected"
    assert await _camera_desired_state(b_cam) == "connected"


# ─── Age-based preset detectors ──────────────────────────────────


@pytest.mark.asyncio
async def test_under_18_seeds_child_detectors(
    client: AsyncClient, auth_user
) -> None:
    """A patient under 18 seeds ONLY the both-cohort safety net (fall).
    The child-only preset set (climbing/distress/child_*) was deleted
    2026-08-16 — adult-only product, zero rows in any database — so a
    minor must get neither those nor the older-adult
    agitation/wandering/medication set."""
    headers = await _provision(client, auth_user)
    pid = (
        await client.post(
            "/v1/patients",
            headers=headers,
            json={
                "full_name": "Sam Lee",
                "preferred_name": "Sam",
                "relation": "other",
                "date_of_birth": "2015-06-01",  # a minor
            },
        )
    ).json()["id"]
    dets = (await client.get(f"/v1/patients/{pid}/detectors", headers=headers)).json()
    ids = {d["identifier"] for d in dets}
    assert ids == {"fall"}


@pytest.mark.asyncio
async def test_adult_and_unknown_age_seed_adult_detectors(
    client: AsyncClient, auth_user
) -> None:
    """18+ and unknown-birthday patients both get the older-adult set."""
    headers = await _provision(client, auth_user)

    adult_pid = (
        await client.post(
            "/v1/patients",
            headers=headers,
            json={
                "full_name": "Margaret O'Brien",
                "preferred_name": "Maggie",
                "relation": "parent",
                "date_of_birth": "1951-03-18",
            },
        )
    ).json()["id"]
    adult_ids = {
        d["identifier"]
        for d in (
            await client.get(f"/v1/patients/{adult_pid}/detectors", headers=headers)
        ).json()
    }
    assert adult_ids == {"fall", "agitation", "wandering", "medication"}

    # No date_of_birth → adult default (unchanged prior behavior).
    nodob_pid = (
        await client.post(
            "/v1/patients",
            headers=headers,
            json={
                "full_name": "No Birthday",
                "preferred_name": "Nb",
                "relation": "other",
            },
        )
    ).json()["id"]
    nodob_ids = {
        d["identifier"]
        for d in (
            await client.get(f"/v1/patients/{nodob_pid}/detectors", headers=headers)
        ).json()
    }
    assert "agitation" in nodob_ids
    assert "climbing" not in nodob_ids
