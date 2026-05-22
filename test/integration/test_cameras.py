"""Integration tests for the camera API.

Cameras are CREATED by wall phones in a later plan. v0 only lists +
deletes, so the test surface is small: list shape, delete happy path,
404s, cross-caregiver isolation, cascade-on-patient-delete.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _insert_camera(
    session: AsyncSession,
    *,
    patient_id: str,
    display_id: int,
    room: str,
    icon_name: str = "tv",
    online: bool = True,
) -> UUID:
    """Insert one camera row via raw SQL (no caregiver-facing POST exists)."""
    cid = uuid4()
    await session.execute(
        text("""
            INSERT INTO cameras (id, patient_id, display_id, room, icon_name, online, created_at)
            VALUES (:id, :pid, :display, :room, :icon, :online, now())
        """),
        {
            "id": str(cid),
            "pid": patient_id,
            "display": display_id,
            "room": room,
            "icon": icon_name,
            "online": online,
        },
    )
    await session.commit()
    return cid


@pytest.mark.asyncio
async def test_list_cameras_empty(client: AsyncClient, auth_user) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.get(f"/v1/patients/{pid}/cameras", headers=headers)
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_list_cameras_returns_sorted_by_display_id(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    # Insert out of order — endpoint should still return ascending display_id.
    await _insert_camera(session, patient_id=pid, display_id=3, room="Bedroom",  icon_name="bed.double")
    await _insert_camera(session, patient_id=pid, display_id=1, room="Living room", icon_name="tv")
    await _insert_camera(session, patient_id=pid, display_id=2, room="Kitchen", icon_name="fork.knife")

    r = await client.get(f"/v1/patients/{pid}/cameras", headers=headers)
    assert r.status_code == 200
    rooms = [c["room"] for c in r.json()]
    assert rooms == ["Living room", "Kitchen", "Bedroom"]


@pytest.mark.asyncio
async def test_list_cameras_full_field_assertion(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_camera(
        session, patient_id=pid, display_id=1, room="Living room",
        icon_name="tv", online=True,
    )

    cams = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert len(cams) == 1
    c = cams[0]
    assert c["patient_id"] == pid
    assert c["display_id"] == 1
    assert c["room"] == "Living room"
    assert c["icon_name"] == "tv"
    assert c["online"] is True
    assert c["last_seen_at"] is None
    assert "id" in c
    assert "created_at" in c


@pytest.mark.asyncio
async def test_list_cameras_unauthenticated_returns_401(
    client: AsyncClient
) -> None:
    r = await client.get(
        "/v1/patients/00000000-0000-0000-0000-000000000000/cameras"
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_delete_camera(client: AsyncClient, auth_user, session) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Living room")

    r = await client.delete(f"/v1/cameras/{cid}", headers=headers)
    assert r.status_code == 204

    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert listed == []


@pytest.mark.asyncio
async def test_delete_camera_404_when_missing(
    client: AsyncClient, auth_user
) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    r = await client.delete(
        "/v1/cameras/00000000-0000-0000-0000-000000000000", headers=headers,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_other_caregiver_cannot_see_or_delete_camera(
    client: AsyncClient, auth_user_factory, session
) -> None:
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]
    cid = await _insert_camera(session, patient_id=pid, display_id=1, room="Living room")

    # B's list omits the camera.
    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=h_b)).json()
    assert listed == []
    # B's DELETE 404 (no existence leak).
    r = await client.delete(f"/v1/cameras/{cid}", headers=h_b)
    assert r.status_code == 404

    # And A's camera is still there.
    listed_a = (await client.get(f"/v1/patients/{pid}/cameras", headers=h_a)).json()
    assert len(listed_a) == 1


@pytest.mark.asyncio
async def test_delete_patient_cascades_to_cameras(
    client: AsyncClient, auth_user, session
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _insert_camera(session, patient_id=pid, display_id=1, room="Living room")
    await _insert_camera(session, patient_id=pid, display_id=2, room="Kitchen")

    assert (await client.delete(f"/v1/patients/{pid}", headers=headers)).status_code == 204

    listed = (await client.get(f"/v1/patients/{pid}/cameras", headers=headers)).json()
    assert listed == []
