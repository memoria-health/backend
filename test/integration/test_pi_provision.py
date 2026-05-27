"""Integration tests for the Pi provisioning endpoint.

POST /v1/patients/{patient_id}/cameras/pi-provision

Distinct from the iPhone pairing flow: a Pi camera is created in a
single caregiver-authenticated POST. There is no device-side bearer
token (the backend's pi_ingest worker authenticates to LiveKit on
the Pi's behalf using server credentials).

Verifies ownership checks, the created row's full field shape, the
preset-detector auto-assignment that matches the iPhone flow, and
the 409 collision on a duplicate pi_stream_url.
"""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_pi_provision_creates_row_with_full_shape(
    client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    headers, pid = await _setup_patient(client, auth_user)

    body = {
        "room": "Living Room",
        "icon_name": "tv",
        "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg",
    }
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json=body,
    )
    assert r.status_code == 201, r.text
    resp = r.json()

    # Response body: caller gets back what it needs to confirm the
    # camera is wired before the worker brings it online.
    assert resp["patient_id"] == pid
    assert resp["pi_stream_url"] == "http://10.0.0.44:8000/stream.mjpg"
    assert resp["livekit_room"] == f"patient_{pid}"
    assert resp["livekit_identity"].startswith("cam_")
    assert len(resp["livekit_identity"]) == len("cam_") + 8  # short hex
    camera_id = resp["camera_id"]

    # DB shape: every field that matters for the iPhone path is also
    # well-defined on the Pi path.
    row = (
        await session.execute(
            text(
                "SELECT patient_id, display_id, room, icon_name, online, "
                "source_type, pi_stream_url, livekit_identity, livekit_room, "
                "state, no_people_count, close_threshold, disabled, "
                "desired_state, pairing_code, api_key_hash "
                "FROM cameras WHERE id = :id"
            ),
            {"id": camera_id},
        )
    ).one()
    assert str(row.patient_id) == pid
    assert row.display_id == 1  # first camera for this patient
    assert row.room == "Living Room"
    assert row.icon_name == "tv"
    assert row.online is False  # worker flips this when it confirms the stream
    assert row.source_type == "pi"
    assert row.pi_stream_url == "http://10.0.0.44:8000/stream.mjpg"
    assert row.livekit_identity == resp["livekit_identity"]
    assert row.livekit_room == resp["livekit_room"]
    assert row.state == "open"
    assert row.no_people_count == 0
    assert row.close_threshold == 6
    assert row.disabled is False
    assert row.desired_state == "connected"
    # Pi rows skip the pairing-code dance: there is no device-side
    # bearer token. The pi_ingest worker authenticates to LiveKit
    # using server credentials.
    assert row.pairing_code is None
    assert row.api_key_hash is None


@pytest.mark.asyncio
async def test_pi_provision_auto_assigns_preset_detectors(
    client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    """Pi cameras must get the same preset-detector auto-assignment
    as iPhone cameras so they are detector-equivalent from day one.
    """
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={
            "room": "Kitchen",
            "pi_stream_url": "http://10.0.0.50:8000/stream.mjpg",
        },
    )
    assert r.status_code == 201
    camera_id = r.json()["camera_id"]

    detector_count = (
        await session.execute(
            text(
                "SELECT count(*) FROM cameras_detectors WHERE camera_id = :id"
            ),
            {"id": camera_id},
        )
    ).scalar_one()
    # At least one preset detector should be assigned (the seed has
    # several, e.g. Falls, Wandering). The exact set depends on the
    # detectors seeded in the test DB; assert >= 1 so the test is
    # robust to seed changes.
    assert detector_count >= 1, (
        f"Pi camera created without any preset detectors; got {detector_count}"
    )


@pytest.mark.asyncio
async def test_pi_provision_rejects_duplicate_url(
    client: AsyncClient, auth_user
) -> None:
    """A pi_stream_url is uniquely tied to one physical device; a
    second provision against the same URL is a configuration error
    that should fail loudly (409), not silently create a duplicate.
    """
    headers, pid = await _setup_patient(client, auth_user)
    body = {
        "room": "Living Room",
        "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg",
    }
    r1 = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision", headers=headers, json=body,
    )
    assert r1.status_code == 201
    r2 = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision", headers=headers, json=body,
    )
    assert r2.status_code == 409
    assert r2.json()["detail"] == "pi_stream_url_already_registered"


@pytest.mark.asyncio
async def test_pi_provision_rejects_other_caregivers_patient(
    client: AsyncClient, auth_user_factory
) -> None:
    """Caregiver B cannot provision a Pi camera for caregiver A's
    patient. Returns 404 (same wire shape as 'patient not found',
    intentionally not 403, to avoid leaking patient existence).
    """
    uid_a, email_a = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(uid_a, email_a)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    pa = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()

    uid_b, email_b = await auth_user_factory()
    h_b = {"Authorization": f"Bearer {mint_test_jwt(uid_b, email_b)}"}
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})

    r = await client.post(
        f"/v1/patients/{pa['id']}/cameras/pi-provision",
        headers=h_b,
        json={
            "room": "Hijacked",
            "pi_stream_url": "http://10.0.0.99:8000/stream.mjpg",
        },
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "patient_not_found"


@pytest.mark.asyncio
async def test_pi_provision_rejects_invalid_request_body(
    client: AsyncClient, auth_user
) -> None:
    headers, pid = await _setup_patient(client, auth_user)

    # Missing pi_stream_url
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Bedroom"},
    )
    assert r.status_code == 422

    # Empty room
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "", "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg"},
    )
    assert r.status_code == 422

    # pi_stream_url too short (< 8 chars)
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Bedroom", "pi_stream_url": "http://"},
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_pi_provision_unauthenticated_returns_401(client: AsyncClient) -> None:
    r = await client.post(
        "/v1/patients/00000000-0000-0000-0000-000000000000/cameras/pi-provision",
        json={"room": "Living Room", "pi_stream_url": "http://x:8/s.mjpg"},
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_pi_provision_assigns_dense_display_id_per_patient(
    client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    """Two Pi cameras for the same patient must get display_id 1 then 2
    (matching the iPhone path's per-patient sequence).
    """
    headers, pid = await _setup_patient(client, auth_user)

    r1 = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={
            "room": "Living Room",
            "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg",
        },
    )
    r2 = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={
            "room": "Kitchen",
            "pi_stream_url": "http://10.0.0.45:8000/stream.mjpg",
        },
    )
    assert r1.status_code == 201
    assert r2.status_code == 201

    rows = (
        await session.execute(
            text("SELECT display_id FROM cameras WHERE patient_id = :pid ORDER BY display_id"),
            {"pid": pid},
        )
    ).all()
    assert [row.display_id for row in rows] == [1, 2]
