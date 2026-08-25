"""A sensor streams the moment it is provisioned, before anyone assigns it.

This is the installer's flow. Someone puts a sensor on a wall, points it
at the account by email, and has to be able to SEE it working before they
leave the house. Until now they could not: `livekit_room` was
`patient_<uuid>`, so a sensor had no room until a caregiver picked a
patient for it, and both roster scopes carried `patient_id IS NOT NULL`,
so an unassigned sensor was absent from the fleet and never published.

The room is the account's — `account_<caregiver_id>` — and it is known at
provision time because the account is the one thing provisioning always
has. Assignment stops being the thing that creates the stream and becomes
what it should always have been: a statement about who the AI is watching
for, which can be made later, changed, or undone, without the video
pausing for a moment.

That also removes the primary patient. It only ever existed to answer
"whose room does a shared sensor publish into", which is not a question
any more.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt, provision_sensor

_PATIENT = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
            "relation": "parent"}


async def _account(client: AsyncClient, auth_user):
    """A caregiver with NO patients yet. That is the point of the flow."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    return headers, email, uid


async def _add_patient(client, headers, name="Margaret O'Brien") -> str:
    r = await client.post("/v1/patients", headers=headers,
                          json={**_PATIENT, "full_name": name})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


async def _provision(client, email, token="gw-sensor-flow", *,
                     headers=None) -> dict:
    """Binding takes a pairing code now, so the caller needs caregiver
    headers. `email` is kept as the typo check the endpoint accepts."""
    body = await provision_sensor(client, headers, token=token,
                                  room="Living Room")
    return body["cameras"][0]


async def _roster(client, token="gw-sensor-flow") -> list[dict]:
    """What the bridge is told to run, through the contract it really uses."""
    r = await client.post(
        "/v1/fleet/poll",
        headers={"Authorization": f"Bearer {token}"},
        json={"contract": "fleet-v1",
              "service": {"kind": "camera-bridge", "instance_id": "bridge-test",
                          "version": "test", "capabilities": []},
              "entities": []})
    assert r.status_code == 200, r.text
    return r.json().get("cameras", [])


@pytest.mark.asyncio
async def test_a_sensor_gets_its_room_at_provision_with_no_patient(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The account is known at provision, so the room is too."""
    headers, email, uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)
    assert camera["livekit_room"] == f"account_{str(uid).lower()}"
    assert camera["patient_id"] is None


@pytest.mark.asyncio
async def test_an_unassigned_sensor_is_on_the_roster_and_publishes(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The installer's whole ask: see it work before leaving the house."""
    headers, email, _uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)

    roster = await _roster(client)
    mine = [c for c in roster if c["camera_id"] == camera["camera_id"]]
    assert mine, "an unassigned sensor must still be told to publish"
    assert mine[0]["livekit_room"] == camera["livekit_room"]
    assert mine[0]["desired_state"] == "connected"


@pytest.mark.asyncio
async def test_assigning_a_patient_does_not_move_the_stream(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Assignment is attribution, not plumbing. The video never pauses."""
    headers, email, _uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)
    room_before = camera["livekit_room"]

    patient = await _add_patient(client, headers)
    r = await client.post(f"/v1/cameras/{camera['camera_id']}/coverage",
                          headers=headers, json={"patient_ids": [patient]})
    assert r.status_code == 200, r.text

    roster = await _roster(client)
    mine = [c for c in roster if c["camera_id"] == camera["camera_id"]][0]
    assert mine["livekit_room"] == room_before


@pytest.mark.asyncio
async def test_two_patients_on_one_sensor_are_equals(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """No primary. Neither patient owns the stream; both are watched by it."""
    headers, email, _uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)
    a = await _add_patient(client, headers, "Margaret O'Brien")
    b = await _add_patient(client, headers, "Arthur O'Brien")

    r = await client.post(f"/v1/cameras/{camera['camera_id']}/coverage",
                          headers=headers, json={"patient_ids": [a, b]})
    assert r.status_code == 200, r.text

    listed = (await client.get("/v1/cameras", headers=headers)).json()
    mine = [c for c in listed if c["id"] == camera["camera_id"]][0]
    assert set(mine["patient_ids"]) == {a, b}
    assert "primary_patient_id" not in mine, (
        "a primary is a fact about LiveKit room naming, and there is no "
        "such fact any more")

    # And the database must not carry the concept either.
    cols = (await session.execute(text(
        "SELECT column_name FROM information_schema.columns "
        " WHERE table_name = 'camera_patients'"))).scalars().all()
    assert "is_primary" not in cols


@pytest.mark.asyncio
async def test_unassigning_leaves_the_sensor_streaming(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Taking a patient off a sensor stops attribution, not the camera.

    The sensor is still on the wall and the caregiver can still look at
    it. Previously this dropped it off the roster entirely.
    """
    headers, email, _uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)
    patient = await _add_patient(client, headers)
    await client.post(f"/v1/cameras/{camera['camera_id']}/coverage",
                      headers=headers, json={"patient_ids": [patient]})

    r = await client.delete(f"/v1/cameras/{camera['camera_id']}/coverage",
                            headers=headers)
    assert r.status_code == 200, r.text

    roster = await _roster(client)
    mine = [c for c in roster if c["camera_id"] == camera["camera_id"]]
    assert mine, "an unassigned sensor is still a sensor"
    assert mine[0]["livekit_room"] == camera["livekit_room"]


@pytest.mark.asyncio
async def test_the_audio_kill_actually_persists(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """PATCH must reach the database, not just return 200.

    `audio_capture_enabled` was added to the table and wired through the
    repo without being mapped on the ORM, so the assignment created a
    plain Python attribute that was never flushed. The endpoint returned
    200, the column stayed true, the roster kept saying true, and the
    microphone stayed live. A dead switch on a privacy control is worse
    than no switch, which is the reason the field moved here at all.
    """
    headers, email, _uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)

    r = await client.patch(f"/v1/cameras/{camera['camera_id']}",
                           headers=headers,
                           json={"audio_capture_enabled": False})
    assert r.status_code == 200, r.text

    stored = (await session.execute(
        text("SELECT audio_capture_enabled FROM public.cameras WHERE id = :i"),
        {"i": camera["camera_id"]},
    )).scalar_one()
    assert stored is False, "PATCH returned 200 and changed nothing"

    # And the bridge must be told, or the sensor keeps publishing audio.
    assert (await _roster(client))[0]["audio_capture_enabled"] is False


@pytest.mark.asyncio
async def test_minting_a_pairing_code_cannot_delete_a_sensor(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The orphan sweep must only reach abandoned wall-phone attempts.

    It clears rows with no `api_key_hash`, which is every gateway row —
    they authenticate with an ingest token. Once the sweep became
    account-scoped it could reach live bridge sensors, so naming a wall
    phone after one deleted it outright.
    """
    headers, email, _uid = await _account(client, auth_user)
    camera = await _provision(client, email, headers=headers)
    patient = await _add_patient(client, headers)

    r = await client.post(
        f"/v1/patients/{patient}/cameras/pairing-code",
        headers=headers, json={"room": "Living Room", "icon_name": "tv"})
    assert r.status_code == 409, (
        "a name already taken by a sensor must be refused, not swept")

    survives = (await session.execute(
        text("SELECT count(*) FROM public.cameras WHERE id = :i"),
        {"i": camera["camera_id"]},
    )).scalar_one()
    assert survives == 1, "the bridge sensor was deleted by a pairing code"


@pytest.mark.asyncio
async def test_the_mini_pc_setup_page_also_provisions_into_the_account_room(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """`gateway-provision` is the live Tapo path, not a dead endpoint.

    nemara-gateway/setup_server.py calls it, so it is how every mini-PC
    camera is created. It still built `patient_<id>` after the migration
    moved every other gateway sensor to the account's room — which put
    two sensors on one account into DIFFERENT rooms, the one combination
    the migration assumes cannot happen.
    """
    headers, _email, uid = await _account(client, auth_user)
    patient = await _add_patient(client, headers)

    r = await client.post(
        f"/v1/patients/{patient}/cameras/gateway-provision",
        headers=headers, json={"room": "Hallway", "icon_name": "tv"})
    assert r.status_code == 201, r.text
    assert r.json()["livekit_room"] == f"account_{str(uid).lower()}"
