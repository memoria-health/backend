"""What a shared sensor saw belongs to EVERY resident it watches.

`analyses.patient_id` is stamped from `cameras.patient_id` at tick time,
and that column is the compatibility shim — the earliest-added covered
patient and nobody else. So on a sensor covering two people, every row
of evidence belonged to whichever resident sorted first, and the second
had no observations in the database at all.

Nothing errored. `home_report` described a covered resident as having no
camera and no activity, every insights widget came back empty, and the
sustained-detector machinery could not resolve an episode boundary for
them because the lookback query returned zero rows. A day of watching
somebody reads as a quiet day, which for an eldercare product is
reassurance the system has not earned.

These tests are written from the SECOND patient's side, because that is
the one who had nothing. Each also asserts the first patient still sees
what they always did — the fix must add a resident, not move one.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.services.active_camera import get_active_camera
from src.services.detection.scheduling.decision import qualifying_tick_count
from test.conftest import mint_gateway_code, mint_test_jwt

_PATIENT = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
            "relation": "parent"}


async def _shared_sensor(client: AsyncClient, auth_user):
    """One account, two patients, ONE camera covering both."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    first = (await client.post(
        "/v1/patients", headers=headers, json=_PATIENT)).json()["id"]
    second = (await client.post(
        "/v1/patients", headers=headers,
        json={**_PATIENT, "full_name": "Arthur O'Brien"})).json()["id"]

    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": "Bearer gw-shared-obs"},
        json={"pairing_code": await mint_gateway_code(client, headers),
              "room": "Living Room"})
    assert r.status_code == 201, r.text
    camera_id = r.json()["cameras"][0]["camera_id"]

    c = await client.post(
        f"/v1/cameras/{camera_id}/coverage", headers=headers,
        json={"patient_ids": [first, second]})
    assert c.status_code == 200, c.text
    return headers, uid, UUID(first), UUID(second), UUID(camera_id)


async def _tick(
    session: AsyncSession, *, caregiver_id, stamped_patient, camera_id,
    captured_at, people_present=True, fields=None,
) -> None:
    """One analyses row, stamped exactly as the pipeline stamps it —
    with the SHIM patient, which is the whole problem being tested."""
    session.add(Analysis(
        id=uuid4(),
        caregiver_id=caregiver_id,
        patient_id=stamped_patient,
        camera_id=camera_id,
        provider="gemini",
        description="test tick",
        detector_fields=fields or {},
        people_present=people_present,
        motion_in_clip=True,
        state_after="open",
        captured_at=captured_at,
        created_at=captured_at,
    ))
    await session.commit()


@pytest.mark.asyncio
async def test_the_second_resident_has_a_last_seen_time(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """THE regression, at its plainest. Arthur is watched by a camera
    that has seen a person, and the app told his caregiver nothing had
    ever been observed."""
    headers, cg, first, second, cam = await _shared_sensor(client, auth_user)
    await _tick(session, caregiver_id=cg, stamped_patient=first,
                camera_id=cam, captured_at=datetime.now(timezone.utc))

    r = await client.get(f"/v1/patients/{second}/last-presence", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["last_person_seen_at"] is not None, (
        "a covered resident with a camera that saw someone must not "
        "report that nobody has ever been seen"
    )


@pytest.mark.asyncio
async def test_the_first_resident_still_has_one(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The fix adds a resident; it must not move one."""
    headers, cg, first, second, cam = await _shared_sensor(client, auth_user)
    await _tick(session, caregiver_id=cg, stamped_patient=first,
                camera_id=cam, captured_at=datetime.now(timezone.utc))

    r = await client.get(f"/v1/patients/{first}/last-presence", headers=headers)
    assert r.json()["last_person_seen_at"] is not None


@pytest.mark.asyncio
async def test_the_second_resident_resolves_an_active_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """`active_camera_for_patient` is how the talk feature and the
    identity gate find the room somebody is in. Returning None for a
    covered resident sends both to the wrong place."""
    _, cg, first, second, cam = await _shared_sensor(client, auth_user)
    await _tick(session, caregiver_id=cg, stamped_patient=first,
                camera_id=cam, captured_at=datetime.now(timezone.utc))

    assert await get_active_camera(session, patient_id=second) == cam


@pytest.mark.asyncio
async def test_the_second_resident_accumulates_qualifying_ticks(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The one that decides whether a detector can ever fire. k-of-N
    entry counts PROMOTABLE ticks over a window; at zero, a sustained
    detector on a shared sensor could never enter a state for the second
    resident no matter how long the behaviour continued."""
    _, cg, first, second, cam = await _shared_sensor(client, auth_user)
    now = datetime.now(timezone.utc)
    for i in range(3):
        await _tick(
            session, caregiver_id=cg, stamped_patient=first, camera_id=cam,
            captured_at=now - timedelta(seconds=10 * i),
            fields={"sleep_detected": "true", "sleep_confidence": "0.9"},
        )

    n = await qualifying_tick_count(
        session, patient_id=second, detected_field="sleep_detected",
        conf_field="sleep_confidence", threshold=0.5,
        since=now - timedelta(minutes=5), before=now + timedelta(seconds=1),
    )
    assert n == 3, "evidence pools across every camera watching the patient"


@pytest.mark.asyncio
async def test_a_patient_the_sensor_does_not_cover_sees_nothing(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The boundary that makes the rest meaningful. Resolving through
    coverage must not become 'everyone on the account sees everything'
    — a household with a bathroom camera assigned to one resident is
    exactly the case where that would be a privacy failure, not a bug."""
    headers, cg, first, second, cam = await _shared_sensor(client, auth_user)
    third = (await client.post(
        "/v1/patients", headers=headers,
        json={**_PATIENT, "full_name": "Uncovered Person"})).json()["id"]
    await _tick(session, caregiver_id=cg, stamped_patient=first,
                camera_id=cam, captured_at=datetime.now(timezone.utc))

    r = await client.get(f"/v1/patients/{third}/last-presence", headers=headers)
    assert r.json()["last_person_seen_at"] is None
    assert await get_active_camera(session, patient_id=UUID(third)) is None


@pytest.mark.asyncio
async def test_removing_coverage_removes_the_observations(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Coverage is live, not a stamp taken at tick time. Un-assigning a
    resident has to stop their reports reading that sensor, or 'remove
    them from this camera' is a lie in the one direction that matters."""
    headers, cg, first, second, cam = await _shared_sensor(client, auth_user)
    await _tick(session, caregiver_id=cg, stamped_patient=first,
                camera_id=cam, captured_at=datetime.now(timezone.utc))

    r = await client.post(f"/v1/cameras/{cam}/coverage", headers=headers,
                          json={"patient_ids": [str(first)]})
    assert r.status_code == 200, r.text

    got = await client.get(f"/v1/patients/{second}/last-presence", headers=headers)
    assert got.json()["last_person_seen_at"] is None
