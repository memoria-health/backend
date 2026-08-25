"""A wall phone is paired to the ACCOUNT, and given residents after.

Pairing was the last thing still scoped to a patient. The URL carried
`/patients/{id}/`, the row was created under that patient, and the room
was `patient_<id>` — so a caregiver adding a spare phone from the sensor
roster, where no patient is in context, had to be asked "who is this
for?" before they could even mint a code. Two device classes, two
different answers to the same question.

Coverage is the one place attribution belongs. This endpoint mints
against the account and takes coverage as an ordinary, optional field:
pass a patient and the phone arrives watching them, pass nobody and it
arrives unassigned exactly like an installed sensor. Same concept, same
screen, whichever kind of device it is.

The phone still learns its room exactly once, at redemption, and never
re-reads it. That constraint was always about the ROOM, never about
attribution; conflating the two is what kept pairing patient-shaped.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services import livekit_rooms
from test.conftest import mint_test_jwt

_PATIENT = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
            "relation": "parent"}


async def _account(client: AsyncClient, auth_user) -> tuple[dict, str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    return headers, uid


async def _mint(client: AsyncClient, headers, **body):
    return await client.post("/v1/cameras/pairing-code", headers=headers,
                             json={"room": "Kitchen", **body})


@pytest.mark.asyncio
async def test_a_code_can_be_minted_with_no_patient_at_all(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The whole point: the roster has no patient in context."""
    headers, _uid = await _account(client, auth_user)
    r = await _mint(client, headers)
    assert r.status_code == 200, r.text
    assert len(r.json()["pairing_code"]) == 6


@pytest.mark.asyncio
async def test_the_room_is_the_account_room(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Not `patient_<id>`. There may be no patient to name."""
    headers, uid = await _account(client, auth_user)
    cam_id = (await _mint(client, headers)).json()["camera_id"]

    room = (await session.execute(
        text("SELECT livekit_room FROM public.cameras WHERE id = :c"),
        {"c": cam_id})).scalar_one()
    assert room == livekit_rooms.account_room(uid)


@pytest.mark.asyncio
async def test_coverage_is_set_when_a_patient_is_named(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """From a patient's Live view the phone should arrive watching them,
    with no second step."""
    headers, _uid = await _account(client, auth_user)
    pid = (await client.post("/v1/patients", headers=headers,
                             json=_PATIENT)).json()["id"]

    cam_id = (await _mint(client, headers, patient_ids=[pid])).json()["camera_id"]

    covered = (await session.execute(
        text("SELECT patient_id::text FROM public.camera_patients "
             " WHERE camera_id = :c"), {"c": cam_id})).scalars().all()
    assert covered == [pid]


@pytest.mark.asyncio
async def test_a_phone_for_nobody_is_left_uncovered_and_still_pairs(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Unassigned-and-waiting is the state the roster flow creates."""
    headers, _uid = await _account(client, auth_user)
    cam_id = (await _mint(client, headers)).json()["camera_id"]

    covered = (await session.execute(
        text("SELECT count(*) FROM public.camera_patients WHERE camera_id = :c"),
        {"c": cam_id})).scalar_one()
    assert covered == 0


@pytest.mark.asyncio
async def test_another_caregivers_patient_is_refused(
    client: AsyncClient, auth_user, auth_user_factory, session: AsyncSession,
) -> None:
    """Coverage is the attribution boundary, so it is also where an
    account boundary can be crossed. Naming someone else's resident must
    not silently create a sensor watching them."""
    headers, _uid = await _account(client, auth_user)
    other = await auth_user_factory()
    other_headers, _ = await _account(client, other)
    stranger = (await client.post("/v1/patients", headers=other_headers,
                                  json=_PATIENT)).json()["id"]

    r = await _mint(client, headers, patient_ids=[stranger])
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_re_minting_supersedes_an_abandoned_attempt(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Generating a code and never entering it leaves an orphan row. A
    second attempt at the same name must replace it, not 409 — the
    caregiver would otherwise be permanently unable to use a name they
    never actually paired."""
    headers, _uid = await _account(client, auth_user)
    first = (await _mint(client, headers)).json()["camera_id"]
    second = await _mint(client, headers)
    assert second.status_code == 200, second.text
    assert second.json()["camera_id"] != first

    survivors = (await session.execute(
        text("SELECT count(*) FROM public.cameras "
             " WHERE lower(room) = 'kitchen' AND source_type = 'iphone'"))
    ).scalar_one()
    assert survivors == 1, "the abandoned attempt should be gone"


@pytest.mark.asyncio
async def test_a_name_already_taken_by_a_PAIRED_sensor_is_refused(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Two sensors on one account may not share a name. Only a real,
    completed pairing blocks — that is what separates this from the
    supersede case above."""
    headers, _uid = await _account(client, auth_user)
    cam_id = (await _mint(client, headers)).json()["camera_id"]
    # Complete the pairing the way redemption does.
    await session.execute(
        text("UPDATE public.cameras SET api_key_hash = 'x' WHERE id = :c"),
        {"c": cam_id})
    await session.commit()

    second = await _mint(client, headers)
    assert second.status_code == 409, second.text
