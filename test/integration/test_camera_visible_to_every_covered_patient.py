"""A shared camera must appear in EVERY patient it covers.

One physical camera streams into the primary patient's LiveKit room —
that is a cost decision, and it is right: a second patient does not need
a second copy of the same video. But it is a decision about LiveKit, and
it must not reach the app. If a caregiver assigns a camera to two
patients, the camera belongs in both patients' views, playing the same
stream.

`list_for_patient` filtered on `cameras.patient_id`, which holds the
PRIMARY, so the second patient's camera list came back empty.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt

_BODY = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
         "relation": "parent"}


async def _two_patients(client: AsyncClient, auth_user):
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    a = (await client.post("/v1/patients", headers=headers, json=_BODY)).json()
    b = (await client.post("/v1/patients", headers=headers,
                           json={**_BODY, "full_name": "Arthur O'Brien"})).json()
    return headers, email, a["id"], b["id"]


async def _shared_camera(client, email, headers, primary, also) -> dict:
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": "Bearer gw-shared-cam"},
        json={"email": email, "room": "Living Room"})
    assert r.status_code == 201, r.text
    camera = r.json()["cameras"][0]
    c = await client.post(
        f"/v1/cameras/{camera['camera_id']}/coverage", headers=headers,
        json={"primary_patient_id": primary, "also_patient_ids": [also]})
    assert c.status_code == 200, c.text
    return camera


async def _listed(client, headers, patient_id) -> list[dict]:
    r = await client.get(f"/v1/patients/{patient_id}/cameras", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_the_primary_patient_sees_the_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, email, primary, also = await _two_patients(client, auth_user)
    camera = await _shared_camera(client, email, headers, primary, also)
    ids = [c["id"] for c in await _listed(client, headers, primary)]
    assert camera["camera_id"] in ids


@pytest.mark.asyncio
async def test_the_second_covered_patient_sees_it_too(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The whole point. A caregiver who assigned this camera to Arthur
    opens Arthur and finds nothing there."""
    headers, email, primary, also = await _two_patients(client, auth_user)
    camera = await _shared_camera(client, email, headers, primary, also)
    ids = [c["id"] for c in await _listed(client, headers, also)]
    assert camera["camera_id"] in ids


@pytest.mark.asyncio
async def test_both_patients_are_pointed_at_the_SAME_room(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """One camera, one stream, one LiveKit participant. The second
    patient watches the primary's room rather than a room of their own,
    which is what keeps a shared camera from costing twice."""
    headers, email, primary, also = await _two_patients(client, auth_user)
    await _shared_camera(client, email, headers, primary, also)
    for_primary = await _listed(client, headers, primary)
    for_also = await _listed(client, headers, also)
    assert for_primary[0]["livekit_room"] == for_also[0]["livekit_room"]
    assert for_primary[0]["livekit_identity"] == for_also[0]["livekit_identity"]


@pytest.mark.asyncio
async def test_an_uncovered_patient_does_not_see_it(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Appearing everywhere is not the fix either."""
    headers, email, primary, also = await _two_patients(client, auth_user)
    third = (await client.post("/v1/patients", headers=headers,
                               json={**_BODY, "full_name": "Nora O'Brien"})).json()
    camera = await _shared_camera(client, email, headers, primary, also)
    ids = [c["id"] for c in await _listed(client, headers, third["id"])]
    assert camera["camera_id"] not in ids
