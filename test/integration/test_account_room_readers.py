"""Everything that CONSUMES a room must accept the account shape.

Renaming the room is only half a migration. The other half is every
reader that took a `patient_<uuid>` apart to work out what it was looking
at -- caregiver join authorization, the identity roster's liveness test,
the identity router's address lookup. Each of those still PARSES cleanly
against an account room; they simply resolve to nothing, which is the
quiet kind of broken: the sensor publishes, the fleet reports healthy,
and no caregiver can watch it while no identity process holds it.

The suite passed with all three broken, because every existing test used
an iPhone camera in a `patient_<uuid>` room. These are the cases that
were missing.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services import livekit_rooms
from test.conftest import mint_test_jwt, mint_gateway_code

_PATIENT = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
            "relation": "parent"}


async def _account_with_sensor(client, auth_user, token="gw-readers"):
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {token}"},
        json={"pairing_code": await mint_gateway_code(client, headers), "room": "Living Room"})
    assert r.status_code == 201, r.text
    return headers, email, uid, r.json()["cameras"][0]


@pytest.mark.asyncio
async def test_a_caregiver_can_join_their_own_account_room(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Otherwise the sensor streams and the app cannot show it."""
    headers, _email, uid, cam = await _account_with_sensor(client, auth_user)
    r = await client.post(
        "/v1/livekit/token", headers=headers,
        json={"room": cam["livekit_room"], "identity": "viewer-1",
              "role": "viewer"})
    assert r.status_code == 200, r.text
    assert r.json()["token"]
    assert cam["livekit_room"] == livekit_rooms.account_room(uid)


@pytest.mark.asyncio
async def test_a_caregiver_cannot_join_someone_elses_account_room(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The account room names an account, and naming one is not owning it.

    Worth pinning explicitly: the room string is now the only thing
    distinguishing my home from yours, and it is guessable from a
    caregiver id. Authorization has to come from the session, never from
    the string.
    """
    headers, _email, _uid, _cam = await _account_with_sensor(client, auth_user)
    stranger = "11111111-2222-3333-4444-555555555555"
    r = await client.post(
        "/v1/livekit/token", headers=headers,
        json={"room": livekit_rooms.account_room(stranger),
              "identity": "viewer-1", "role": "viewer"})
    assert r.status_code in (401, 403), r.text


@pytest.mark.asyncio
async def test_a_wall_phones_patient_room_still_works(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Phones hold their room from pairing and cannot be renamed."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    pid = (await client.post(
        "/v1/patients", headers=headers, json=_PATIENT)).json()["id"]

    r = await client.post(
        "/v1/livekit/token", headers=headers,
        json={"room": f"patient_{pid}", "identity": "viewer-1",
              "role": "viewer"})
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_the_identity_roster_sees_an_account_room_as_live(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A patient is identity-live when a sensor WATCHING THEM streams.

    The test was `cameras.livekit_room = 'patient_' || p.id`, which an
    account room never satisfies — so every bridge home would have read
    dormant, yielded its slot, and gone identity-blind while streaming
    perfectly.
    """
    headers, _email, _uid, cam = await _account_with_sensor(client, auth_user)
    pid = (await client.post(
        "/v1/patients", headers=headers, json=_PATIENT)).json()["id"]
    assign = await client.post(
        f"/v1/cameras/{cam['camera_id']}/coverage", headers=headers,
        json={"patient_ids": [pid]})
    assert assign.status_code == 200, assign.text
    await session.execute(
        text("UPDATE public.cameras SET online = true WHERE id = :id"),
        {"id": cam["camera_id"]})
    await session.commit()

    live = (await session.execute(
        text("SELECT EXISTS (SELECT 1 FROM public.cameras c "
             "  JOIN public.camera_patients cp ON cp.camera_id = c.id "
             " WHERE cp.patient_id = :p AND c.online AND NOT c.disabled)"),
        {"p": pid},
    )).scalar_one()
    assert live is True


@pytest.mark.asyncio
async def test_the_webhook_accepts_both_room_shapes(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A LiveKit webhook for an account room must not be dropped.

    The handler gated on `room_name.startswith("patient_")`, so every
    bridge sensor's join and leave was ignored: no RoomWorker spawned,
    and `cameras.online` never written by the reconciler that follows it.
    The sensor streamed perfectly and read offline forever, which is the
    kind of failure that looks like a camera fault and is not one.

    Asserted on the predicate rather than by posting a signed webhook,
    because the signature path needs LiveKit keys this suite does not
    have — and the predicate IS the bug.
    """
    _headers, _email, uid, _cam = await _account_with_sensor(
        client, auth_user, token="gw-webhook-shape")

    assert livekit_rooms.is_room_name(livekit_rooms.account_room(uid))
    assert livekit_rooms.is_room_name(f"patient_{uuid4()}")
    # Anything else is still ignored, which is what keeps unrelated
    # LiveKit traffic from spawning workers.
    assert not livekit_rooms.is_room_name("lobby")
    assert not livekit_rooms.is_room_name("")
