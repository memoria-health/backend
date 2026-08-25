"""Integration tests for per-camera setup self-test arming.

The setup test proves one camera's alert path end to end. It used to be
fire-once per PATIENT: the first pass flipped the single shared detector
row to is_on=false, so a second camera in the same home could never run
the test at all. Arming now lives on `cameras.setup_test_passed_at`:

  - every camera runs the test until IT passes,
  - passing stamps only that camera,
  - a restart (participant_joined) clears the stamp and re-arms it,
  - and the stamp gates the setup test ONLY — no other detector, and no
    other behaviour, reads it.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.detection.registry import get_enabled_detectors_for_camera
from test.conftest import mint_test_jwt

_PATIENT_BODY = {
    "full_name": "Jane Doe",
    "preferred_name": "Jane",
    "relation": "parent",
}


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _create_camera(
    client: AsyncClient, headers: dict[str, str], patient_id: str, room: str,
) -> str:
    r = await client.post(
        f"/v1/patients/{patient_id}/cameras/pairing-code",
        headers=headers,
        json={"room": room},
    )
    assert r.status_code == 200, r.text
    return r.json()["camera_id"]


async def _identifiers_for(session: AsyncSession, camera_id: str) -> set[str]:
    specs = await get_enabled_detectors_for_camera(session, camera_id)
    return {s.identifier for s in specs}


async def _stamp_passed(session: AsyncSession, camera_id: str) -> None:
    await session.execute(
        text(
            "UPDATE cameras SET setup_test_passed_at = now() WHERE id = :c"
        ),
        {"c": camera_id},
    )
    await session.commit()


@pytest.mark.asyncio
async def test_setup_test_is_armed_on_a_fresh_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    cam = await _create_camera(client, headers, pid, "LIVING ROOM")
    assert "setup_test" in await _identifiers_for(session, cam)


@pytest.mark.asyncio
async def test_passing_disarms_only_that_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The regression this whole change exists for: the second camera used
    to lose its test the moment the first one passed."""
    headers, pid = await _setup_patient(client, auth_user)
    first = await _create_camera(client, headers, pid, "LIVING ROOM")
    second = await _create_camera(client, headers, pid, "BEDROOM")

    await _stamp_passed(session, first)
    session.expire_all()

    assert "setup_test" not in await _identifiers_for(session, first)
    assert "setup_test" in await _identifiers_for(session, second)


@pytest.mark.asyncio
async def test_passing_leaves_every_other_detector_running(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The stamp arms the setup test and nothing else — it must never
    read as a gate on the detectors that actually watch the patient."""
    headers, pid = await _setup_patient(client, auth_user)
    # A real detector has to exist for "leaves every OTHER detector
    # running" to mean anything. Patient create no longer seeds any
    # (1cd354e), so without this the assertion passes vacuously against
    # an empty loadout — which is the very thing it is guarding against.
    await client.post(
        f"/v1/patients/{pid}/detectors/from-library",
        headers=headers, json={"keys": ["fall"]},
    )
    cam = await _create_camera(client, headers, pid, "LIVING ROOM")
    before = await _identifiers_for(session, cam)

    await _stamp_passed(session, cam)
    session.expire_all()

    after = await _identifiers_for(session, cam)
    assert after == before - {"setup_test"}
    assert after, "a passed setup test must not empty the loadout"


@pytest.mark.asyncio
async def test_restart_rearms_the_test(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A rebooted camera gets to prove its alert path again — the whole
    point of clearing the stamp on participant_joined."""
    from src.services.detection import camera_presence

    headers, pid = await _setup_patient(client, auth_user)
    cam = await _create_camera(client, headers, pid, "LIVING ROOM")
    await _stamp_passed(session, cam)
    session.expire_all()
    assert "setup_test" not in await _identifiers_for(session, cam)

    identity = (
        await session.execute(
            text("SELECT livekit_identity FROM cameras WHERE id = :c"),
            {"c": cam},
        )
    ).scalar_one()

    # The helper opens its own session from the sessionmaker; hand it the
    # test's so the write lands on the same transaction the assertions read.
    class _SessionMaker:
        def __call__(self):
            return _Ctx()

    class _Ctx:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *args):
            return None

    rearmed = await camera_presence.rearm_setup_test_by_identity(
        _SessionMaker(), identity,
    )
    assert rearmed is True
    session.expire_all()
    assert "setup_test" in await _identifiers_for(session, cam)
