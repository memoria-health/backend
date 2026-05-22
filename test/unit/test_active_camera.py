"""Tests for `get_active_camera`.

What matters is the predicate: the right row wins, stale rows are
excluded, and `people_present=false` rows are excluded. Uses the same
HTTP-driven setup the rest of `test/unit/` uses so we don't have to
duplicate auth.users wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.services.active_camera import get_active_camera
from test.conftest import mint_test_jwt


async def _setup_patient_with_camera(
    client: AsyncClient, auth_user,
) -> tuple[UUID, UUID, UUID, UUID]:
    """Returns (caregiver_id, patient_id, camera_id_a, camera_id_b).
    Both cameras are paired against the same patient.
    """
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "Margaret O'Brien", "preferred_name": "Maggie", "relation": "parent",
    })).json()
    pid = UUID(p["id"])

    async def _pair(room: str) -> UUID:
        mint = (await client.post(
            f"/v1/patients/{pid}/cameras/pairing-code",
            headers=headers, json={"room": room, "icon_name": "tv"},
        )).json()
        pair = (await client.post(
            "/v1/cameras/pair", json={"pairing_code": mint["pairing_code"]},
        )).json()
        return UUID(pair["camera_id"])

    cam_a = await _pair("Kitchen")
    cam_b = await _pair("Bedroom")
    return uid, pid, cam_a, cam_b


def _make_analysis(
    *,
    caregiver_id: UUID,
    patient_id: UUID,
    camera_id: UUID,
    captured_at: datetime,
    people_present: bool,
) -> Analysis:
    return Analysis(
        id=uuid4(),
        caregiver_id=caregiver_id,
        patient_id=patient_id,
        camera_id=camera_id,
        provider="gemini",
        description="test",
        detector_fields={},
        people_present=people_present,
        motion_in_clip=False,
        state_after="open",
        captured_at=captured_at,
        created_at=captured_at,
    )


@pytest.mark.asyncio
async def test_returns_none_when_no_rows(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    _, pid, _, _ = await _setup_patient_with_camera(client, auth_user)
    result = await get_active_camera(session, patient_id=pid)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_no_people_present(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A row with people_present=false should NOT qualify, even if
    it's the most recent. The detector saw no humans — Talk has
    nothing to route to.
    """
    cid, pid, cam_a, _ = await _setup_patient_with_camera(client, auth_user)
    now = datetime.now(timezone.utc)
    session.add(_make_analysis(
        caregiver_id=cid, patient_id=pid, camera_id=cam_a,
        captured_at=now - timedelta(seconds=5),
        people_present=False,
    ))
    await session.commit()
    result = await get_active_camera(session, patient_id=pid)
    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_too_old(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A row >30s old should be excluded — beyond the two-tick
    "still present but quiet" window.
    """
    cid, pid, cam_a, _ = await _setup_patient_with_camera(client, auth_user)
    now = datetime.now(timezone.utc)
    session.add(_make_analysis(
        caregiver_id=cid, patient_id=pid, camera_id=cam_a,
        captured_at=now - timedelta(seconds=31),
        people_present=True,
    ))
    await session.commit()
    result = await get_active_camera(session, patient_id=pid)
    assert result is None


@pytest.mark.asyncio
async def test_returns_camera_with_freshest_qualifying_row(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Two cameras both have qualifying rows; the freshest one wins."""
    cid, pid, cam_a, cam_b = await _setup_patient_with_camera(client, auth_user)
    now = datetime.now(timezone.utc)
    # cam_a saw a person 20s ago (within window but older)
    session.add(_make_analysis(
        caregiver_id=cid, patient_id=pid, camera_id=cam_a,
        captured_at=now - timedelta(seconds=20),
        people_present=True,
    ))
    # cam_b saw a person 5s ago (freshest)
    session.add(_make_analysis(
        caregiver_id=cid, patient_id=pid, camera_id=cam_b,
        captured_at=now - timedelta(seconds=5),
        people_present=True,
    ))
    await session.commit()
    result = await get_active_camera(session, patient_id=pid)
    assert result == cam_b


@pytest.mark.asyncio
async def test_explicit_now_pins_the_window(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The `now` parameter lets tests freeze the cutoff so the same
    row can be tested as qualifying / stale across moments in time.
    """
    cid, pid, cam_a, _ = await _setup_patient_with_camera(client, auth_user)
    fixed_now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    session.add(_make_analysis(
        caregiver_id=cid, patient_id=pid, camera_id=cam_a,
        captured_at=fixed_now - timedelta(seconds=10),
        people_present=True,
    ))
    await session.commit()

    # Pin the clock at the moment that makes the row 10s old → qualifies.
    result_recent = await get_active_camera(session, patient_id=pid, now=fixed_now)
    assert result_recent == cam_a

    # Roll forward 25s → same row is now 35s old → no longer qualifies.
    result_stale = await get_active_camera(
        session, patient_id=pid, now=fixed_now + timedelta(seconds=25),
    )
    assert result_stale is None
