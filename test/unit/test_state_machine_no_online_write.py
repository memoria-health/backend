"""Regression tests: after the REST-polled reconciler refactor,
`apply_tick_update` MUST NOT write to `cameras.online` or
`cameras.last_seen_at`. Those columns are owned exclusively by
`LiveKitRoomReconciler` (see room_reconciler.py).

Why the explicit regression net: it would be very easy to "re-add"
those columns to the state machine UPDATE in some future change
("just stamp last_seen_at on every tick — what's the harm?"), and
that change would silently re-couple stream-presence to detector
ticks. These tests fail loudly the moment that happens.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.services.detection.state_machine import apply_tick_update
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(client: AsyncClient, auth_user) -> str:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return p["id"]


async def _insert_camera(
    session: AsyncSession,
    *,
    patient_id: str,
    online: bool,
    last_seen_at: datetime | None,
    state: str = "open",
    no_people_count: int = 0,
    close_threshold: int = 6,
) -> UUID:
    """Insert a camera with a CALLER-CONTROLLED `online` and
    `last_seen_at` so the test can verify the state machine doesn't
    rewrite either one.
    """
    cid = uuid4()
    now = datetime.now(timezone.utc)
    await session.execute(
        text(
            """
            INSERT INTO cameras (
                id, patient_id, display_id, room, icon_name, online,
                created_at, state, no_people_count, state_changed_at,
                close_threshold, last_seen_at
            ) VALUES (
                :id, :pid, 1, 'Living room', 'tv', :online,
                :now, :state, :nc, :now, :ct, :lsa
            )
            """
        ),
        {
            "id": str(cid),
            "pid": patient_id,
            "now": now,
            "state": state,
            "nc": no_people_count,
            "ct": close_threshold,
            "online": online,
            "lsa": last_seen_at,
        },
    )
    await session.commit()
    return cid


@pytest.mark.asyncio
async def test_apply_tick_update_does_not_set_online_true(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Seed `online=false`; run a tick; assert online stays false.

    The reconciler is now the sole authority on `cameras.online`. A
    detector tick on an offline camera (e.g. a tick that completes
    just before the reconciler catches up) must not pre-emptively
    flip the column.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(
        session, patient_id=pid, online=False, last_seen_at=None,
    )

    await apply_tick_update(
        session, camera_id=cid, people_present=True, motion_in_clip=False,
    )
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.online is False
    assert cam.last_seen_at is None
    # State machine SHOULD still have transitioned (people_present
    # forces state='open' and no_people_count=0).
    assert cam.state == "open"
    assert cam.no_people_count == 0


@pytest.mark.asyncio
async def test_apply_tick_update_does_not_clear_online_true(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Seed `online=true`; run a tick; assert online stays true."""
    pid = await _setup_patient(client, auth_user)
    seeded_last_seen = datetime.now(timezone.utc) - timedelta(seconds=5)
    cid = await _insert_camera(
        session, patient_id=pid, online=True, last_seen_at=seeded_last_seen,
    )

    await apply_tick_update(
        session, camera_id=cid, people_present=False, motion_in_clip=False,
    )
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.online is True
    # `last_seen_at` must not have moved either way — state machine
    # doesn't read or write it.
    assert cam.last_seen_at is not None
    assert abs((cam.last_seen_at - seeded_last_seen).total_seconds()) < 0.01


@pytest.mark.asyncio
async def test_apply_tick_update_does_not_set_last_seen_at_when_null(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Seed `last_seen_at=NULL`; run several ticks; assert it stays NULL."""
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(
        session, patient_id=pid, online=True, last_seen_at=None,
    )

    for _ in range(3):
        await apply_tick_update(
            session, camera_id=cid, people_present=False, motion_in_clip=False,
        )
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.last_seen_at is None


@pytest.mark.asyncio
async def test_apply_tick_update_state_transition_works_with_online_false(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """End-to-end check: an offline camera still goes through normal
    open → closed transition rules. Detection logic is unaffected by
    the column-write split.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(
        session, patient_id=pid, online=False, last_seen_at=None,
    )

    # Drive 6 (people=false) ticks to reach the close threshold.
    last = None
    for _ in range(6):
        last = await apply_tick_update(
            session, camera_id=cid,
            people_present=False, motion_in_clip=False,
        )
    await session.commit()

    assert last is not None
    assert last.state == "closed"
    assert last.no_people_count == 6

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "closed"
    # Stream-presence columns untouched throughout the full sequence.
    assert cam.online is False
    assert cam.last_seen_at is None
