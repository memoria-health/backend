"""Regression tests: after the REST-polled reconciler refactor,
`apply_tick_update` MUST NOT write to `cameras.online`. That column is
owned exclusively by `LiveKitRoomReconciler` (see room_reconciler.py),
which derives presence from REST polls + an in-memory fresh-frame
timestamp.

Why the explicit regression net: it would be very easy to "re-add" the
online write to the state machine UPDATE in some future change ("just
stamp it on every tick — what's the harm?"), and that change would
silently re-couple stream-presence to detector ticks. These tests fail
loudly the moment that happens.

(The previously-tested `cameras.last_seen_at` column was dropped in
`bb2ad6f refactor(cameras): drop last_seen_at column and writers` —
presence freshness is now an in-memory signal, not a DB column.)
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.services.detection.state_machine import apply_tick_update
from test.conftest import mint_test_jwt

from datetime import datetime, timezone


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
    state: str = "open",
    no_people_count: int = 0,
    close_threshold: int = 6,
) -> UUID:
    """Insert a camera with a CALLER-CONTROLLED `online` so the test can
    verify the state machine doesn't rewrite it."""
    cid = uuid4()
    now = datetime.now(timezone.utc)
    await session.execute(
        text(
            """
            INSERT INTO cameras (
                id, patient_id, display_id, room, icon_name, online,
                created_at, state, no_people_count, state_changed_at,
                close_threshold
            ) VALUES (
                :id, :pid, 1, 'Living room', 'tv', :online,
                :now, :state, :nc, :now, :ct
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
    cid = await _insert_camera(session, patient_id=pid, online=False)

    await apply_tick_update(
        session, camera_id=cid, people_present=True, motion_in_clip=False,
    )
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.online is False
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
    cid = await _insert_camera(session, patient_id=pid, online=True)

    await apply_tick_update(
        session, camera_id=cid, people_present=False, motion_in_clip=False,
    )
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.online is True


@pytest.mark.asyncio
async def test_apply_tick_update_state_transition_works_with_online_false(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """End-to-end check: an offline camera still goes through normal
    open → closed transition rules. Detection logic is unaffected by
    the column-write split.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid, online=False)

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
    # Stream-presence column untouched throughout the full sequence.
    assert cam.online is False
