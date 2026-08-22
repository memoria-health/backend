"""State machine unit tests (integration-style: real DB row).

Verbatim port of
`nemara-temp-dashboard/tests/unit/state-machine.test.mjs` semantics.
The Python implementation reads/writes the `cameras` row directly via
the ORM, so each test needs a real camera. We use the same `auth_user`
+ `session` fixtures the rest of the suite uses.
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
from test.conftest import mint_test_jwt, seed_camera

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
    close_threshold: int = 6,
    state: str = "open",
    no_people_count: int = 0,
) -> UUID:
    """This suite's state-machine defaults, on the shared seeder."""
    now = datetime.now(timezone.utc)
    return await seed_camera(
        session, patient_id=patient_id, room="Living room",
        state=state, no_people_count=no_people_count,
        close_threshold=close_threshold, state_changed_at=now,
    )


@pytest.mark.asyncio
async def test_open_to_closed_when_no_people_count_reaches_threshold(
    client: AsyncClient, auth_user, session
) -> None:
    """Six consecutive (people=false, motion=false) ticks should
    transition from open → closed at exactly the threshold and stamp
    state_changed_at.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid)

    pre = await session.get(Camera, cid)
    assert pre is not None
    initial_state_changed_at = pre.state_changed_at
    assert pre.state == "open"
    assert pre.no_people_count == 0
    assert pre.close_threshold == 6

    # Drive ticks 1..5 — should stay open with counter rising.
    last = None
    for expected_count in range(1, 6):
        last = await apply_tick_update(
            session, camera_id=cid, people_present=False, motion_in_clip=False,
        )
        assert last.state == "open"
        assert last.no_people_count == expected_count
        assert last.close_threshold == 6

    # 6th tick: open → closed at exactly threshold.
    last = await apply_tick_update(
        session, camera_id=cid, people_present=False, motion_in_clip=False,
    )
    assert last.state == "closed"
    assert last.no_people_count == 6
    assert last.close_threshold == 6
    await session.commit()

    # Verify DB-side state mirrors the returned dataclass and
    # state_changed_at was updated on the transition.
    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "closed"
    assert cam.no_people_count == 6
    assert cam.close_threshold == 6
    assert cam.state_changed_at is not None
    assert initial_state_changed_at is None or cam.state_changed_at > initial_state_changed_at


@pytest.mark.asyncio
async def test_closed_to_open_on_people_present(
    client: AsyncClient, auth_user, session
) -> None:
    """A closed camera with people_present=true (motion=false) must
    transition to open and reset the counter to 0.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(
        session, patient_id=pid, state="closed", no_people_count=6,
    )

    res = await apply_tick_update(
        session, camera_id=cid, people_present=True, motion_in_clip=False,
    )
    assert res.state == "open"
    assert res.no_people_count == 0
    assert res.close_threshold == 6
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "open"
    assert cam.no_people_count == 0
    assert cam.close_threshold == 6


@pytest.mark.asyncio
async def test_closed_with_motion_but_no_people_stays_closed(
    client: AsyncClient, auth_user, session
) -> None:
    """Motion alone never opens the camera. When state=closed and the
    pipeline verdict is people_present=False, the camera stays closed
    with the counter clamped — even if motion fired (a false-positive
    wake-up). This is the post-a785c31 contract: the LLM/Re-ID verdict
    is the sole gate, motion is just the wake-up trigger.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(
        session, patient_id=pid, state="closed", no_people_count=6,
    )

    res = await apply_tick_update(
        session, camera_id=cid, people_present=False, motion_in_clip=True,
    )
    assert res.state == "closed"
    assert res.no_people_count == 6
    assert res.close_threshold == 6
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "closed"
    assert cam.no_people_count == 6
    assert cam.close_threshold == 6


@pytest.mark.asyncio
async def test_closed_counter_clamps_at_threshold(
    client: AsyncClient, auth_user, session
) -> None:
    """Once closed, additional idle ticks must NOT push the counter
    past the threshold (no runaway growth).
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(
        session, patient_id=pid, state="closed", no_people_count=6,
    )

    for _ in range(5):
        res = await apply_tick_update(
            session, camera_id=cid, people_present=False, motion_in_clip=False,
        )
        assert res.state == "closed"
        assert res.no_people_count == 6
        assert res.close_threshold == 6
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "closed"
    assert cam.no_people_count == 6
    assert cam.close_threshold == 6


@pytest.mark.asyncio
async def test_no_people_count_resets_to_zero_on_activity(
    client: AsyncClient, auth_user, session
) -> None:
    """Counter must reset to 0 on the first tick with activity, even
    if state was already open and the counter was mid-rise.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid)

    # Push the counter up but stay open.
    for expected_count in range(1, 4):
        res = await apply_tick_update(
            session, camera_id=cid, people_present=False, motion_in_clip=False,
        )
        assert res.state == "open"
        assert res.no_people_count == expected_count
        assert res.close_threshold == 6

    # Activity tick resets.
    res = await apply_tick_update(
        session, camera_id=cid, people_present=True, motion_in_clip=False,
    )
    assert res.state == "open"
    assert res.no_people_count == 0
    assert res.close_threshold == 6
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "open"
    assert cam.no_people_count == 0


@pytest.mark.asyncio
async def test_state_changed_at_only_updates_on_transition(
    client: AsyncClient, auth_user, session
) -> None:
    """state_changed_at is touched only when state actually flips. Idle
    open→open ticks (counter rising) must NOT bump the column.
    """
    pid = await _setup_patient(client, auth_user)
    cid = await _insert_camera(session, patient_id=pid)
    # Seed state_changed_at to a known past value so we can verify
    # whether subsequent ticks touched it.
    seed_ts = datetime.now(timezone.utc) - timedelta(hours=1)
    cam = await session.get(Camera, cid)
    assert cam is not None
    cam.state_changed_at = seed_ts
    await session.commit()

    # Five no-activity ticks — counter goes to 5, state stays open,
    # state_changed_at must NOT be touched.
    for expected_count in range(1, 6):
        res = await apply_tick_update(
            session, camera_id=cid, people_present=False, motion_in_clip=False,
        )
        assert res.state == "open"
        assert res.no_people_count == expected_count
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "open"
    assert cam.no_people_count == 5
    # Tolerate DB tz-round-tripping by comparing the unix epoch.
    assert cam.state_changed_at is not None
    assert int(cam.state_changed_at.timestamp()) == int(seed_ts.timestamp())

    # Sixth tick crosses the threshold → state flips → ts must update.
    res = await apply_tick_update(
        session, camera_id=cid, people_present=False, motion_in_clip=False,
    )
    assert res.state == "closed"
    assert res.no_people_count == 6
    await session.commit()

    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.state == "closed"
    assert cam.state_changed_at is not None
    assert cam.state_changed_at > seed_ts


@pytest.mark.asyncio
async def test_apply_tick_raises_for_unknown_camera(session) -> None:
    """A bug-shaped call (auth path didn't filter) must raise rather
    than silently no-op.
    """
    with pytest.raises(ValueError):
        await apply_tick_update(
            session,
            camera_id=uuid4(),
            people_present=False,
            motion_in_clip=False,
        )
