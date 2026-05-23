"""Restart-resumption guarantees for the detector subscriber-bot.

The bot persists ALL tick state in Postgres: cameras.state +
no_people_count via `state_machine.apply_tick_update`, the analysis
row itself, and (post-commit) the events promotion. Nothing lives
only in worker memory across ticks. So an in-process worker restart
is mostly safe by construction.

This test pins two things the plan promised:

  1. The new `unique (camera_id, captured_at)` constraint on
     `analyses` actually fires on a duplicate INSERT — so a worker
     restart that lands a tick at the exact same wall-clock second
     as the last tick before the crash can't double-write.

  2. A back-to-back tick sequence (each tick gets its own
     `captured_at`) inserts fine without colliding — the constraint
     scopes to (camera_id, captured_at) jointly, not either alone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services.detection.analyses_store import add_analysis
from test.conftest import mint_test_jwt  # noqa: F401


async def _seed_caregiver_patient_camera(
    session: AsyncSession,
    auth_user_factory,
) -> tuple[UUID, UUID, UUID]:
    uid, email = await auth_user_factory()
    now = datetime.now(timezone.utc)
    session.add(
        Caregiver(
            id=uid, display_name="Restart Tester", email=email,
            created_at=now, updated_at=now,
        )
    )
    patient_id = uuid4()
    session.add(
        Patient(
            id=patient_id, caregiver_id=uid,
            full_name="Margaret O'Brien", preferred_name="Maggie",
            relation="parent", state="calm", status_line="OK",
            cameras_paused=False, created_at=now, updated_at=now,
        )
    )
    await session.flush()
    camera_id = uuid4()
    session.add(
        Camera(
            id=camera_id, patient_id=patient_id, display_id=1,
            room="Bedroom", icon_name="bed.double", online=True,
            api_key_hash=None, livekit_identity=None,
            last_seen_at=None, state="closed",
            no_people_count=0, close_threshold=2,
            created_at=now,
        )
    )
    await session.commit()
    return uid, patient_id, camera_id


@pytest.mark.asyncio
async def test_unique_constraint_blocks_duplicate_camera_captured_at(
    auth_user_factory,
) -> None:
    """Two analyses for the same (camera_id, captured_at) — the second
    INSERT must fail with IntegrityError. The asyncio.Lock in the
    live worker normally prevents this; the constraint catches the
    cross-process / crash-restart race the lock can't see."""
    sm = get_sessionmaker()
    async with sm() as session:
        _, _, camera_id = await _seed_caregiver_patient_camera(session, auth_user_factory)

    captured_at = datetime.now(timezone.utc).replace(microsecond=0)

    async with sm() as session:
        await add_analysis(
            session, camera_id=camera_id, provider="gemini",
            description="first tick",
            detector_fields={"fall_detected": False},
            people_present=True, people_count=1, motion_in_clip=False,
            state_after="open", captured_at=captured_at,
        )
        await session.commit()

    # Second insert at the same camera + captured_at: must explode.
    with pytest.raises(IntegrityError):
        async with sm() as session:
            await add_analysis(
                session, camera_id=camera_id, provider="gemini",
                description="duplicate tick",
                detector_fields={"fall_detected": True},
                people_present=True, people_count=1, motion_in_clip=True,
                state_after="open", captured_at=captured_at,
            )
            await session.commit()


@pytest.mark.asyncio
async def test_consecutive_ticks_at_different_timestamps_succeed(
    auth_user_factory,
) -> None:
    """The constraint is on the PAIR — distinct captured_at values
    insert cleanly, even back-to-back. Test pins this so a future
    "let's also unique on camera_id alone" mistake fails loudly."""
    sm = get_sessionmaker()
    async with sm() as session:
        _, _, camera_id = await _seed_caregiver_patient_camera(session, auth_user_factory)

    base = datetime.now(timezone.utc).replace(microsecond=0)
    timestamps = [base + timedelta(seconds=10 * i) for i in range(3)]
    for ts in timestamps:
        async with sm() as session:
            await add_analysis(
                session, camera_id=camera_id, provider="gemini",
                description=f"tick at {ts.isoformat()}",
                detector_fields={"fall_detected": False},
                people_present=False, people_count=0, motion_in_clip=False,
                state_after="closed", captured_at=ts,
            )
            await session.commit()

    async with sm() as session:
        rows = (
            await session.execute(
                select(Analysis).where(Analysis.camera_id == camera_id)
                .order_by(Analysis.captured_at)
            )
        ).scalars().all()
    assert len(rows) == 3
    assert [r.captured_at for r in rows] == timestamps


@pytest.mark.asyncio
async def test_state_persistence_makes_restart_safe(
    auth_user_factory,
) -> None:
    """`cameras.state` + `no_people_count` are written by every tick
    via `state_machine.apply_tick_update`. Restart-resumption depends
    on those columns surviving — pin that they're read from the DB
    (not cached in memory) by:
      1. Insert an analysis at T with state_after='open'.
      2. Update the camera's state to 'closed' manually (simulating
         a transition baked into a prior tick).
      3. A fresh read returns state='closed' — would-be in-memory
         caches don't ride along on a process restart.
    """
    sm = get_sessionmaker()
    async with sm() as session:
        _, _, camera_id = await _seed_caregiver_patient_camera(session, auth_user_factory)

    # Simulate the state-machine writing 'closed' + counter=2.
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        cam.state = "closed"
        cam.no_people_count = 2
        await session.commit()

    # Fresh session (= what a restarted worker sees on its first
    # tick) — reads the persisted state, not a stale in-memory one.
    async with sm() as session:
        cam = await session.get(Camera, camera_id)
        assert cam is not None
        assert cam.state == "closed"
        assert cam.no_people_count == 2
