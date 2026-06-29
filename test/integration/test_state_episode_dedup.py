"""Continuous ("state") detector rising-edge episode dedup.

A `state` detector (e.g. "watching TV") is true across many consecutive
ticks. It must file ONE event at the start of an episode and then stay
quiet while the activity is ongoing — only firing again after a gap of
absence longer than the detector's `episode_gap_seconds`. point_in_time
detectors are unaffected (covered elsewhere).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.services.detection.event_promotion import maybe_promote_analysis
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera

_IDENT = "watching_tv"
_GAP = 600  # 10 min


async def _add_state_detector(session: AsyncSession, patient_id) -> Detector:
    now = datetime.now(timezone.utc)
    det = Detector(
        id=uuid4(), patient_id=patient_id, name="Watching TV", icon_name="tv",
        description="Watching TV", category="routine", source="custom",
        shape="state", where_scope="ALL ROOMS", when_scope="ALWAYS",
        sensitivity="standard", is_on=True, created_at=now, updated_at=now,
        identifier=_IDENT, prompt_fragment=None, schema_json=None,
        always_on=False, is_builtin=False, episode_gap_seconds=_GAP,
    )
    session.add(det)
    await session.flush()
    return det


async def _add_prior_analysis(
    session: AsyncSession, patient_id, camera_id, *, detected: bool,
    captured_at: datetime, confidence: float = 0.9,
) -> None:
    caregiver_id = (await session.get(Patient, patient_id)).caregiver_id
    session.add(
        Analysis(
            id=uuid4(), caregiver_id=caregiver_id, patient_id=patient_id,
            camera_id=camera_id, provider="gemini", description="x",
            detector_fields={f"{_IDENT}_detected": detected, f"{_IDENT}_confidence": confidence},
            people_present=True, motion_in_clip=True, state_after="open",
            captured_at=captured_at, created_at=captured_at,
        )
    )
    await session.flush()


_FIELDS = {f"{_IDENT}_detected": True, f"{_IDENT}_confidence": 0.9, "identity_reid_ran": False}


@pytest.mark.asyncio
async def test_rising_edge_fires_one_event(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """No prior detection → the episode start fires exactly one event."""
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det = await _add_state_detector(session, patient_id)

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc), description="watching tv",
        detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 1
    assert result.fired[0].detector_id == det.id
    assert not result.skipped_ongoing


@pytest.mark.asyncio
async def test_ongoing_episode_is_suppressed(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A prior detected=true tick WITHIN the gap → ongoing → no new event."""
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    await _add_state_detector(session, patient_id)
    now = datetime.now(timezone.utc)
    await _add_prior_analysis(
        session, patient_id, camera_id, detected=True, captured_at=now - timedelta(seconds=60),
    )

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="still watching tv", detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 0
    assert _IDENT in result.skipped_ongoing
    # And no event row was written.
    from sqlalchemy import select
    rows = (await session.execute(select(Event).where(Event.patient_id == patient_id))).scalars().all()
    assert len(rows) == 0


@pytest.mark.asyncio
async def test_subthreshold_prior_does_not_suppress_first_event(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A prior detected=true tick BELOW threshold (never fired) must NOT count
    as an ongoing episode — the genuine first above-threshold event still fires.
    """
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det = await _add_state_detector(session, patient_id)
    now = datetime.now(timezone.utc)
    # Prior tick within the gap: detected=true but confidence 0.4 < standard
    # threshold 0.70, so it never promoted.
    await _add_prior_analysis(
        session, patient_id, camera_id, detected=True, confidence=0.4,
        captured_at=now - timedelta(seconds=60),
    )

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="now clearly watching tv", detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 1, "a sub-threshold blip must not swallow the real first event"
    assert result.fired[0].detector_id == det.id


@pytest.mark.asyncio
async def test_resumption_after_gap_fires_again(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A prior detection OLDER than the gap → new episode → fires again."""
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det = await _add_state_detector(session, patient_id)
    now = datetime.now(timezone.utc)
    await _add_prior_analysis(
        session, patient_id, camera_id, detected=True,
        captured_at=now - timedelta(seconds=_GAP + 120),  # well outside the gap
    )

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="watching tv again", detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 1
    assert result.fired[0].detector_id == det.id
