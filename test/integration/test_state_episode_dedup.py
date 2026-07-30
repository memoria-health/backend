"""Continuous ("state") detector rising-edge episode dedup.

A `state` detector (e.g. "watching TV") is true across many consecutive
ticks. It must file ONE event at the start of an episode and then stay
quiet for as long as the activity continues, firing again only after
positive evidence has stopped for the category exit gap
(`decision.episode_exit_gap_for` — 300s routine / 60s safety).
point_in_time detectors are unaffected (covered elsewhere).

Note the exit gap is NOT `episode_gap_seconds`; that column is no longer
read (2026-07-29). An episode's length is bounded by the behaviour, not
by a per-detector duration estimate.
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
from test.integration.identity_helpers import _setup_patient_with_camera

_IDENT = "watching_tv"
# Routine exit gap (decision.EXIT_GAP_ROUTINE_SECONDS). Kept local so the
# timings below read explicitly rather than importing a constant the
# assertions then can't be checked against.
_EXIT_GAP = 300


async def _add_state_detector(session: AsyncSession, patient_id) -> Detector:
    now = datetime.now(timezone.utc)
    det = Detector(
        id=uuid4(), patient_id=patient_id, name="Watching TV", icon_name="tv",
        description="Watching TV", category="routine", source="custom",
        alert_trigger="leading", is_sustained=True,
        when_scope="ALWAYS",
        sensitivity="standard", is_on=True, created_at=now, updated_at=now,
        identifier=_IDENT, prompt_fragment=None, schema_json=None,
        # Deliberately set to a value the code must IGNORE: if anything
        # still read this column, the 600s here would change the timings
        # the tests below assert against the 300s routine exit gap.
        always_on=False, episode_gap_seconds=600,
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


async def _add_prior_event(
    session: AsyncSession, det: Detector, patient_id, camera_id,
    *, occurred_at: datetime,
) -> None:
    """Seed the event that was filed at this episode's start."""
    session.add(
        Event(
            id=uuid4(), patient_id=patient_id, detector_id=det.id,
            camera_id=camera_id, category="routine", source="custom",
            icon_name="tv", title="Possible Watching TV", location="Living Room",
            occurred_at=occurred_at, created_at=occurred_at,
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_ongoing_episode_is_suppressed(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Evidence still arriving AND we already filed an event for this
    episode → ongoing → no second event, however long it runs."""
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det = await _add_state_detector(session, patient_id)
    now = datetime.now(timezone.utc)
    await _add_prior_analysis(
        session, patient_id, camera_id, detected=True, captured_at=now - timedelta(seconds=60),
    )
    await _add_prior_event(
        session, det, patient_id, camera_id, occurred_at=now - timedelta(seconds=60),
    )

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="still watching tv", detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 0
    assert _IDENT in result.skipped_ongoing
    # Still exactly the one event from the episode start.
    from sqlalchemy import select
    rows = (await session.execute(select(Event).where(Event.patient_id == patient_id))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_long_episode_never_refires_however_sparse_the_evidence(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The 2026-07-29 regression: a session that keeps producing evidence
    must stay ONE event even when the evidence is intermittent.

    Live, a Watching-TV session filed 5 events in 25 minutes because the
    old rule counted only ABOVE-THRESHOLD ticks in a sliding window; the
    count drained during ordinary confidence wobble and re-armed entry.
    Here every other tick is sub-threshold, and gaps stay under the
    routine exit gap, so it must remain a single episode.
    """
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det = await _add_state_detector(session, patient_id)
    now = datetime.now(timezone.utc)
    start = now - timedelta(seconds=1500)          # 25 minutes of session
    await _add_prior_event(
        session, det, patient_id, camera_id, occurred_at=start,
    )
    # Alternate 0.9 / 0.4 every 60s: continuous behaviour, wobbling score.
    age = 1500
    flip = True
    while age > 0:
        await _add_prior_analysis(
            session, patient_id, camera_id, detected=True,
            confidence=0.9 if flip else 0.4,
            captured_at=now - timedelta(seconds=age),
        )
        flip = not flip
        age -= 60

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="still watching tv", detector_fields=_FIELDS,
    )
    await session.commit()
    assert result.fired == [], "a continuing episode must not file a second event"
    assert _IDENT in result.skipped_ongoing


@pytest.mark.asyncio
async def test_evidence_without_a_prior_event_still_fires(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Prior above-threshold evidence but NO event on record → fire.

    Suppression now asks "did we already tell the caregiver about this
    episode", not "was there a promotable tick recently". If the first
    tick of an episode lost the one-event-per-clip arbitration to a
    higher-severity detector, the caregiver was never told — so the next
    tick should still be allowed to file, rather than the whole episode
    going silent.
    """
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det = await _add_state_detector(session, patient_id)
    now = datetime.now(timezone.utc)
    await _add_prior_analysis(
        session, patient_id, camera_id, detected=True, confidence=0.9,
        captured_at=now - timedelta(seconds=60),
    )

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="watching tv", detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 1
    assert result.fired[0].detector_id == det.id


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
        captured_at=now - timedelta(seconds=_EXIT_GAP + 120),  # past the exit gap
    )

    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="watching tv again", detector_fields=_FIELDS,
    )
    await session.commit()
    assert len(result.fired) == 1
    assert result.fired[0].detector_id == det.id
