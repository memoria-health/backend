"""One event per clip (overlapping-detector dedup).

When a single action satisfies several detectors at once (a fall is also
a general emergency; agitation overlaps calling-for-help) the pipeline
must file exactly ONE event for that clip — the model's `primary_detection`
pick when present, else the highest-confidence detection — never one event
per matching detector.

These exercise the real DB firing path in `maybe_promote_analysis`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.detector import Detector
from src.services.detection.event_promotion import (
    _promotion_spec_for,
    maybe_promote_analysis,
)
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera


async def _two_firing_detectors(
    session: AsyncSession, patient_id,
) -> tuple[Detector, Detector]:
    """Two of the patient's auto-seeded, reactive (non-scheduled) detectors."""
    rows = (
        await session.execute(
            select(Detector)
            .where(Detector.patient_id == patient_id)
            .order_by(Detector.created_at)
        )
    ).scalars().all()
    reactive = [d for d in rows if d.is_on and d.shape != "scheduled"]
    assert len(reactive) >= 2, "need ≥2 reactive presets to test overlap"
    return reactive[0], reactive[1]


def _fields_for(det: Detector, confidence: float) -> dict:
    detected_field, conf_field, _base = _promotion_spec_for(det)
    return {detected_field: True, conf_field: confidence}


@pytest.mark.asyncio
async def test_two_detectors_one_clip_fires_only_the_model_pick(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Both detectors cross threshold; the model names the SECOND as the
    dominant action. Exactly one event fires, and it's that pick — even
    though the other detector reported HIGHER confidence.
    """
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    det_a, det_b = await _two_firing_detectors(session, patient_id)

    detector_fields = {
        # det_a is MORE confident, but the model says det_b dominates.
        **_fields_for(det_a, 0.95),
        **_fields_for(det_b, 0.80),
        "primary_detection": det_b.identifier,
        # No enrollment seeded → identity-agnostic, so promotion isn't gated.
        "identity_reid_ran": False,
    }
    result = await maybe_promote_analysis(
        session,
        patient_id=patient_id,
        camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc),
        description="one action, two detectors true",
        detector_fields=detector_fields,
    )
    await session.commit()

    assert len(result.fired) == 1, "exactly one event per clip"
    assert result.fired[0].detector_id == det_b.id, "model's pick wins"
    assert det_a.identifier in result.skipped_for_primary

    # And the DB agrees: only one event row exists for this patient.
    from src.db.event import Event
    events = (
        await session.execute(
            select(Event).where(Event.patient_id == patient_id)
        )
    ).scalars().all()
    assert len(events) == 1


@pytest.mark.asyncio
async def test_two_detectors_one_clip_without_hint_fires_highest_confidence(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """No usable `primary_detection` → the highest-confidence detection is
    the single event that fires; the other is suppressed.
    """
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    det_a, det_b = await _two_firing_detectors(session, patient_id)

    detector_fields = {
        **_fields_for(det_a, 0.91),   # highest
        **_fields_for(det_b, 0.74),
        "primary_detection": "",      # model didn't pick
        "identity_reid_ran": False,
    }
    result = await maybe_promote_analysis(
        session,
        patient_id=patient_id,
        camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc),
        description="one action, two detectors true, no model pick",
        detector_fields=detector_fields,
    )
    await session.commit()

    assert len(result.fired) == 1
    assert result.fired[0].detector_id == det_a.id
    assert det_b.identifier in result.skipped_for_primary
