"""One event per clip (overlapping-detector dedup).

When a single action satisfies several detectors at once (a fall is also
a general emergency; agitation overlaps calling-for-help) the pipeline
must file exactly ONE event for that clip — the model's `primary_detection`
pick when present, else the highest-confidence detection — never one event
per matching detector.

These exercise the real DB firing path in `maybe_promote_analysis`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.detector import Detector
from src.db.event import Event
from src.services.detection.event_promotion import (
    _promotion_spec_for,
    maybe_promote_analysis,
)
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera


async def _two_firing_detectors(
    session: AsyncSession, patient_id,
) -> tuple[Detector, Detector]:
    """Two EQUAL-SEVERITY custom detectors. Severity is ranking tier 1 in
    `_select_primary` (choking beats agitation no matter what the model or
    confidence say), so the hint/confidence tiers this file tests are only
    reachable between detectors of the same severity — catalog presets all
    differ, customs share the default. Tier-1 itself is covered by
    test/unit/test_event_primary_selection.py.
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    now = datetime.now(timezone.utc)
    dets = []
    for ident in ("custom_ovlap_a0", "custom_ovlap_b0"):
        det = Detector(
            id=uuid4(),
            created_at=now,
            updated_at=now,
            patient_id=patient_id,
            name=f"Custom {ident[-2:]}",
            icon_name="sparkles",
            description="overlap test detector",
            category="safety",
            source="custom",
            shape="point_in_time",
            where_scope="ALL ROOMS",
            when_scope="ALWAYS",
            sensitivity="standard",
            is_on=True,
            custom_prompt="overlap test",
            identifier=ident,
            prompt_fragment="overlap test fragment",
            schema_json={
                f"{ident}_detected": {"type": "boolean"},
                f"{ident}_confidence": {"type": "number"},
                f"{ident}_reasoning": {"type": "string"},
            },
            is_builtin=False,
        )
        session.add(det)
        dets.append(det)
    await session.commit()
    return dets[0], dets[1]


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


async def _add_prior_event(session, patient_id, camera_id, det, *, occurred_at) -> None:
    session.add(
        Event(
            id=uuid4(), patient_id=patient_id, detector_id=det.id, camera_id=camera_id,
            category=det.category, source=det.source, icon_name=det.icon_name,
            title="prior", location="Test Room", occurred_at=occurred_at, has_clip=False,
            clip_suppressed=False, narrative="prior", opened=False, reviewed=False,
            review_choice=None, review_notes=None, reviewed_at=None, created_at=occurred_at,
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_deduped_winner_falls_back_to_next_distinct_candidate(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """If the top pick is within its 30s dedup window (a repeat), the clip
    still fires the next-best DISTINCT candidate rather than no event at all.
    """
    patient_id, camera_id = await _setup_patient_with_camera(client, auth_user, session)
    det_a, det_b = await _two_firing_detectors(session, patient_id)
    now = datetime.now(timezone.utc)
    # det_a is the higher-confidence pick, but it already fired 10s ago.
    await _add_prior_event(session, patient_id, camera_id, det_a, occurred_at=now - timedelta(seconds=10))

    detector_fields = {
        **_fields_for(det_a, 0.95),
        **_fields_for(det_b, 0.80),
        "primary_detection": "",
        "identity_reid_ran": False,
    }
    result = await maybe_promote_analysis(
        session, patient_id=patient_id, camera_id=camera_id,
        occurred_at=now, description="top pick is a repeat", detector_fields=detector_fields,
    )
    await session.commit()
    assert len(result.fired) == 1, "should still fire the distinct candidate"
    assert result.fired[0].detector_id == det_b.id
    assert det_a.identifier in result.skipped_for_dedup


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
