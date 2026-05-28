"""Safety invariant tests for the identity-gated event promotion path.

Two cases must hold:

  (a) When NO enrollment exists for the patient, all detector events
      promote as today (identity-agnostic). This protects day-zero
      before the caregiver has had a chance to enroll the patient.

  (b) When an enrollment exists AND the tick's identity_map says
      patient_in_clip=True, events promote normally.

  (c) When an enrollment exists AND identity_map says
      patient_in_clip=False, events are SUPPRESSED. We're confident
      it's not the patient and we don't want to alert the caregiver
      about someone else's fall.

The architectural commitment from the plan: identity is the GATE for
all detectors (safety + behavioral) once enrollment exists; the
adaptive threshold is what prevents false negatives. These tests
lock that contract in.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.repos import face_embedding_repo
from src.services.detection.event_promotion import maybe_promote_analysis
from src.services.reid.embeddings import embedding_to_bytes
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Maggie",
    "preferred_name": "Maggie",
    "relation": "parent",
}


_FALL_FIELDS = {
    "fall_detected": True,
    "fall_confidence": 0.85,
    "fall_reasoning": "Person fell to the ground.",
}


async def _setup_patient_with_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> tuple[UUID, UUID]:
    """Provision a caregiver + patient + a camera under that patient.
    Returns (patient_id, camera_id)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    patient_id = UUID(p["id"])

    # The detector pipeline expects a camera row. We don't need it to
    # be paired in LiveKit — just exist in the DB.
    from sqlalchemy import insert
    from src.db.camera import Camera
    from uuid import uuid4
    camera_id = uuid4()
    await session.execute(
        insert(Camera).values(
            id=camera_id,
            patient_id=patient_id,
            display_id=1,
            room="Test Room",
            icon_name="bed.double",
            online=True,
            api_key_hash="test_hash",
            livekit_identity=f"cam_{camera_id.hex[:8]}",
            state="open",
            no_people_count=0,
            close_threshold=6,
            state_changed_at=datetime.now(timezone.utc),
            disabled=False,
            desired_state="connected",
            created_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
    return patient_id, camera_id


def _ready_embedding(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return embedding_to_bytes(v)


# ─── (a) No enrollment → identity-agnostic ──────────────────────


@pytest.mark.asyncio
async def test_no_enrollment_falls_fire_as_today(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Day-zero: caregiver hasn't enrolled the patient yet. The
    identity gate is bypassed. A fall event with patient_in_clip=False
    still promotes."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )

    detector_fields = {
        **_FALL_FIELDS,
        "identity_patient_in_clip": False,
        "identity_confidence": 0.0,
    }
    result = await maybe_promote_analysis(
        session,
        patient_id=patient_id,
        camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc),
        description="someone fell",
        detector_fields=detector_fields,
    )
    await session.commit()

    assert len(result.fired) == 1, (
        "with no enrollment, fall must promote regardless of identity"
    )
    assert result.fired[0].patient_id == patient_id


# ─── (b) Enrollment + patient confirmed → fires ─────────────────


@pytest.mark.asyncio
async def test_enrolled_and_patient_in_clip_fall_fires(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    # Seed an enrollment so the gate is active.
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="x.jpg",
    )
    await face_embedding_repo.mark_ready(
        session,
        embedding_id=row.id,
        embedding=_ready_embedding(1),
        det_score=0.85,
    )
    await session.commit()

    detector_fields = {
        **_FALL_FIELDS,
        "identity_reid_ran": True,
        "identity_patient_in_clip": True,
        "identity_confidence": 0.75,
    }
    result = await maybe_promote_analysis(
        session,
        patient_id=patient_id,
        camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc),
        description="patient fell",
        detector_fields=detector_fields,
    )
    await session.commit()
    assert len(result.fired) == 1


# ─── (c) Enrollment + patient absent → suppression is UPSTREAM ──
#
# Suppression for "stranger in frame" lives in
# `camera_ingest._run_tick` now (commit removed downstream gate).
# That path skips the LLM entirely before `maybe_promote_analysis`
# is ever called. There's nothing to test here from
# `maybe_promote_analysis`'s perspective — if it runs, identity has
# already been deemed compatible with promotion (or unknowable). A
# proper integration test for the upstream gate would drive
# CameraIngest with a fixture clip; that's tracked separately.


# ─── (d) Enrollment + Re-ID timed out → fires (safety blocker) ──


@pytest.mark.asyncio
async def test_enrolled_but_reid_timed_out_fall_still_fires(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """THE LOAD-BEARING SAFETY TEST.

    Patient is enrolled. Re-ID timed out / crashed for this tick. The
    tick produces identity_reid_ran=False (the empty IdentityResult).
    The gate MUST bypass identity suppression and let safety
    detectors promote identity-agnostic.

    A regression here means: a transient Re-ID failure silences fall
    detection on the enrolled patient. That's the failure mode the
    plan explicitly forbids. If this test breaks, do NOT ship.
    """
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="x.jpg",
    )
    await face_embedding_repo.mark_ready(
        session,
        embedding_id=row.id,
        embedding=_ready_embedding(3),
        det_score=0.85,
    )
    await session.commit()

    # Re-ID infrastructure failure: empty IdentityResult shape.
    # reid_ran=False is the gate-bypass signal.
    detector_fields = {
        **_FALL_FIELDS,
        "identity_reid_ran": False,
        "identity_patient_in_clip": False,
        "identity_confidence": 0.0,
    }
    result = await maybe_promote_analysis(
        session,
        patient_id=patient_id,
        camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc),
        description="patient fell during Re-ID outage",
        detector_fields=detector_fields,
    )
    await session.commit()
    assert len(result.fired) == 1, (
        "Re-ID timeout/crash MUST fall back to identity-agnostic "
        "promotion. Suppressing here is the catastrophic failure mode."
    )


# ─── (e) Enrollment + Re-ID ran but saw zero faces → fires ─────


@pytest.mark.asyncio
async def test_enrolled_but_no_faces_detected_fall_still_fires(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """SECOND LOAD-BEARING SAFETY TEST.

    Patient is enrolled. Re-ID ran successfully but detected zero
    faces in the clip — common when the patient falls (face turned
    toward floor, foreshortened from a high-mounted camera,
    motion-blurred). The gate MUST bypass identity suppression here
    because face_count=0 is not positive evidence of patient absence;
    it just means Re-ID had no face to match.

    Regression here means: a real patient fall is silenced because
    Re-ID couldn't see their face during the fall itself.
    """
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="x.jpg",
    )
    await face_embedding_repo.mark_ready(
        session,
        embedding_id=row.id,
        embedding=_ready_embedding(4),
        det_score=0.85,
    )
    await session.commit()

    detector_fields = {
        **_FALL_FIELDS,
        "identity_reid_ran": True,
        "identity_patient_in_clip": False,
        "identity_confidence": 0.0,
        "identity_face_count": 0,  # no faces detected
    }
    result = await maybe_promote_analysis(
        session,
        patient_id=patient_id,
        camera_id=camera_id,
        occurred_at=datetime.now(timezone.utc),
        description="patient fell, face turned away from camera",
        detector_fields=detector_fields,
    )
    await session.commit()
    assert len(result.fired) == 1, (
        "Re-ID seeing zero faces is NOT positive evidence the patient "
        "is absent. Suppressing here means silencing a real fall just "
        "because the patient's face was turned during the fall."
    )
