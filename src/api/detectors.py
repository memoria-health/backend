"""Detector API surface.

- GET    /v1/patients/{id}/detectors  — list.
- POST   /v1/patients/{id}/detectors  — create a custom detector.
- PATCH  /v1/detectors/{id}           — update (is_on, sensitivity, …).
- DELETE /v1/detectors/{id}           — remove.

All endpoints require CaregiverAuth and 404 (not 403) for cross-account
attempts.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CaregiverAuth, SessionDep
from src.db.detector import Detector
from src.repos import detector_repo, patient_repo
from src.schemas.detector import (
    CreateDetectorRequest,
    DetectorResponse,
    UpdateDetectorRequest,
)

router = APIRouter(prefix="/v1", tags=["detector"])


def _to_response(d: Detector) -> DetectorResponse:
    return DetectorResponse.model_validate(d, from_attributes=True)


@router.get(
    "/patients/{patient_id}/detectors",
    response_model=list[DetectorResponse],
)
async def list_detectors(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[DetectorResponse]:
    rows = await detector_repo.list_for_patient(session, patient_id, caregiver_id=auth.id)
    return [_to_response(d) for d in rows]


@router.post(
    "/patients/{patient_id}/detectors",
    response_model=DetectorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_detector(
    patient_id: UUID,
    auth: CaregiverAuth,
    body: CreateDetectorRequest,
    session: SessionDep,
) -> DetectorResponse:
    """Create a custom detector. Server forces `source = custom`."""
    # 404 if the parent patient isn't owned by the caller — never leak
    # patient existence.
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )

    now = datetime.now(timezone.utc)
    det = Detector(
        id=uuid4(),
        patient_id=patient_id,
        name=body.name,
        icon_name=body.icon_name,
        description=body.description,
        category=body.category.value,
        source="custom",
        shape=body.shape.value,
        where_scope=body.where_scope,
        when_scope=body.when_scope,
        sensitivity=body.sensitivity.value,
        is_on=True,
        created_at=now,
        updated_at=now,
        custom_prompt=body.custom_prompt,
    )
    await detector_repo.insert(session, det)
    await session.commit()
    return _to_response(det)


@router.patch(
    "/detectors/{detector_id}",
    response_model=DetectorResponse,
)
async def patch_detector(
    detector_id: UUID,
    auth: CaregiverAuth,
    body: UpdateDetectorRequest,
    session: SessionDep,
) -> DetectorResponse:
    fields = body.model_dump(exclude_unset=True)
    # Enum → string for the ORM string columns.
    for k in ("category", "shape", "sensitivity"):
        if k in fields and fields[k] is not None:
            fields[k] = fields[k].value
    det = await detector_repo.update_partial(
        session, detector_id, caregiver_id=auth.id, fields=fields
    )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="detector_not_found",
        )
    await session.commit()
    return _to_response(det)


@router.delete(
    "/detectors/{detector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_detector(
    detector_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> None:
    deleted = await detector_repo.delete_by_id(session, detector_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="detector_not_found",
        )
    await session.commit()
