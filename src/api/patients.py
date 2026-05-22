"""Patient API surface — `/v1/patients/*` endpoints.

- GET    /v1/patients         — list the caregiver's patients.
- POST   /v1/patients         — create. Auto-seeds preset detectors.
- GET    /v1/patients/{id}    — fetch one.
- PATCH  /v1/patients/{id}    — partial update.
- DELETE /v1/patients/{id}    — remove (cascades to children).

All endpoints require CaregiverAuth. 404 (not 403) for cross-account
attempts so a probing client can't discover patient IDs.
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CaregiverAuth, SessionDep
from src.db.patient import Patient
from src.repos import patient_repo
from src.schemas.patient import (
    CreatePatientRequest,
    PatientResponse,
    UpdatePatientRequest,
)
from src.services import onboarding

router = APIRouter(prefix="/v1/patients", tags=["patient"])


def _to_response(p: Patient) -> PatientResponse:
    return PatientResponse.model_validate(p, from_attributes=True)


@router.get("", response_model=list[PatientResponse])
async def list_patients(
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[PatientResponse]:
    rows = await patient_repo.list_for_caregiver(session, auth.id)
    return [_to_response(p) for p in rows]


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_patient(
    auth: CaregiverAuth,
    body: CreatePatientRequest,
    session: SessionDep,
) -> PatientResponse:
    """Create a patient under the current caregiver.

    Auto-seeds the four preset detectors (Falls / Agitation / Wandering
    / Medication adherence) via `services.onboarding.seed_detectors_for_patient`
    in the same transaction. Lands in step 7 of the build sequence;
    until then, new patients have an empty detector list.
    """
    now = datetime.now(timezone.utc)
    p = Patient(
        id=uuid4(),
        caregiver_id=auth.id,
        full_name=body.full_name,
        preferred_name=body.preferred_name,
        relation=body.relation.value,
        state="calm",
        status_line="All quiet",
        cameras_paused=False,
        created_at=now,
        updated_at=now,
        date_of_birth=body.date_of_birth,
        gender=body.gender,
        home_address=body.home_address,
        last_event_at=None,
        battery_percent=None,
    )
    await patient_repo.insert(session, p)
    await onboarding.seed_detectors_for_patient(session, p.id)
    await session.commit()
    return _to_response(p)


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> PatientResponse:
    p = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )
    return _to_response(p)


@router.patch("/{patient_id}", response_model=PatientResponse)
async def patch_patient(
    patient_id: UUID,
    auth: CaregiverAuth,
    body: UpdatePatientRequest,
    session: SessionDep,
) -> PatientResponse:
    """Partial update.

    The Pydantic model only exposes caregiver-writable fields, so even
    a payload that tries to set `state` / `status_line` / `last_event_at`
    / `battery_percent` is rejected as 422 by `extra="forbid"`. Those
    columns transition via wall-phone telemetry (not in scope here).
    """
    fields = body.model_dump(exclude_unset=True)
    # Enum → string for the ORM string columns.
    if "relation" in fields and fields["relation"] is not None:
        fields["relation"] = fields["relation"].value
    p = await patient_repo.update_partial(
        session, patient_id, caregiver_id=auth.id, fields=fields
    )
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )
    await session.commit()
    return _to_response(p)


@router.delete(
    "/{patient_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None
)
async def delete_patient(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> None:
    """Remove a caregiver-owned patient. Cascades to cameras /
    detectors / events via DB-level `ON DELETE CASCADE`.
    """
    deleted = await patient_repo.delete_by_id(session, patient_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )
    await session.commit()
