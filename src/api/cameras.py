"""Camera API surface.

- GET    /v1/patients/{id}/cameras  — list.
- DELETE /v1/cameras/{id}           — caregiver-initiated removal.

Cameras are CREATED by the wall-phone pairing flow (a later plan).
The iOS app never POSTs cameras.
"""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CaregiverAuth, SessionDep
from src.db.camera import Camera
from src.repos import camera_repo
from src.schemas.camera import CameraResponse

router = APIRouter(prefix="/v1", tags=["camera"])


def _to_response(c: Camera) -> CameraResponse:
    return CameraResponse.model_validate(c, from_attributes=True)


@router.get(
    "/patients/{patient_id}/cameras",
    response_model=list[CameraResponse],
)
async def list_cameras(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[CameraResponse]:
    rows = await camera_repo.list_for_patient(session, patient_id, caregiver_id=auth.id)
    return [_to_response(c) for c in rows]


@router.delete(
    "/cameras/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_camera(
    camera_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> None:
    deleted = await camera_repo.delete_by_id(session, camera_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera_not_found",
        )
    await session.commit()
