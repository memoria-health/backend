"""Caregiver device-token registration.

`POST /v1/me/devices` is called by the caregiver app on launch and on
APNs token refresh. Idempotent on `apns_token` — replays from the same
device update timestamps without creating duplicate rows.
"""

from fastapi import APIRouter, status

from src.api.deps import CaregiverAuth, SessionDep
from src.repos import caregiver_device_repo
from src.schemas.devices import DeviceResponse, RegisterDeviceRequest

router = APIRouter(prefix="/v1/me", tags=["devices"])


@router.post(
    "/devices",
    response_model=DeviceResponse,
    status_code=status.HTTP_200_OK,
)
async def post_device(
    auth: CaregiverAuth,
    body: RegisterDeviceRequest,
    session: SessionDep,
) -> DeviceResponse:
    """Register or refresh an APNs device token for the caller.

    Returns 200 with the canonical row whether the row was newly
    inserted or updated — iOS doesn't need to distinguish.
    """
    row = await caregiver_device_repo.upsert_by_token(
        session,
        caregiver_id=auth.id,
        apns_token=body.apns_token,
        bundle_id=body.bundle_id,
        environment=body.environment,
    )
    await session.commit()
    return DeviceResponse(
        id=row.id,
        caregiver_id=row.caregiver_id,
        apns_token=row.apns_token,
        bundle_id=row.bundle_id,
        environment=row.environment,  # type: ignore[arg-type]
        created_at=row.created_at,
        updated_at=row.updated_at,
    )
