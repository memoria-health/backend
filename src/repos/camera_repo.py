"""Repository for the camera aggregate.

Reads scoped through `patients.caregiver_id`. v0 only ever lists and
deletes — cameras are CREATED by the wall-phone pairing flow in a
later plan.
"""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.patient import Patient


async def list_for_patient(
    session: AsyncSession, patient_id: UUID, *, caregiver_id: UUID
) -> Sequence[Camera]:
    """Return every camera for a caregiver-owned patient."""
    result = await session.execute(
        select(Camera)
        .join(Patient, Patient.id == Camera.patient_id)
        .where(
            Camera.patient_id == patient_id,
            Patient.caregiver_id == caregiver_id,
        )
        .order_by(Camera.display_id.asc())
    )
    return result.scalars().all()


async def find_by_id(
    session: AsyncSession, camera_id: UUID, *, caregiver_id: UUID
) -> Camera | None:
    result = await session.execute(
        select(Camera)
        .join(Patient, Patient.id == Camera.patient_id)
        .where(
            Camera.id == camera_id,
            Patient.caregiver_id == caregiver_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_by_id(
    session: AsyncSession, camera_id: UUID, *, caregiver_id: UUID
) -> bool:
    """Delete a caregiver-owned camera. Returns True if deleted."""
    existing = await find_by_id(session, camera_id, caregiver_id=caregiver_id)
    if existing is None:
        return False
    await session.execute(delete(Camera).where(Camera.id == camera_id))
    await session.flush()
    return True


async def update_partial(
    session: AsyncSession,
    camera_id: UUID,
    *,
    caregiver_id: UUID,
    disabled: bool | None = None,
    desired_state: str | None = None,
) -> Camera | None:
    """Partial-update a caregiver-owned camera. Returns the refreshed
    row, or None if the camera doesn't exist for this caregiver.

    Only the user-facing toggle fields are updatable here. Lifecycle
    fields (state, no_people_count, livekit_identity, pairing_code, …)
    are owned by the detector pipeline / pairing flow and are not
    touched by this method.

    `desired_state` is validated at the API boundary (Pydantic
    Literal) AND at the DB layer (CHECK constraint), so by the time
    we reach here it's already guaranteed to be 'connected' or
    'disconnected' — no defensive validation needed.
    """
    cam = await find_by_id(session, camera_id, caregiver_id=caregiver_id)
    if cam is None:
        return None
    if disabled is not None:
        cam.disabled = disabled
    if desired_state is not None:
        cam.desired_state = desired_state
    await session.flush()
    return cam
