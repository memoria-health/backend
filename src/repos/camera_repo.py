"""Repository for the camera aggregate.

Reads scoped through `patients.caregiver_id`. v0 only ever lists and
deletes — cameras are CREATED by the wall-phone pairing flow in a
later plan.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.patient import Patient
from src.repos.access import accessible_patient_ids


async def list_for_patient(
    session: AsyncSession, patient_id: UUID, *, caregiver_id: UUID
) -> Sequence[Camera]:
    """Return every camera for a caregiver-owned patient."""
    result = await session.execute(
        select(Camera)
        .join(Patient, Patient.id == Camera.patient_id)
        .where(
            Camera.patient_id == patient_id,
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
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
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
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
    ai_paused: bool | None = None,
    desired_state: str | None = None,
) -> Camera | None:
    """Partial-update a caregiver-owned camera. Returns the refreshed
    row, or None if the camera doesn't exist for this caregiver.

    Only the user-facing toggle fields are updatable here. Lifecycle
    fields (state, no_people_count, livekit_identity, pairing_code, …)
    are owned by the detector pipeline / pairing flow and are not
    touched by this method. In particular `online` is NOT written here:
    flipping `desired_state="disconnected"` makes the wall phone leave
    the LiveKit room, which fires a `participant_left` webhook that
    writes `online=false` via `camera_presence.mark_offline_by_identity`
    (the single owner of the column). The clients already render
    "offline" instantly off `desired_state`, so there's nothing to gain
    from a redundant write on this path — see `camera_presence.py` for
    the full trigger-point map.

    `desired_state` is validated at the API boundary (Pydantic
    Literal) AND at the DB layer (CHECK constraint), so by the time
    we reach here it's already guaranteed to be 'connected' or
    'disconnected' — no defensive validation needed.
    """
    cam = await find_by_id(session, camera_id, caregiver_id=caregiver_id)
    if cam is None:
        return None

    # Detect "resume" transitions BEFORE we mutate the flags. Either
    # disabled true→false (un-hide) or ai_paused true→false (resume AI)
    # is a user-initiated wake-up: they want AI to start running again
    # immediately, not "wait for motion to wake the closed-state
    # machine." Without this, even a brief pause that happened to span
    # ~6 ticks would leave state="closed" — and the closed-state check
    # in camera_ingest._run_tick would short-circuit the very next tick
    # on the resumed camera as "camera_closed."
    resuming = (
        (disabled is False and cam.disabled)
        or (ai_paused is False and cam.ai_paused)
    )

    if disabled is not None:
        cam.disabled = disabled
    if ai_paused is not None:
        cam.ai_paused = ai_paused
    if desired_state is not None:
        cam.desired_state = desired_state

    if resuming:
        cam.state = "open"
        cam.no_people_count = 0
        cam.state_changed_at = datetime.now(timezone.utc)

    await session.flush()
    return cam
