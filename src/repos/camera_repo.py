"""Repository for the camera aggregate.

Reads scoped through `patients.caregiver_id`. v0 only ever lists and
deletes — cameras are CREATED by the wall-phone pairing flow in a
later plan.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.patient import Patient
from src.repos.access import accessible_patient_ids


async def list_for_patient(
    session: AsyncSession, patient_id: UUID, *, caregiver_id: UUID
) -> Sequence[Camera]:
    """Every camera that WATCHES this patient, not just the ones it owns.

    A camera covers 0..N patients (`public.camera_patients`) and streams
    into the PRIMARY patient's LiveKit room only — one camera, one
    stream, one participant. That is a cost decision and it is the right
    one: a second patient does not need a second copy of the same video.

    But it is a decision about LiveKit, and it must not reach the app.
    This filtered on `cameras.patient_id`, which holds the primary, so a
    caregiver who assigned a camera to two patients opened the second one
    and found nothing there. Both of them get the camera now, pointed at
    the same room.
    """
    covered = (await session.execute(
        text("SELECT camera_id FROM public.camera_patients "
             " WHERE patient_id = :p"),
        {"p": patient_id},
    )).scalars().all()

    result = await session.execute(
        select(Camera)
        # Scoped on the camera's PRIMARY patient, which is always one of
        # this caregiver's — coverage cannot cross accounts, because
        # set_coverage will not link a patient the caller does not own.
        .join(Patient, Patient.id == Camera.patient_id)
        .where(
            or_(Camera.patient_id == patient_id, Camera.id.in_(covered)),
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
        )
        .order_by(Camera.display_id.asc())
    )
    return result.scalars().all()


async def find_by_id(
    session: AsyncSession, camera_id: UUID, *, caregiver_id: UUID
) -> Camera | None:
    """A sensor on this caregiver's account, assigned or not.

    Scoped by ACCOUNT, mirroring the `cameras_member` RLS policy. It was
    an INNER JOIN on the patient, so an unassigned sensor — patient_id
    NULL — joined nothing and every per-sensor control 404'd: rename,
    disable, desired_state, the audio kill, delete. The sensor streamed
    and could not be turned off, which is the opposite of what a privacy
    control is for.
    """
    result = await session.execute(
        select(Camera).where(
            Camera.id == camera_id,
            Camera.caregiver_id == caregiver_id,
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
    room: str | None = None,
    audio_capture_enabled: bool | None = None,
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
    if audio_capture_enabled is not None:
        cam.audio_capture_enabled = audio_capture_enabled
    if room is not None and room != cam.room:
        # Room rename → re-classify (room-privacy escape hatch ⚑3): the
        # caregiver saying "this is actually the Living Room" is the ONE
        # user signal that the room verdict may be wrong. Clearing
        # room_classified_at re-arms the classifier on the next frames;
        # room_class + privacy_enabled are KEPT until the new verdict
        # lands (fail-closed — a bathroom stays private through the
        # rename window rather than briefly unlocking). NOTE: the
        # privacy fields themselves are NOT writable here — this is a
        # side effect of the cosmetic rename, never a direct set.
        cam.room = room
        cam.room_class_confidence = None
        cam.room_classified_at = None

    if resuming:
        cam.state = "open"
        cam.no_people_count = 0
        cam.state_changed_at = datetime.now(timezone.utc)

    await session.flush()
    return cam
