"""Repository for the cameras_detectors join.

Reads + writes are scoped through `cameras.patient_id → patients.caregiver_id`.
RLS is also configured (defense in depth), but the app layer is the
load-bearing check — same pattern as the other repos.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.camera_detector import CameraDetector
from src.db.detector import Detector
from src.db.patient import Patient


async def list_detector_ids_for_camera(
    session: AsyncSession, camera_id: UUID, *, caregiver_id: UUID,
) -> Sequence[UUID]:
    """Return the detector IDs assigned to `camera_id`.

    Joined through cameras → patients so a request for a camera not
    owned by `caregiver_id` returns empty.
    """
    stmt = (
        select(CameraDetector.detector_id)
        .join(Camera, Camera.id == CameraDetector.camera_id)
        .join(Patient, Patient.id == Camera.patient_id)
        .where(
            CameraDetector.camera_id == camera_id,
            Patient.caregiver_id == caregiver_id,
        )
    )
    return (await session.execute(stmt)).scalars().all()


async def replace_assignments_for_camera(
    session: AsyncSession,
    camera_id: UUID,
    detector_ids: Sequence[UUID],
    *,
    caregiver_id: UUID,
) -> Sequence[UUID]:
    """Replace the camera's assignment set with exactly `detector_ids`.

    Validates that every detector belongs to the same patient as the
    camera (no cross-patient leakage). Returns the final set on success;
    raises ValueError if the camera doesn't belong to the caregiver or
    any detector_id isn't valid for this patient.
    """
    cam_stmt = (
        select(Camera.id, Camera.patient_id)
        .join(Patient, Patient.id == Camera.patient_id)
        .where(Camera.id == camera_id, Patient.caregiver_id == caregiver_id)
    )
    cam_row = (await session.execute(cam_stmt)).one_or_none()
    if cam_row is None:
        raise ValueError(f"camera {camera_id} not owned by caregiver {caregiver_id}")
    patient_id = cam_row.patient_id

    requested = list(dict.fromkeys(detector_ids))  # dedupe, preserve order
    if requested:
        valid_stmt = select(Detector.id).where(
            Detector.id.in_(requested),
            Detector.patient_id == patient_id,
        )
        valid_ids = set((await session.execute(valid_stmt)).scalars().all())
        bad = [d for d in requested if d not in valid_ids]
        if bad:
            raise ValueError(
                f"detectors {bad} do not belong to patient {patient_id}; "
                f"refusing to assign across patients"
            )

    await session.execute(
        delete(CameraDetector).where(CameraDetector.camera_id == camera_id)
    )
    if requested:
        now = datetime.now(timezone.utc)
        await session.execute(
            insert(CameraDetector),
            [{"camera_id": camera_id, "detector_id": d, "created_at": now}
             for d in requested],
        )
    await session.flush()
    return requested


async def assign_detector_to_cameras(
    session: AsyncSession,
    detector_id: UUID,
    patient_id: UUID,
    *,
    room: str | None = None,
) -> int:
    """Assign `detector_id` to cameras of `patient_id`, optionally
    filtered by room name.

    - `room=None` → every camera under the patient (used when the
      detector's where_rooms is empty / every room).
    - `room="Kitchen"` → every camera whose `room` column matches
      exactly. Multiple cameras can share a room name (e.g. two
      angles in the kitchen) — they all get assigned.

    Idempotent: ON CONFLICT DO NOTHING so re-running the same
    assignment is safe. Returns number of rows inserted.

    This exists so the detector form's "Where" dropdown does the
    obvious thing — pick a room, the detector starts running on that
    camera(s) without the caregiver having to navigate to
    /cameras/[id] afterwards.
    """
    stmt = select(Camera.id).where(Camera.patient_id == patient_id)
    if room is not None:
        stmt = stmt.where(Camera.room == room)
    cam_ids = (await session.execute(stmt)).scalars().all()
    if not cam_ids:
        return 0
    now = datetime.now(timezone.utc)
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    insert_stmt = pg_insert(CameraDetector).values(
        [{"camera_id": c, "detector_id": detector_id, "created_at": now}
         for c in cam_ids]
    ).on_conflict_do_nothing(
        index_elements=["camera_id", "detector_id"],
    )
    result = await session.execute(insert_stmt)
    return result.rowcount or 0


def rooms_mean_all(rooms: Sequence[str] | None) -> bool:
    """True when a room set means 'every room' — None, empty, or the
    "ALL ROOMS" sentinel present (case/space-insensitive). The single
    place that decision is made, shared by the endpoint + assignment."""
    if not rooms:
        return True
    return any((r or "").strip().upper() == "ALL ROOMS" for r in rooms)


async def set_detector_rooms(
    session: AsyncSession,
    detector_id: UUID,
    patient_id: UUID,
    rooms: Sequence[str] | None,
) -> int:
    """Replace `detector_id`'s camera assignments so it runs on exactly the
    cameras in `rooms` — or every camera when `rooms` means ALL ROOMS.

    Atomic replace: the detector's prior assignments are cleared and
    rebuilt from the cameras currently in the selected rooms, so
    deselecting a room stops it there and selecting a new one starts it in
    one operation. Returns the assignment count after the replace.

    Custom-detector room scoping only — it does a blanket delete of the
    detector's rows, which is safe because customs carry no per-camera
    `door_config` (that's the preset wandering picker, never room-scoped
    through here).
    """
    stmt = select(Camera.id).where(Camera.patient_id == patient_id)
    if not rooms_mean_all(rooms):
        stmt = stmt.where(Camera.room.in_(list(rooms)))
    cam_ids = (await session.execute(stmt)).scalars().all()

    await session.execute(
        delete(CameraDetector).where(CameraDetector.detector_id == detector_id)
    )
    if cam_ids:
        now = datetime.now(timezone.utc)
        await session.execute(
            insert(CameraDetector),
            [{"camera_id": c, "detector_id": detector_id, "created_at": now}
             for c in cam_ids],
        )
    await session.flush()
    return len(cam_ids)


async def assign_preset_detectors_of_patient_to_camera(
    session: AsyncSession, camera_id: UUID, patient_id: UUID,
) -> int:
    """Assign every PRESET detector under `patient_id` to `camera_id`.

    Used when a new camera is paired. Preset detectors (Falls,
    Wandering, etc.) are auto-seeded on patient create and the
    caregiver expects them to "just work" everywhere — so a freshly
    paired camera inherits them automatically.

    Custom detectors are intentionally excluded — they're opt-in per
    camera. A caregiver who created a 'stove_on' detector for the
    kitchen camera doesn't expect it to start running on the bedroom
    camera the moment the bedroom phone is paired. They opt the new
    camera in explicitly from the /cameras/[id] page.

    Returns number of rows inserted.
    """
    det_ids = (
        await session.execute(
            select(Detector.id).where(
                Detector.patient_id == patient_id,
                Detector.source == "preset",
            )
        )
    ).scalars().all()
    if not det_ids:
        return 0
    now = datetime.now(timezone.utc)
    from sqlalchemy.dialects.postgresql import insert as pg_insert
    stmt = pg_insert(CameraDetector).values(
        [{"camera_id": camera_id, "detector_id": d, "created_at": now}
         for d in det_ids]
    ).on_conflict_do_nothing(
        index_elements=["camera_id", "detector_id"],
    )
    result = await session.execute(stmt)
    return result.rowcount or 0
