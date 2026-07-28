"""Repository for the cameras_detectors join.

Reads + writes are scoped through `cameras.patient_id → patients.caregiver_id`.
RLS is also configured (defense in depth), but the app layer is the
load-bearing check — same pattern as the other repos.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import delete, insert, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.camera_detector import CameraDetector
from src.db.detector import Detector
from src.db.patient import Patient
from src.repos.access import accessible_patient_ids


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
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
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
        .where(Camera.id == camera_id, Patient.id.in_(accessible_patient_ids(caregiver_id)))
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
        now = datetime.now(UTC)
        await session.execute(
            insert(CameraDetector),
            [{"camera_id": camera_id, "detector_id": d, "created_at": now}
             for d in requested],
        )
    await session.flush()
    return requested



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
        now = datetime.now(UTC)
        await session.execute(
            insert(CameraDetector),
            [{"camera_id": c, "detector_id": detector_id, "created_at": now}
             for c in cam_ids],
        )
    await session.flush()
    return len(cam_ids)


async def rewrite_where_rooms_from_links(
    session: AsyncSession,
    patient_id: UUID,
    *,
    detector_ids: set[UUID] | None = None,
) -> int:
    """Write each detector's `where_rooms` back FROM its current links.

    Inverse of `sync_links_for_patient`, for the one flow where the
    links are edited directly: the per-camera detector picker
    (PUT /cameras/{id}/detectors). Links are derived state, so a picker
    edit must be folded back into `where_rooms` or the next sync would
    revert it.

    Per detector (restricted to `detector_ids` when given):
      - custom + where_rooms IS NULL → skipped; opt-in links stay
        authoritative and sync never touches them.
      - otherwise: linked to every camera of the patient → where_rooms
        = NULL (all rooms); else the sorted distinct `room` values of
        the linked cameras. Preset linked to nothing → [] (runs
        nowhere, visible as such — NOT NULL, which would mean
        everywhere).

    Returns the number of detectors whose `where_rooms` changed.
    """
    detectors = (
        await session.execute(
            select(Detector).where(Detector.patient_id == patient_id)
        )
    ).scalars().all()
    cameras = (
        await session.execute(
            select(Camera.id, Camera.room).where(Camera.patient_id == patient_id)
        )
    ).all()
    room_by_cam = {c.id: c.room for c in cameras}
    all_cam_ids = set(room_by_cam)

    links = (
        await session.execute(
            select(CameraDetector.detector_id, CameraDetector.camera_id).where(
                CameraDetector.camera_id.in_(all_cam_ids)
                if all_cam_ids
                else CameraDetector.camera_id.is_(None)
            )
        )
    ).all()
    cams_by_det: dict[UUID, set[UUID]] = {}
    for det_id, cam_id in links:
        cams_by_det.setdefault(det_id, set()).add(cam_id)

    changed = 0
    now = datetime.now(UTC)
    for det in detectors:
        if detector_ids is not None and det.id not in detector_ids:
            continue
        if det.source != "preset" and det.where_rooms is None:
            continue  # opt-in custom — links are authoritative, hands off
        linked = cams_by_det.get(det.id, set())
        if all_cam_ids and linked == all_cam_ids:
            new_rooms = None
        else:
            new_rooms = sorted({room_by_cam[c] for c in linked})
        if new_rooms != det.where_rooms and not (
            new_rooms is None and rooms_mean_all(det.where_rooms)
        ):
            det.where_rooms = new_rooms
            det.updated_at = now
            changed += 1
    return changed


async def sync_links_for_patient(
    session: AsyncSession, patient_id: UUID,
) -> tuple[int, int]:
    """Reconcile `cameras_detectors` with each detector's `where_rooms`.

    The links are DERIVED state — `where_rooms` is the source of truth
    for where a detector runs — but they used to be maintained by
    scattered imperative calls on individual create/update paths, and
    drifted whenever a path was missed (gateway-created cameras, room
    renames, …). Field bug: an all-rooms preset with ZERO links, so its
    is_on toggle did nothing while the app showed it armed. This is the
    single idempotent repair pass, called from every camera/detector
    lifecycle mutation in the same transaction.

    Desired camera set per detector:
      - preset  + where_rooms meaning ALL (NULL / empty / "ALL ROOMS"
        sentinel, see `rooms_mean_all`) → every camera of the patient.
      - custom  + where_rooms IS NULL → SKIPPED ENTIRELY. Customs are
        opt-in per camera by design — a caregiver who made a 'stove_on'
        detector for the kitchen camera doesn't expect it on a freshly
        paired bedroom camera. A null-scoped custom's current links are
        the caregiver's authoritative opt-in set, never auto-grown or
        auto-pruned here.
      - any detector with explicit where_rooms (preset or custom) →
        cameras whose `room` matches an entry, exact string compare —
        the same semantics `set_detector_rooms` uses (the clients build the rooms picker
        from the cameras' own `room` values, so they match verbatim).

    Missing links are inserted (ON CONFLICT DO NOTHING); stale links are
    deleted. Existing in-scope rows are never rewritten, so per-camera
    `door_config` on kept links survives. Returns (inserted, deleted).
    """
    detectors = (
        await session.execute(
            select(Detector.id, Detector.source, Detector.where_rooms).where(
                Detector.patient_id == patient_id,
            )
        )
    ).all()
    cameras = (
        await session.execute(
            select(Camera.id, Camera.room).where(Camera.patient_id == patient_id)
        )
    ).all()

    all_cam_ids = {c.id for c in cameras}
    desired: set[tuple[UUID, UUID]] = set()
    managed_det_ids: list[UUID] = []
    for det in detectors:
        if det.source != "preset" and det.where_rooms is None:
            continue  # opt-in custom — links are authoritative, hands off
        managed_det_ids.append(det.id)
        if rooms_mean_all(det.where_rooms):
            cam_ids = all_cam_ids
        else:
            wanted = set(det.where_rooms)
            cam_ids = {c.id for c in cameras if c.room in wanted}
        desired.update((cid, det.id) for cid in cam_ids)

    if not managed_det_ids:
        return (0, 0)

    current = {
        (r.camera_id, r.detector_id)
        for r in (
            await session.execute(
                select(CameraDetector.camera_id, CameraDetector.detector_id).where(
                    CameraDetector.detector_id.in_(managed_det_ids)
                )
            )
        ).all()
    }

    stale = current - desired
    missing = desired - current

    deleted = 0
    if stale:
        result = await session.execute(
            delete(CameraDetector).where(
                tuple_(CameraDetector.camera_id, CameraDetector.detector_id).in_(
                    list(stale)
                )
            )
        )
        deleted = result.rowcount or 0

    inserted = 0
    if missing:
        now = datetime.now(UTC)
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        result = await session.execute(
            pg_insert(CameraDetector)
            .values(
                [{"camera_id": c, "detector_id": d, "created_at": now}
                 for c, d in missing]
            )
            .on_conflict_do_nothing(index_elements=["camera_id", "detector_id"])
        )
        inserted = result.rowcount or 0

    await session.flush()
    return (inserted, deleted)


