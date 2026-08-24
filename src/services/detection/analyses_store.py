"""INSERT into the `analyses` table.

Ports `nemara-temp-dashboard/lib/analyses-store.ts:addAnalysis` but
adapted to the nemara-backend schema (where `caregiver_id` denormalizes
the join through patients → cameras so RLS can scope by
`caregiver_id = auth.uid()` without an extra subselect on every read).
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.patient import Patient


async def add_analysis(
    session: AsyncSession,
    *,
    camera_id: UUID,
    provider: str,
    description: str,
    detector_fields: dict[str, Any],
    people_present: bool,
    people_count: int,
    motion_in_clip: bool,
    state_after: str,
    captured_at: datetime,
) -> Analysis:
    """Persist one analysis tick. Looks up caregiver_id + patient_id
    via the camera row. Includes `people_count` inside `detector_fields`
    (matches dashboard's behavior — keeps the row's column count
    tight). Caller owns the transaction.
    """
    cam = await session.get(Camera, camera_id)
    if cam is None:
        raise ValueError(f"[analyses_store] camera {camera_id} does not exist")
    patient = await session.get(Patient, cam.patient_id)
    if patient is None:
        raise ValueError(
            f"[analyses_store] patient {cam.patient_id} for camera "
            f"{camera_id} does not exist"
        )

    # Merge people_count into detector_fields so we don't drop it on
    # the floor (no dedicated column on analyses, same as dashboard).
    merged_fields = {**detector_fields, "people_count": people_count}

    row = Analysis(
        id=uuid4(),
        caregiver_id=patient.caregiver_id,
        # `cameras.patient_id`, so the earliest-added covered patient and
        # nobody else. A tick is a fact about a SENSOR, not about a
        # person, and this column cannot say otherwise on a shared
        # sensor. It stays for RLS scoping beside `caregiver_id`; NOTHING
        # filters a person's observations on it any more — those resolve
        # through `camera_patients` (see `camera_coverage.covered_cameras`),
        # which is the only answer that includes the second resident.
        patient_id=patient.id,
        camera_id=camera_id,
        provider=provider,
        description=description,
        detector_fields=merged_fields,
        people_present=people_present,
        motion_in_clip=motion_in_clip,
        state_after=state_after,
        captured_at=captured_at,
        created_at=datetime.now(timezone.utc),
    )
    session.add(row)
    await session.flush()
    return row
