"""Detector registry — load enabled detectors for a camera.

The dashboard caches detectors in-memory and hydrates from the DB on
boot + via Supabase Realtime; for nemara-backend we keep it simple
and read the detectors table per ingest call (small N per patient,
~5 rows). If profiling later shows this is hot, add an in-process
cache here.
"""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.detector import Detector
from src.services.detection.prompt import DetectorSpec, detector_spec_from_row


async def get_enabled_detectors_for_camera(
    session: AsyncSession, camera_id: UUID,
) -> Sequence[DetectorSpec]:
    """Return the runtime detector specs for `camera_id`.

    Joins through cameras → patients → detectors and filters by
    `is_on=true`. `people_present` is implicit (merged in by
    `prompt.build_prompt_and_schema`) so it's not in the returned list.
    """
    # Find the patient that owns this camera.
    cam = await session.get(Camera, camera_id)
    if cam is None:
        raise ValueError(
            f"[registry] camera {camera_id} does not exist; cannot resolve detectors"
        )

    # All enabled detectors for that patient.
    stmt = (
        select(Detector)
        .where(Detector.patient_id == cam.patient_id)
        .where(Detector.is_on.is_(True))
        .where(Detector.identifier.is_not(None))  # only runtime-ready rows
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [detector_spec_from_row(r) for r in rows]
