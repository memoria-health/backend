"""Repository for the detector aggregate.

Reads are scoped through `patients.caregiver_id` — the application
layer enforces ownership; RLS catches anything that gets past.
"""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.detector import Detector
from src.db.patient import Patient


async def list_for_patient(
    session: AsyncSession, patient_id: UUID, *, caregiver_id: UUID
) -> Sequence[Detector]:
    """Return every detector for a caregiver-owned patient.

    Joins through `patients` so a request for someone else's
    patient_id silently returns empty.
    """
    result = await session.execute(
        select(Detector)
        .join(Patient, Patient.id == Detector.patient_id)
        .where(
            Detector.patient_id == patient_id,
            Patient.caregiver_id == caregiver_id,
        )
        .order_by(Detector.created_at.asc())
    )
    return result.scalars().all()


async def find_by_id(
    session: AsyncSession, detector_id: UUID, *, caregiver_id: UUID
) -> Detector | None:
    """Return the detector if it exists AND the parent patient is owned
    by `caregiver_id`. Returns None otherwise (the API layer maps to 404).
    """
    result = await session.execute(
        select(Detector)
        .join(Patient, Patient.id == Detector.patient_id)
        .where(
            Detector.id == detector_id,
            Patient.caregiver_id == caregiver_id,
        )
    )
    return result.scalar_one_or_none()


async def insert(session: AsyncSession, detector: Detector) -> Detector:
    """Insert a new detector. Caller owns FK validity + transaction."""
    session.add(detector)
    await session.flush()
    return detector


async def update_partial(
    session: AsyncSession,
    detector_id: UUID,
    *,
    caregiver_id: UUID,
    fields: Mapping[str, Any],
) -> Detector | None:
    """Apply a partial update to a caregiver-owned detector. Returns
    the row, or None if not found / not owned.
    """
    # Verify ownership first (avoids a wasted UPDATE when not owned).
    existing = await find_by_id(session, detector_id, caregiver_id=caregiver_id)
    if existing is None:
        return None
    if fields:
        await session.execute(
            update(Detector).where(Detector.id == detector_id).values(**fields)
        )
        await session.flush()
        return await find_by_id(session, detector_id, caregiver_id=caregiver_id)
    return existing


async def delete_by_id(
    session: AsyncSession, detector_id: UUID, *, caregiver_id: UUID
) -> bool:
    """Delete a caregiver-owned detector. Returns True if deleted."""
    existing = await find_by_id(session, detector_id, caregiver_id=caregiver_id)
    if existing is None:
        return False
    await session.execute(delete(Detector).where(Detector.id == detector_id))
    await session.flush()
    return True
