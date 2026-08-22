"""Repository for the patient aggregate.

All reads are scoped by `caregiver_id` to prevent cross-account leaks
at the application layer (RLS is the DB-level belt-and-braces). Each
function takes an explicit AsyncSession; transaction boundaries live
in the caller (the service layer).
"""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.patient import Patient
from src.repos.access import accessible_patient_ids, owned_patient_ids


async def list_for_caregiver(
    session: AsyncSession, caregiver_id: UUID
) -> Sequence[Patient]:
    """Return every patient owned by `caregiver_id`, newest first."""
    result = await session.execute(
        select(Patient)
        .where(Patient.id.in_(accessible_patient_ids(caregiver_id)))
        .order_by(Patient.created_at.desc())
    )
    return result.scalars().all()


async def find_by_id(
    session: AsyncSession, patient_id: UUID, *, caregiver_id: UUID
) -> Patient | None:
    """Return the patient if it exists AND is owned by `caregiver_id`.

    The caregiver-scoped lookup is what makes the API return 404 (vs.
    403) for cross-account access — we don't tell a probing client
    whether the row exists at all.
    """
    result = await session.execute(
        select(Patient).where(
            Patient.id == patient_id,
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
        )
    )
    return result.scalar_one_or_none()


async def insert(session: AsyncSession, patient: Patient) -> Patient:
    """Insert a new patient row. Caller owns FK validity + transaction."""
    session.add(patient)
    await session.flush()
    return patient


async def update_partial(
    session: AsyncSession,
    patient_id: UUID,
    *,
    caregiver_id: UUID,
    fields: Mapping[str, Any],
) -> Patient | None:
    """Apply a partial update to a caregiver-owned patient row.

    Returns the updated row, or None if no caregiver-owned row matched.
    Empty `fields` is a no-op (returns the row unchanged or None if
    missing).
    """
    if fields:
        matched = await session.execute(
            update(Patient)
            .where(
                Patient.id == patient_id,
                Patient.id.in_(accessible_patient_ids(caregiver_id)),
            )
            .values(**fields)
            .returning(Patient.id)
        )
        if matched.scalar_one_or_none() is None:
            return None
        await session.flush()
    return await find_by_id(session, patient_id, caregiver_id=caregiver_id)


async def delete_by_id(
    session: AsyncSession, patient_id: UUID, *, caregiver_id: UUID
) -> bool:
    """Delete a patient the caregiver OWNS. Returns True if a row was
    deleted, False if no owned row matched.

    Cascades to detectors / events via DB-level `ON DELETE CASCADE`.
    SENSORS survive: `cameras.patient_id` is `ON DELETE SET NULL` since
    2026-08-22, because a sensor can watch more than one person and
    deleting one of them must not take the hardware from the others. The
    caller re-points the shim from whatever coverage remains.

    Scoped with `owned_patient_ids`, not `accessible_patient_ids`
    (audit finding H3). Membership grants read/write; it must not grant
    the ability to destroy the record — a caregiver invited to help
    watch someone's parent could otherwise delete that patient and
    cascade away every camera, detector and event with it. A non-owner
    member now falls through to the same 404 as a stranger, which also
    avoids confirming the patient exists.
    """
    result = await session.execute(
        delete(Patient)
        .where(
            Patient.id == patient_id,
            Patient.id.in_(owned_patient_ids(caregiver_id)),
        )
        .returning(Patient.id)
    )
    return result.scalar_one_or_none() is not None
