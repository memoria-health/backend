"""Patient-access helpers (membership-based authorization).

The single place the app answers "which patients can this caregiver touch?"
Repos scope patient-scoped queries with `accessible_patient_ids(...)` instead of
the old `Patient.caregiver_id == caregiver_id`. In the MVP every membership row
grants full access; `role` is used only for owner-guarded actions (share/unshare,
delete) and, later, per-patient permission levels.
"""

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.patient_member import PatientMember

OWNER_ROLE = "owner"


def accessible_patient_ids(caregiver_id: UUID):
    """Scalar subquery of patient ids `caregiver_id` is a member of.

    Use as `Patient.id.in_(accessible_patient_ids(cid))` or, on a child table,
    `Model.patient_id.in_(accessible_patient_ids(cid))`.

    READ/WRITE scope, not a destructive-action scope — every member matches,
    including someone the owner shared with. For anything irreversible use
    `owned_patient_ids` instead.
    """
    return select(PatientMember.patient_id).where(
        PatientMember.caregiver_id == caregiver_id
    )


def owned_patient_ids(caregiver_id: UUID):
    """Scalar subquery of patient ids `caregiver_id` OWNS.

    The creator gets `role='owner'` (patients.py create); everyone added later
    gets `role='caregiver'` (patient_members.py). Destructive, irreversible
    actions must scope with this rather than `accessible_patient_ids`, or any
    invited member can destroy the patient and cascade away every camera,
    detector and event belonging to someone else's family.
    """
    return select(PatientMember.patient_id).where(
        PatientMember.caregiver_id == caregiver_id,
        PatientMember.role == OWNER_ROLE,
    )


async def add_member(
    session: AsyncSession,
    *,
    patient_id: UUID,
    caregiver_id: UUID,
    role: str = "caregiver",
    invited_by: UUID | None = None,
) -> None:
    """Insert a membership row (idempotent on (patient_id, caregiver_id))."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    stmt = (
        pg_insert(PatientMember)
        .values(
            patient_id=patient_id,
            caregiver_id=caregiver_id,
            role=role,
            invited_by=invited_by,
        )
        .on_conflict_do_nothing(index_elements=["patient_id", "caregiver_id"])
    )
    await session.execute(stmt)


async def role_of(
    session: AsyncSession, *, patient_id: UUID, caregiver_id: UUID
) -> str | None:
    """The caregiver's role for the patient, or None if not a member."""
    result = await session.execute(
        select(PatientMember.role).where(
            PatientMember.patient_id == patient_id,
            PatientMember.caregiver_id == caregiver_id,
        )
    )
    return result.scalar_one_or_none()


async def is_member(
    session: AsyncSession, *, patient_id: UUID, caregiver_id: UUID
) -> bool:
    return (await role_of(session, patient_id=patient_id, caregiver_id=caregiver_id)) is not None


async def is_owner(
    session: AsyncSession, *, patient_id: UUID, caregiver_id: UUID
) -> bool:
    return (await role_of(session, patient_id=patient_id, caregiver_id=caregiver_id)) == "owner"


async def list_members(
    session: AsyncSession, *, patient_id: UUID
) -> Sequence[PatientMember]:
    result = await session.execute(
        select(PatientMember)
        .where(PatientMember.patient_id == patient_id)
        .order_by(PatientMember.created_at.asc())
    )
    return result.scalars().all()


async def owner_count(session: AsyncSession, *, patient_id: UUID) -> int:
    result = await session.execute(
        select(PatientMember).where(
            PatientMember.patient_id == patient_id,
            PatientMember.role == "owner",
        )
    )
    return len(result.scalars().all())


async def remove_member(
    session: AsyncSession, *, patient_id: UUID, caregiver_id: UUID
) -> None:
    await session.execute(
        delete(PatientMember).where(
            PatientMember.patient_id == patient_id,
            PatientMember.caregiver_id == caregiver_id,
        )
    )
