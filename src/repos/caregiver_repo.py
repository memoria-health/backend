"""Repository for the caregiver aggregate.

Pure SQLAlchemy queries. No business logic — that lives in services/.
Caller passes an explicit AsyncSession; transaction boundaries live
in the caller too.
"""

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver


async def find_by_id(session: AsyncSession, caregiver_id: UUID) -> Caregiver | None:
    """Return the caregiver row for `caregiver_id`, or None if not found."""
    result = await session.execute(select(Caregiver).where(Caregiver.id == caregiver_id))
    return result.scalar_one_or_none()


async def insert(session: AsyncSession, caregiver: Caregiver) -> Caregiver:
    """Insert a new caregiver row.

    Caller owns FK validity (auth.users(id) must exist) and uniqueness
    (email must not collide). Constraint violations propagate as
    `sqlalchemy.exc.IntegrityError`.
    """
    session.add(caregiver)
    await session.flush()
    return caregiver


async def update_partial(
    session: AsyncSession,
    caregiver_id: UUID,
    *,
    fields: Mapping[str, Any],
) -> Caregiver | None:
    """Apply a partial update to `caregiver_id`. Empty `fields` is a
    no-op (returns the row unchanged or None if missing). Returns the
    updated row, or None if no row matched.
    """
    if fields:
        matched = await session.execute(
            update(Caregiver)
            .where(Caregiver.id == caregiver_id)
            .values(**fields)
            .returning(Caregiver.id)
        )
        if matched.scalar_one_or_none() is None:
            return None
        await session.flush()
    return await find_by_id(session, caregiver_id)
