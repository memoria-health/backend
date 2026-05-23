"""Repository for the caregiver_device aggregate.

Pure SQLAlchemy queries. No business logic — that lives in services/.
Caller passes an explicit AsyncSession; transaction boundaries are
the caller's responsibility.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver_device import CaregiverDevice


async def upsert_by_token(
    session: AsyncSession,
    *,
    caregiver_id: UUID,
    apns_token: str,
    bundle_id: str,
    environment: str,
) -> CaregiverDevice:
    """Insert or update the row for `apns_token`.

    The `apns_token` column is uniquely indexed; on conflict we update
    `caregiver_id`, `bundle_id`, `environment`, and `updated_at` so a
    token that moves between caregivers (rare, but possible after a
    sign-out + sign-in on the same device) ends up owned by the
    current caller.
    """
    now = datetime.now(timezone.utc)
    stmt = (
        insert(CaregiverDevice)
        .values(
            id=uuid4(),
            caregiver_id=caregiver_id,
            apns_token=apns_token,
            bundle_id=bundle_id,
            environment=environment,
            created_at=now,
            updated_at=now,
        )
        .on_conflict_do_update(
            index_elements=[CaregiverDevice.apns_token],
            set_={
                "caregiver_id": caregiver_id,
                "bundle_id": bundle_id,
                "environment": environment,
                "updated_at": now,
            },
        )
        .returning(CaregiverDevice)
    )
    result = await session.execute(stmt)
    return result.scalar_one()


async def find_by_caregiver(
    session: AsyncSession,
    caregiver_id: UUID,
) -> Sequence[CaregiverDevice]:
    """Return every device registered for `caregiver_id`. Ordered by
    `created_at` so the oldest registration leads — irrelevant to the
    push dispatcher (it fans out to every device) but deterministic
    in tests.
    """
    stmt = (
        select(CaregiverDevice)
        .where(CaregiverDevice.caregiver_id == caregiver_id)
        .order_by(CaregiverDevice.created_at)
    )
    return (await session.execute(stmt)).scalars().all()
