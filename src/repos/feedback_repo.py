"""Repository for the feedback aggregate.

Pure SQLAlchemy. No business logic — that lives in the API/service
layer. Caller passes an explicit AsyncSession and owns the transaction
boundary (the /v1/feedback handler commits after `create`).
"""

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.feedback import Feedback


async def create(
    session: AsyncSession,
    *,
    caregiver_id: UUID,
    rating: int | None,
    whats_working: str | None,
    whats_frustrating: str | None,
    category: str | None,
) -> Feedback:
    """Insert one feedback row and return it. Caller owns commit.

    `id` and `created_at` are assigned here in Python (mirrors
    event_repo / face_embedding_repo) — the ORM Base is
    `MappedAsDataclass`, which makes any column without a Python-side
    default a required constructor arg. The DB-side `server_default`s on
    those columns remain as a safety net for non-ORM inserts.

    `caregiver_id` must reference an existing `caregivers` row;
    constraint violations propagate as `sqlalchemy.exc.IntegrityError`.
    """
    row = Feedback(
        id=uuid4(),
        caregiver_id=caregiver_id,
        created_at=datetime.now(timezone.utc),
        rating=rating,
        whats_working=whats_working,
        whats_frustrating=whats_frustrating,
        category=category,
    )
    session.add(row)
    await session.flush()
    return row
