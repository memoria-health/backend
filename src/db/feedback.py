"""ORM model for the `feedback` table.

Rows are written by the caregiver iOS app's Settings → Send feedback
sheet (POST /v1/feedback). Storing the row is the source of truth; an
email notification to the founders is fired best-effort afterward and
its failure never affects the persisted row.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, SmallInteger, String, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[UUID] = mapped_column(
        primary_key=True, server_default=text("gen_random_uuid()"),
    )
    caregiver_id: Mapped[UUID] = mapped_column(
        ForeignKey("caregivers.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"),
    )

    # 1-5 star rating. Optional: a caregiver may leave only free-text.
    rating: Mapped[int | None] = mapped_column(SmallInteger, default=None)
    whats_working: Mapped[str | None] = mapped_column(String, default=None)
    whats_frustrating: Mapped[str | None] = mapped_column(String, default=None)
    category: Mapped[str | None] = mapped_column(String, default=None)
