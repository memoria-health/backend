"""ORM model for the `events` table.

Rows are written by wall phones (a later plan). The caregiver app only
reads + PATCHes them in v0.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.enums import DetectorCategoryEnum, DetectorSourceEnum, ReviewChoiceEnum


class Event(Base):
    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(DetectorCategoryEnum)
    source: Mapped[str] = mapped_column(DetectorSourceEnum)
    icon_name: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(200))
    location: Mapped[str] = mapped_column(String(120))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    has_clip: Mapped[bool] = mapped_column(Boolean)
    narrative: Mapped[str] = mapped_column(String(4000))
    opened: Mapped[bool] = mapped_column(Boolean)
    reviewed: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    detector_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("detectors.id", ondelete="SET NULL"), default=None
    )
    camera_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), default=None
    )
    review_choice: Mapped[str | None] = mapped_column(ReviewChoiceEnum, default=None)
    review_notes: Mapped[str | None] = mapped_column(String(2000), default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
