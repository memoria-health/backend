"""ORM model for the `cameras` table.

Rows are written by the wall-phone pairing flow (a later plan). For
v0, populated by the seed script; the caregiver app can list + delete
them but does not POST.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    display_id: Mapped[int] = mapped_column(Integer)
    room: Mapped[str] = mapped_column(String(80))
    icon_name: Mapped[str] = mapped_column(String(80))
    online: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
