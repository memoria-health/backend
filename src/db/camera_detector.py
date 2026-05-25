"""ORM model for the `cameras_detectors` join table.

Many-to-many between cameras and detectors. A row means "this detector
runs on this camera." See `supabase/schema.sql` §cameras_detectors for
the full semantics (master switch on detectors.is_on, always_on bypass
for people_present, etc.).
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class CameraDetector(Base):
    __tablename__ = "cameras_detectors"

    camera_id: Mapped[UUID] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True,
    )
    detector_id: Mapped[UUID] = mapped_column(
        ForeignKey("detectors.id", ondelete="CASCADE"), primary_key=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
