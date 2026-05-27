"""ORM model for the `pi_discoveries` table.

One row per Pi device that has announced itself to the backend but
has not yet been paired to a camera row. Caregivers see unclaimed
discoveries in the "Add a camera" picker and select one to bind to
a patient + room.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class PiDiscovery(Base):
    __tablename__ = "pi_discoveries"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    hardware_id: Mapped[str] = mapped_column(String(64))
    device_label: Mapped[str] = mapped_column(String(120))
    current_url: Mapped[str] = mapped_column(String(2048))
    # All remaining columns have Python-side defaults so the
    # SQLAlchemy dataclass transform accepts the field ordering.
    # `first_seen_at` / `last_seen_at` rely on the DB
    # server_default in production; the Python-side `default=None`
    # exists only to satisfy the dataclass field-ordering rule.
    first_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), default=None,
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), default=None,
    )
    current_audio_url: Mapped[str | None] = mapped_column(
        String(2048), default=None,
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
    claimed_camera_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), default=None,
    )
