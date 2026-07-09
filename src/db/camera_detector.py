"""ORM model for the `cameras_detectors` join table.

Many-to-many between cameras and detectors. A row means "this detector
runs on this camera." See `supabase/schema.sql` §cameras_detectors for
the full semantics (master switch on detectors.is_on, always_on bypass
for people_present, etc.).
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
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

    # ── Per-sensor wandering exit-door config ───────────────────────
    # Only meaningful for the 'wandering' detector. The exit-door choice is
    # PER CAMERA (each sensor sees a different scene), so it lives on the
    # assignment, not the per-patient detector row. Shape:
    #   { doors: [{box:[x,y,w,h], label, source}], no_exit_door: bool }
    # The runtime (get_enabled_detectors_for_camera) builds THIS camera's
    # wandering prompt from these doors. no_exit_door=true => wandering is
    # skipped on this camera (nowhere to wander to here).
    door_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
