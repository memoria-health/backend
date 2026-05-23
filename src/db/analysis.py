"""ORM model for the `analyses` table.

One row per detector-pipeline tick (~10s per camera). Inserted by
`CameraIngest._run_tick` in `src/services/detection/camera_ingest.py`
as the subscriber-bot consumes LiveKit tracks.

Not exposed on the wire to the caregiver app — `events` is the
user-facing surface. Analyses are for debugging (Supabase Studio) and
for the detector worker itself (active-camera resolution for Talk
routing).
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    caregiver_id: Mapped[UUID] = mapped_column(ForeignKey("caregivers.id", ondelete="CASCADE"))
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    camera_id: Mapped[UUID] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))

    provider: Mapped[str] = mapped_column(String(16))
    description: Mapped[str] = mapped_column(String)
    # No `default=` on JSONB — SQLAlchemy treats explicit defaults as
    # dataclass-style and then complains when subsequent columns are
    # non-default. The caller always passes detector_fields explicitly.
    detector_fields: Mapped[dict[str, Any]] = mapped_column(JSONB)
    people_present: Mapped[bool] = mapped_column(Boolean)
    motion_in_clip: Mapped[bool] = mapped_column(Boolean)
    state_after: Mapped[str] = mapped_column(String(8))
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
