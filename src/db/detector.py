"""ORM model for the `detectors` table.

Maps 1:1 with `supabase/schema.sql` §detectors. Enums on the wire are
Python `str` here — boundary validation runs at the Pydantic schema
layer; the DB-level enum types catch anything that gets past.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Detector(Base):
    __tablename__ = "detectors"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    icon_name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(String(500))
    category: Mapped[str] = mapped_column(String(20))      # detector_category
    source: Mapped[str] = mapped_column(String(20))        # detector_source
    shape: Mapped[str] = mapped_column(String(20))         # detector_shape
    where_scope: Mapped[str] = mapped_column(String(80))
    when_scope: Mapped[str] = mapped_column(String(80))
    sensitivity: Mapped[str] = mapped_column(String(20))   # detector_sensitivity
    is_on: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    custom_prompt: Mapped[str | None] = mapped_column(String(2000), default=None)
