"""ORM model for the `patients` table.

Maps 1:1 with `supabase/schema.sql` §patients. Postgres enums
(`relation_kind`, `patient_state`) are mapped to Python `str` here —
boundary validation runs at the Pydantic schema layer; the DB-level
check constraints catch anything that gets past Pydantic.
"""

from datetime import date, datetime
from uuid import UUID

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.enums import PatientStateEnum, RelationKindEnum


class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    caregiver_id: Mapped[UUID] = mapped_column(ForeignKey("caregivers.id", ondelete="CASCADE"))
    full_name: Mapped[str] = mapped_column(String(200))
    preferred_name: Mapped[str] = mapped_column(String(80))
    relation: Mapped[str] = mapped_column(RelationKindEnum)
    state: Mapped[str] = mapped_column(PatientStateEnum)
    status_line: Mapped[str] = mapped_column(String(200))
    cameras_paused: Mapped[bool] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    date_of_birth: Mapped[date | None] = mapped_column(Date, default=None)
    gender: Mapped[str | None] = mapped_column(String(40), default=None)
    home_address: Mapped[str | None] = mapped_column(String(500), default=None)
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    battery_percent: Mapped[int | None] = mapped_column(Integer, default=None)
    photo_storage_key: Mapped[str | None] = mapped_column(String, default=None)

    # Per-patient email alerts. Each patient can route alerts to a
    # different set of family/aides. `email_alerts_enabled` is the
    # master switch; when on, alerts go to the owning caregiver's
    # account email PLUS every address in `email_alert_recipients`.
    email_alerts_enabled: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )
    email_alert_recipients: Mapped[list[str]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb"), default=list,
    )
