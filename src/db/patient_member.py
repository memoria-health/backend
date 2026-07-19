"""ORM model for `patient_members` — a caregiver's access to a patient.

Maps 1:1 with `supabase/schema.sql` §patient_members. A row grants a caregiver
access to a patient; `role` is present for future per-patient permission levels
but is UNENFORCED in the MVP (every member has full access). The original owner
is mirrored as a role='owner' row.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class PatientMember(Base):
    __tablename__ = "patient_members"

    # Server-generated → excluded from the dataclass constructor (init=False).
    id: Mapped[UUID] = mapped_column(
        primary_key=True, server_default=text("gen_random_uuid()"), init=False,
    )
    patient_id: Mapped[UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
    )
    caregiver_id: Mapped[UUID] = mapped_column(
        ForeignKey("caregivers.id", ondelete="CASCADE"),
    )
    role: Mapped[str] = mapped_column(
        String(20), server_default=text("'caregiver'"), default="caregiver",
    )
    invited_by: Mapped[UUID | None] = mapped_column(
        ForeignKey("caregivers.id", ondelete="SET NULL"), default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), init=False,
    )
