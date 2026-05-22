"""ORM model for the `caregivers` table.

Maps 1:1 with `supabase/schema.sql` §caregivers. Note that the row's
PK equals Supabase's `auth.users.id` (FK to `auth.users(id)` with
`ON DELETE CASCADE` enforced at the DB level).
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Caregiver(Base):
    __tablename__ = "caregivers"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    display_name: Mapped[str] = mapped_column(String(80))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    phone: Mapped[str | None] = mapped_column(String(32), default=None)
