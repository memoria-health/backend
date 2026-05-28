"""ORM model for the `caregivers` table.

Maps 1:1 with `supabase/schema.sql` §caregivers. Note that the row's
PK equals Supabase's `auth.users.id` (FK to `auth.users(id)` with
`ON DELETE CASCADE` enforced at the DB level).
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, String, text
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

    # Web onboarding walkthrough flag — false until the user dismisses
    # or completes the modal on the dashboard. Per-surface (iOS gets
    # its own column when that walkthrough ships) so learning one
    # surface doesn't suppress the walkthrough on the other.
    onboarded_web: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )

    # IANA timezone (e.g. "America/New_York"). Scheduled detectors
    # evaluate their windows in this zone. Auto-set from the web
    # client's browser tz; the default is a fallback.
    timezone: Mapped[str] = mapped_column(
        String(64), server_default=text("'America/New_York'"), default="America/New_York",
    )
