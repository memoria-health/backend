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

    # Per-account VLM override: "gemini" | "openai" | "dashscope".
    # NULL ⇒ use the global `settings.default_vlm_provider`. Lets us route
    # individual families onto a different model (e.g. Qwen3-VL via
    # OpenRouter) by flipping this one column — no deploy. Read (cached) by
    # services/detection/vlm_routing.py on the detection hot path.
    vlm_provider: Mapped[str | None] = mapped_column(String(20), default=None)

    # Per-account toggle for the dashboard's "Live analyses" feed. Defaults
    # TRUE (everyone sees it) — flip to false to hide it for a specific
    # account (e.g. demo logins) without a deploy. The web reads this from
    # /v1/me and gates the section on it (AND a global kill-switch).
    show_live_analyses: Mapped[bool] = mapped_column(
        Boolean, server_default=text("true"), default=True,
    )

    # Terms of Service + Privacy Notice acceptance. `tos_accepted_at` is the
    # server-stamped moment the caregiver accepted (NULL = never accepted →
    # the signup gate blocks the app until they do). `tos_version` is the
    # document version they accepted, so a later material update can bump the
    # version and re-prompt. Mirrors the patient data-collection consent
    # (data_collection_decided_at); accepted_at is never client-writable —
    # the API stamps it when tos_version is sent.
    tos_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
    tos_version: Mapped[str | None] = mapped_column(String(40), default=None)

    # Data-residency. `country` is the ISO alpha-2 the caregiver picked at
    # signup (recorded verbatim); `region` is DERIVED from it (US/IN) and is the
    # authoritative "India key" every downstream sensitive row resolves through
    # (region_of). Both server_default 'US' so pre-country rows stay valid; set
    # once at provisioning and immutable from the client path (DB trigger).
    country: Mapped[str] = mapped_column(
        String, server_default=text("'US'"), default="US",
    )
    region: Mapped[str] = mapped_column(
        String, server_default=text("'US'"), default="US",
    )
