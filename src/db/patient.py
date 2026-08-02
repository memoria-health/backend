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

    # Data-residency region for this patient's data: 'US' | 'IN'. The
    # authoritative residency key — storage routing (Backblaze bucket,
    # Supabase key prefix) and lift/breach/DSAR scoping key off it. Set at
    # patient creation; immutable thereafter (DB trigger blocks client-path
    # changes). server_default keeps pre-region rows/inserts valid.
    region: Mapped[str] = mapped_column(
        String, server_default=text("'US'"), default="US",
    )
    last_event_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    photo_storage_key: Mapped[str | None] = mapped_column(String, default=None)

    # Per-patient email alerts. Each patient can route alerts to a
    # different set of family/aides. `email_alerts_enabled` is the
    # master switch; when on, alerts go to the owning caregiver's
    # account email PLUS every address in `email_alert_recipients`.
    email_alerts_enabled: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )
    # `default_factory` (not `default`) is the dataclass-style invokable
    # default. `default=list` stores the type itself, which the JSONB
    # bind processor then tries to JSON-encode and fails with
    # "Object of type type is not JSON serializable". analysis.py
    # has the same trap noted in a comment; same fix needed here.
    email_alert_recipients: Mapped[list[str]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb"), default_factory=list,
    )

    # ── Data-collection consent (research opt-in) ────────────────
    # Per-patient switch to store short video+audio clips of this
    # patient for research (only while someone is present).
    # `data_collection_enabled` is the master on/off — default OFF, so
    # nothing records until the caregiver explicitly opts in.
    # `data_collection_decided_at` records WHEN a decision was made:
    # NULL = never asked, which drives the first-open consent banner on
    # web + iOS. Both "Share" and "Not now" stamp it, so once a decision
    # exists the banner never re-nags.
    data_collection_enabled: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )
    data_collection_decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None,
    )

    # ── Privacy mode (dignity override) ──────────────────────────
    # Per-patient kill switch. While ON it OVERRIDES data-collection
    # (no clips stored, in-flight sessions aborted without upload),
    # severs all live video+audio to caregivers at the SFU, and strips
    # event clips from insights — while detection + TEXT insights keep
    # flowing. A live switch, not a consent record, so no decided_at:
    # privacy SUSPENDS `data_collection_enabled` without mutating it,
    # so the stored consent decision is restored verbatim when privacy
    # clears. Enforced server-side; the UI flag is cosmetic.
    privacy_mode_enabled: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )

    # review-gate: hold alerts for human YES/NO before dispatch (additive)
    require_review: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )
