"""ORM model for the `detectors` table.

Maps 1:1 with `supabase/schema.sql` §detectors. Enums on the wire are
Python `str` here — boundary validation runs at the Pydantic schema
layer; the DB-level enum types catch anything that gets past.
"""

from datetime import date, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import Boolean, Date, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.enums import (
    DetectorCategoryEnum,
    DetectorSensitivityEnum,
    DetectorShapeEnum,
    DetectorSourceEnum,
)


class Detector(Base):
    __tablename__ = "detectors"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(120))
    icon_name: Mapped[str] = mapped_column(String(80))
    description: Mapped[str] = mapped_column(String(500))
    category: Mapped[str] = mapped_column(DetectorCategoryEnum)
    source: Mapped[str] = mapped_column(DetectorSourceEnum)
    shape: Mapped[str] = mapped_column(DetectorShapeEnum)
    where_scope: Mapped[str] = mapped_column(String(80))
    when_scope: Mapped[str] = mapped_column(String(80))
    sensitivity: Mapped[str] = mapped_column(DetectorSensitivityEnum)
    is_on: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    custom_prompt: Mapped[str | None] = mapped_column(String(2000), default=None)

    # ── Runtime contract (detector pipeline) ────────────────────────
    # See `src/detectors/registry.py` for how these flow into LLM calls.
    # `identifier` is the stable key in the JSON-schema (e.g. "fall");
    # `prompt_fragment` is the per-detector instructions appended to the
    # base prompt; `schema_json` is `{field_name: {type, description?}}`.
    identifier: Mapped[str | None] = mapped_column(String(80), default=None)
    prompt_fragment: Mapped[str | None] = mapped_column(String, default=None)
    schema_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)
    always_on: Mapped[bool] = mapped_column(Boolean, default=False)
    is_builtin: Mapped[bool] = mapped_column(Boolean, default=True)

    # ── Scheduled ("absence") detectors ─────────────────────────────
    # Null for reactive (point_in_time / state) detectors. When shape
    # == 'scheduled', holds the window the runtime scheduler evaluates:
    #   { cadence: 'daily'|'weekly', days_of_week: [0..6],
    #     window_start: 'HH:MM', window_end: 'HH:MM', deadline: 'HH:MM' }
    # The per-tick detection (identifier/prompt_fragment/schema_json) is
    # unchanged — the schedule just inverts + windows when it fires.
    schedule_config: Mapped[dict[str, Any] | None] = mapped_column(JSONB, default=None)

    # Single-fire guard for the scheduler — the last date this scheduled
    # detector was evaluated (whether or not it fired). Internal; not
    # exposed in DetectorResponse. NULL = never evaluated.
    last_scheduled_eval_date: Mapped[date | None] = mapped_column(Date, default=None)
