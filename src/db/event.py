"""ORM model for the `events` table.

Rows are written by wall phones (a later plan). The caregiver app only
reads + PATCHes them in v0.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import ARRAY, Boolean, DateTime, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base
from src.db.enums import DetectorCategoryEnum, DetectorSourceEnum, ReviewChoiceEnum


class Event(Base):
    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(DetectorCategoryEnum)
    source: Mapped[str] = mapped_column(DetectorSourceEnum)
    icon_name: Mapped[str] = mapped_column(String(80))
    title: Mapped[str] = mapped_column(String(200))
    location: Mapped[str] = mapped_column(String(120))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    has_clip: Mapped[bool] = mapped_column(Boolean)
    narrative: Mapped[str] = mapped_column(String(4000))
    opened: Mapped[bool] = mapped_column(Boolean)
    reviewed: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    detector_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("detectors.id", ondelete="SET NULL"), default=None
    )
    camera_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"), default=None
    )
    review_choice: Mapped[str | None] = mapped_column(ReviewChoiceEnum, default=None)
    review_notes: Mapped[str | None] = mapped_column(String(2000), default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    # OPERATOR gate state — null | pending | approved | rejected | auto.
    # Nothing to do with `reviewed` three lines up, which is the CAREGIVER
    # looking back at an alert they already received. This one decides
    # whether they receive it at all. See src/services/event_gate.py.
    # Null for every patient with the gate off, which is all of them.
    gate_state: Mapped[str | None] = mapped_column(String, default=None)
    # When the hold started. The auto-release deadline is measured from
    # this, not from created_at, and GateSweeper needs it to survive a
    # restart. Set in the same INSERT as gate_state so a held event is
    # never published without a deadline.
    gate_held_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    # Bucket-relative key in `incident-clips`; NULL until clip archival
    # completes (post-event window captured + ffmpeg concat uploaded).
    clip_storage_key: Mapped[str | None] = mapped_column(String, default=None)
    # True when this event's clip was intentionally DROPPED, so the apps
    # can show a "no clip" placeholder instead of an endless "Generating
    # clip…" for `has_clip=false`. Since the censor pipeline (2026-07-27)
    # the only writer is clip_archival's fail-closed terminal: the clip
    # needed censoring and censoring failed, so it was discarded rather
    # than uploaded raw. (Historical rows: legacy privacy-mode
    # suppression.)
    # Sustained episodes: last positive tick seen (extends while the
    # episode runs; final value = the falling edge). Null for instants.
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None)
    clip_suppressed: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )
    # True when the archived clip was CENSORED (person-mask blur + audio
    # stripped) because the sensitivity screen flagged a dignity-
    # sensitive moment — or because the screen failed and the clip was
    # blurred full-frame, fail-closed. Written by clip_archival at
    # upload time — see docs/plans/2026-07-27-room-privacy-and-clip-
    # censoring.md.
    clip_censored: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )
    # Which sensitivity categories the screen flagged (e.g.
    # ['bathing_showering', 'nudity_or_undress']). NULL for
    # non-sensitive clips and for clips censored without a trusted
    # verdict (screen failure → full-frame blur).
    sensitivity_categories: Mapped[list[str] | None] = mapped_column(
        ARRAY(String), default=None,
    )
