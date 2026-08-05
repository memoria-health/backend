"""ORM model for the `episodes` table — the annotation layer.

An episode is something that HAPPENED OVER A SPAN: a fall detector
arming, firing and resolving; a night's sleep from 22:14 to 06:03. It is
the middle of three layers:

  1. `analyses` — one row per ~10s tick. Dense, immutable, machine-made.
  2. `episodes` — intervals with state. Mutable, revisable. This file.
  3. `events`   — the alerts that actually fired.

Two properties make this layer necessary rather than a view over
`analyses`:

**It outlives a clip.** A sleep episode crosses ~30 recording rolls plus
gaps where nothing recorded at all. Nothing attached to a single clip can
hold it, which is why episodes live on a per-camera WALL-CLOCK timeline
and the link to video is computed at read time
(`services/annotations/projection.py`) rather than stored.

**It is not known when it starts.** `ended_at` is NULL while an episode
is still running, and sleep is only confirmed hours after the fact.
Revision is therefore normal, not exceptional — and it is done by
superseding rather than overwriting, so a verdict a human already
attached to an episode can never be silently re-pointed at a different
span.

`payload` deliberately carries EVERYTHING the detector was thinking:
reasoning text, k-of-N counters, latch state, quality flags. Filtering
happens at export, where the audience is known. Deciding at write time
what a future consumer may see means losing it forever.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base

# An episode's lifecycle. `open` while it is still happening; the rest
# are terminal.
EPISODE_STATE_OPEN = "open"
EPISODE_STATE_CLOSED = "closed"
EPISODE_STATE_RETRACTED = "retracted"  # decided afterwards it never happened


class Episode(Base):
    __tablename__ = "episodes"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    caregiver_id: Mapped[UUID] = mapped_column(
        ForeignKey("caregivers.id", ondelete="CASCADE")
    )
    patient_id: Mapped[UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE")
    )
    camera_id: Mapped[UUID] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE")
    )

    # What kind of thing this is: a detector slug ("fall", "wandering") or
    # a derived one ("sleep"). Free text so a new detector needs no
    # migration.
    kind: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(16))

    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # NULL while the episode is still running. Sleep stays open all night.
    ended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    # See the module docstring: everything, filtered at export.
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)

    # Revision without destruction. When an episode is re-cut — split,
    # merged, re-dated — the old row keeps its span and points here.
    # Anything already hanging off it (a reviewer's true/false-positive
    # verdict) still refers to the span that was actually judged.
    superseded_by: Mapped[UUID | None] = mapped_column(
        ForeignKey("episodes.id", ondelete="SET NULL"), default=None
    )

    # Human adjudication, when it exists. NULL means nobody has looked —
    # which is the labelling backlog, and is queryable as exactly that.
    # True = the detector was right, False = false positive.
    reviewer_verdict: Mapped[bool | None] = mapped_column(default=None)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
