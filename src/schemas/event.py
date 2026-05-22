"""Pydantic boundary models for the event API surface.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/DetectionEvent.swift`.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.detector import DetectorCategory, DetectorSource


class ReviewChoice(str, Enum):
    correct = "correct"
    wrong = "wrong"
    unsure = "unsure"


class EventFilter(str, Enum):
    all = "all"
    unreviewed = "unreviewed"
    reviewed = "reviewed"


Title = Annotated[str, Field(min_length=1, max_length=200)]
Location = Annotated[str, Field(min_length=1, max_length=120)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]
Narrative = Annotated[str, Field(max_length=4000)]
Notes = Annotated[str, Field(max_length=2000)]


class EventResponse(BaseModel):
    """Full event row."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    patient_id: UUID
    detector_id: UUID | None
    camera_id: UUID | None
    category: DetectorCategory
    source: DetectorSource
    icon_name: IconName
    title: Title
    location: Location
    occurred_at: datetime
    has_clip: bool
    narrative: Narrative
    opened: bool
    reviewed: bool
    review_choice: ReviewChoice | None
    review_notes: Notes | None
    reviewed_at: datetime | None
    created_at: datetime


class EventListResponse(BaseModel):
    """Paged event list. Cursor is opaque to the client — pass it back
    verbatim on the next call. Empty `next_cursor` ⇒ end of stream."""

    model_config = ConfigDict(frozen=True)

    items: list[EventResponse]
    next_cursor: str | None


class UpdateEventRequest(BaseModel):
    """Body of PATCH /v1/events/{id}.

    Server stamps `reviewed_at = now()` when `reviewed` flips from
    `false` to `true`. All fields optional; `extra="forbid"` rejects
    unknown keys.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    opened: bool | None = None
    reviewed: bool | None = None
    review_choice: ReviewChoice | None = None
    review_notes: Notes | None = None
