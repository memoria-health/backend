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
    ended_at: datetime | None = None
    occurred_at: datetime
    has_clip: bool
    narrative: Narrative
    opened: bool
    reviewed: bool
    review_choice: ReviewChoice | None
    review_notes: Notes | None
    reviewed_at: datetime | None
    clip_storage_key: str | None
    # True when the clip was genuinely dropped (vs still processing):
    # since the censor pipeline (2026-07-27) that means the fail-closed
    # terminal — the clip needed censoring and censoring failed, so it
    # was discarded rather than uploaded raw. (Historical rows: legacy
    # privacy-mode suppression.) Drives the "no clip" placeholder.
    clip_suppressed: bool
    # True when the archived clip was CENSORED (person-mask blur + audio
    # stripped) after the sensitivity screen flagged a dignity-sensitive
    # moment. Drives the "Sensitive moment — censored for dignity"
    # treatment in event detail. Written by clip_archival at upload time
    # (services/detection/sensitivity.py + censor.py, 2026-07-27).
    clip_censored: bool = False
    # Which sensitivity categories the screen flagged (enum strings, e.g.
    # "bathing_showering"). Null for non-sensitive clips and for clips
    # censored without a trusted verdict (screen failure → full-frame).
    sensitivity_categories: list[str] | None = None
    created_at: datetime


class ClipURLResponse(BaseModel):
    """Returned from `GET /v1/events/{id}/clip-url`. Signed URL is
    short-lived (default 10 min via storage_signed_url_default_ttl_seconds);
    the caregiver app fetches a fresh one on every player open."""

    model_config = ConfigDict(frozen=True)

    url: str
    expires_in_seconds: int


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
