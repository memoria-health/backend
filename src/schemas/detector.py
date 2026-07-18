"""Pydantic boundary models for the detector API surface.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Detector.swift`.
The four enum values match the SQL enums in `supabase/schema.sql`.
"""

from datetime import datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class DetectorCategory(str, Enum):
    safety = "safety"
    routine = "routine"


class DetectorSource(str, Enum):
    preset = "preset"
    custom = "custom"


class DetectorShape(str, Enum):
    point_in_time = "point_in_time"  # momentary — fires the moment it's seen
    state = "state"          # continuous/ongoing activity — one event per session
    scheduled = "scheduled"  # absence detector — fires if signal absent in window


class DetectorSensitivity(str, Enum):
    low = "low"
    standard = "standard"
    high = "high"


class ScheduleCadence(str, Enum):
    daily = "daily"
    weekly = "weekly"


class ScheduleConfig(BaseModel):
    """Window an `scheduled` detector evaluates. All fields optional so
    the LLM can return a partial inference (blanks the user fills in on
    the review screen). Times are local "HH:MM" 24h strings;
    days_of_week is 0=Mon..6=Sun (weekly only)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cadence: ScheduleCadence | None = None
    days_of_week: list[int] | None = None
    window_start: str | None = None
    window_end: str | None = None
    deadline: str | None = None


Name = Annotated[str, Field(min_length=1, max_length=120)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]
Description = Annotated[str, Field(min_length=1, max_length=500)]
Scope = Annotated[str, Field(min_length=1, max_length=80)]
Prompt = Annotated[str, Field(min_length=1, max_length=2000)]


class DetectorResponse(BaseModel):
    """Full detector row returned by list / create / update endpoints."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    patient_id: UUID
    name: Name
    icon_name: IconName
    description: Description
    category: DetectorCategory
    source: DetectorSource
    shape: DetectorShape
    where_scope: Scope
    when_scope: Scope
    sensitivity: DetectorSensitivity
    is_on: bool
    custom_prompt: Prompt | None
    # Runtime-contract identifier — `fall` / `wandering` / `agitation` /
    # `medication` for presets, `custom_<8hex>` for compiled custom
    # detectors. NULL on a freshly-inserted custom detector that's still
    # being compiled in the background (see create_detector + the
    # async-compile flow). The dashboard needs this to resolve
    # custom-detector reasoning fields in analyses.detector_fields
    # back to a human-readable name.
    identifier: str | None = None
    # Null for reactive detectors; the evaluation window for `scheduled`.
    schedule_config: ScheduleConfig | None = None
    # Null unless shape == 'state'; the rising-edge episode gap (seconds).
    episode_gap_seconds: int | None = None
    # Wandering only: the caregiver's saved exit-door choice, so the picker can
    # pre-highlight it. {box:[x,y,w,h]|null, label, source, no_exit_door}.
    door_config: dict | None = None
    created_at: datetime
    updated_at: datetime


class CreateDetectorRequest(BaseModel):
    """Body of POST /v1/patients/{id}/detectors.

    Server forces `source = custom`. `custom_prompt` is the caregiver's
    plain-English description (required). The other fields are normally
    LLM-inferred (via /infer) then confirmed on the review screen, but
    the client sends the final values here.

    `where_scope` / `when_scope` are legacy free-text scope hints no
    longer shown in the UI — they default server-side. `shape` defaults
    to point_in_time (reactive); the client sends `scheduled` +
    `schedule_config` for absence detectors.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Name
    custom_prompt: Prompt
    description: Description = "Custom detector"
    category: DetectorCategory = DetectorCategory.safety
    shape: DetectorShape = DetectorShape.point_in_time
    icon_name: IconName = "eye"
    where_scope: Scope = "ALL ROOMS"
    when_scope: Scope = "ALWAYS"
    sensitivity: DetectorSensitivity = DetectorSensitivity.standard
    schedule_config: ScheduleConfig | None = None
    # Set only for shape == 'state' (the rising-edge episode gap, seconds).
    episode_gap_seconds: int | None = None


class InferDetectorRequest(BaseModel):
    """Body of POST /v1/detectors/infer — the plain-English ask the LLM
    classifies + structures for the review screen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Name | None = None
    prompt: Prompt


class DetectorPlan(BaseModel):
    """What the compiler deduced from the caregiver's words, in plain
    English — surfaced on the review screen so the detector's actual
    configured behavior is visible (and correctable) BEFORE it exists
    (Aryaman 2026-07-18: customs visibility)."""

    model_config = ConfigDict(frozen=True)

    watches: str   # who: "Your patient only" / "Anyone in view" / room
    media: str     # what pixels: whole-room view vs per-person crop
    fires: str     # the fire rule as a sentence
    checks: str    # look cadence as a sentence
    listens: bool  # audio evidence used


class InferDetectorResponse(BaseModel):
    """LLM-inferred fields for the create review screen. Anything the
    model can't confidently determine comes back null/empty so the UI
    can highlight it for the user to fill."""

    model_config = ConfigDict(frozen=True)

    shape: DetectorShape
    category: DetectorCategory
    icon_name: IconName
    sensitivity: DetectorSensitivity
    schedule_config: ScheduleConfig | None = None
    # Set when shape == 'state': the inferred rising-edge episode gap
    # (seconds). The review screen may show/adjust it; null otherwise.
    episode_gap_seconds: int | None = None
    # Compiled-behavior preview (pure classification, no extra LLM
    # spend). Null only if composition fails — the review screen then
    # hides the card rather than blocking creation.
    plan: DetectorPlan | None = None


class UpdateDetectorRequest(BaseModel):
    """Body of PATCH /v1/detectors/{id}. All fields optional.

    `source` is server-managed — once a detector exists you can't flip
    it between preset and custom. Everything else is mutable.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Name | None = None
    icon_name: IconName | None = None
    description: Description | None = None
    category: DetectorCategory | None = None
    shape: DetectorShape | None = None
    where_scope: Scope | None = None
    when_scope: Scope | None = None
    sensitivity: DetectorSensitivity | None = None
    is_on: bool | None = None
    custom_prompt: Prompt | None = None


class DetectorLibraryEntry(BaseModel):
    """One detector-catalog entry, returned by
    GET /v1/patients/{id}/detectors/library. `already_added` is true when the
    patient already has a detector with this identifier (so the picker can
    pre-check / disable it)."""

    model_config = ConfigDict(frozen=True)

    key: str
    name: Name
    icon_name: IconName
    description: Description
    category: DetectorCategory
    cohort: str                 # "child" | "adult" | "both"
    recommended: bool
    already_added: bool


class AddFromLibraryRequest(BaseModel):
    """Body of POST /v1/patients/{id}/detectors/from-library — the catalog keys
    to instantiate as preset detectors for the patient. Unknown keys and keys
    the patient already has are skipped server-side; an empty list is a 422."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: list[str] = Field(min_length=1)
