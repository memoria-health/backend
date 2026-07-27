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


class AlertTrigger(str, Enum):
    """USER-FACING spine (2026-07-26): which moment in the detection stream
    fires the alert. Derived from the caregiver's words; the only manual
    choice is leading↔trailing (↔duration) on a sustained positive."""

    leading = "leading"    # the moment it starts / happens
    trailing = "trailing"  # when the activity ends (sustained only)
    duration = "duration"  # once it's held >= duration_seconds (sustained only)
    absence = "absence"    # NOT detected within its active window (when_scope)


class DetectorSensitivity(str, Enum):
    low = "low"
    standard = "standard"
    high = "high"


class ScheduleCadence(str, Enum):
    daily = "daily"
    weekly = "weekly"


class ScheduleConfig(BaseModel):
    """Evaluation window for an `absence` detector — legacy fallback; the app
    increasingly uses `when_scope` as THE window mechanism. All fields optional
    so the LLM can return a partial inference (blanks the user fills in on the
    review screen). Times are local "HH:MM" 24h; days_of_week 0=Mon..6=Sun."""

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
# Multi-room scope: room names (or the "ALL ROOMS" sentinel). Bounded so a
# client can't send an unbounded blob. None = every room (back-compat).
Rooms = Annotated[list[Annotated[str, Field(min_length=1, max_length=80)]], Field(max_length=40)]
# Bounded client-supplied durations (seconds): 1s .. 24h. Guards the
# duration-trigger threshold against negative / absurd values from the client.
DurationSeconds = Annotated[int, Field(ge=1, le=86400)]


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
    # Alerting spine (2026-07-26): which moment fires. is_sustained is the
    # internal temporal nature.
    alert_trigger: AlertTrigger = AlertTrigger.leading
    is_sustained: bool = False
    # Duration threshold (seconds) for alert_trigger == duration — the
    # user-meaningful "has it gone on too long" bar. Distinct from
    # episode_gap_seconds (hysteresis) and from the internal M-of-N.
    duration_seconds: int | None = None
    # Legacy display scalar — DERIVED on serialize from where_rooms (no longer
    # stored). "ALL ROOMS" for every-room, else the joined room names.
    where_scope: Scope = "ALL ROOMS"
    # Rooms the detector is scoped to (the canonical stored rep). None → every
    # room (back-compat with rows created before multi-room existed).
    where_rooms: Rooms | None = None
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
    # Caregiver-facing summary of the compiled prompt (3-5 plain lines:
    # "Counts: …" / "Doesn't count: …" / "Caution: …"). Regenerated on
    # every compile. Null while compiling or if summarization failed.
    user_criteria: list[str] | None = None
    # Plain-words change requests accumulated via /refine, oldest first.
    refinements: list[str] | None = None
    # Wandering only: the caregiver's saved exit-door choice, so the picker can
    # pre-highlight it. {box:[x,y,w,h]|null, label, source, no_exit_door}.
    door_config: dict | None = None
    # True for detectors that read FINE visual detail — small motion
    # (seizure/choking) or small objects (medication/stove). Their
    # accuracy varies with camera distance, lighting, and angle, so the
    # caregiver UI shows an expectations notice at enroll + in detail.
    # Server-computed (see api.detectors._accuracy_caveat); never stored.
    accuracy_caveat: bool = False
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
    # Alerting spine (client sends the confirmed inference).
    alert_trigger: AlertTrigger = AlertTrigger.leading
    is_sustained: bool = False
    duration_seconds: DurationSeconds | None = None
    icon_name: IconName = "eye"
    # Rooms the caregiver picked (multi-select). None / ["ALL ROOMS"] = every
    # room. The canonical scope — drives camera auto-assignment.
    where_rooms: Rooms | None = None
    when_scope: Scope = "ALWAYS"
    sensitivity: DetectorSensitivity = DetectorSensitivity.standard
    schedule_config: ScheduleConfig | None = None
    # Set only for shape == 'state' (the rising-edge episode gap, seconds).
    episode_gap_seconds: int | None = None
    # Create-as-draft: the review flow now creates the detector OFF, lands
    # the caregiver on the detail editor to check the compiled criteria,
    # and only then flips it on. Defaults True so any existing caller (and
    # the "turn it on immediately" path) is unaffected.
    is_on: bool = True


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

    watches: str   # who: "Only the person you care for" / "Anyone in view" / room
    fires: str     # the fire rule as a sentence
    checks: str    # look cadence as a sentence
    listens: bool  # audio evidence used


class InferDetectorResponse(BaseModel):
    """LLM-inferred fields for the create review screen. Anything the
    model can't confidently determine comes back null/empty so the UI
    can highlight it for the user to fill."""

    model_config = ConfigDict(frozen=True)

    alert_trigger: AlertTrigger
    is_sustained: bool
    duration_seconds: int | None = None
    category: DetectorCategory
    icon_name: IconName
    sensitivity: DetectorSensitivity
    schedule_config: ScheduleConfig | None = None
    # Set when is_sustained: the inferred rising-edge episode gap (seconds).
    # The review screen may show/adjust it; null otherwise.
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
    # Alerting spine (the toggle the picker sets).
    alert_trigger: AlertTrigger | None = None
    is_sustained: bool | None = None
    duration_seconds: DurationSeconds | None = None
    # Re-scope the detector's rooms; the endpoint replaces its camera
    # assignments to match. ["ALL ROOMS"] or null-via-omit = every room.
    where_rooms: Rooms | None = None
    when_scope: Scope | None = None
    sensitivity: DetectorSensitivity | None = None
    is_on: bool | None = None
    custom_prompt: Prompt | None = None


class RefineDetectorRequest(BaseModel):
    """Body of POST /v1/detectors/{id}/refine — one plain-words change
    request from the caregiver ("don't count eating as choking"). The
    compiler receives the FULL accumulated list on every recompile.

    `preview` = dry-run: compile the detector WITH this feedback applied
    and return the resulting criteria, but persist nothing (no refinement
    appended, no row update). This backs the edit UI's "Re-generate"
    button — the caregiver sees what the change would produce, and only a
    real (non-preview) submit on Save commits it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    feedback: Prompt
    preview: bool = False


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
    # Fine-detail detector → the add flow shows the accuracy notice
    # before the caregiver enrolls it. See api.detectors._accuracy_caveat.
    accuracy_caveat: bool = False


class AddFromLibraryRequest(BaseModel):
    """Body of POST /v1/patients/{id}/detectors/from-library — the catalog keys
    to instantiate as preset detectors for the patient. Unknown keys and keys
    the patient already has are skipped server-side; an empty list is a 422."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    keys: list[str] = Field(min_length=1)
