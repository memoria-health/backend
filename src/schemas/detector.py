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
    point_in_time = "point_in_time"
    state = "state"


class DetectorSensitivity(str, Enum):
    low = "low"
    standard = "standard"
    high = "high"
    off = "off"


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
    created_at: datetime
    updated_at: datetime


class CreateDetectorRequest(BaseModel):
    """Body of POST /v1/patients/{id}/detectors.

    Server forces `source = custom` — preset detectors come only from
    the auto-seed on patient create. `custom_prompt` is required (it's
    what makes a custom detector custom); the v0 backend stores the
    string verbatim and a later plan compiles it into a runtime config
    via an LLM call.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Name
    icon_name: IconName
    description: Description
    category: DetectorCategory
    shape: DetectorShape
    where_scope: Scope
    when_scope: Scope
    custom_prompt: Prompt
    sensitivity: DetectorSensitivity = DetectorSensitivity.standard


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
