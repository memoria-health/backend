"""Pydantic boundary models for the patient API surface.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Patient.swift` —
same fields, same enum values. Server-managed fields (`state`,
`status_line`, `last_event_at`, `battery_percent`) are not writable
from the iOS app — see `UpdatePatientRequest`.
"""

from datetime import date, datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Relation(str, Enum):
    parent = "parent"
    spouse = "spouse"
    grandparent = "grandparent"
    sibling = "sibling"
    client = "client"
    other = "other"


class PatientState(str, Enum):
    calm = "calm"
    critical = "critical"
    offline = "offline"


Name200 = Annotated[str, Field(min_length=1, max_length=200)]
Name80 = Annotated[str, Field(min_length=1, max_length=80)]
Address = Annotated[str, Field(min_length=1, max_length=500)]
Gender = Annotated[str, Field(min_length=1, max_length=40)]
BatteryPct = Annotated[int, Field(ge=0, le=100)]
Email = Annotated[str, Field(min_length=3, max_length=255)]


class PatientResponse(BaseModel):
    """Full patient row returned by list / get / create / update endpoints."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    caregiver_id: UUID
    full_name: Name200
    preferred_name: Name80
    date_of_birth: date | None
    gender: Gender | None
    relation: Relation
    home_address: Address | None
    state: PatientState
    status_line: str
    last_event_at: datetime | None
    battery_percent: BatteryPct | None
    cameras_paused: bool
    photo_storage_key: str | None
    created_at: datetime
    updated_at: datetime
    # Per-patient email alerts. enabled = master switch; recipients =
    # ADDITIONAL addresses beyond the owning caregiver's account email
    # (which always receives alerts for this patient when enabled).
    email_alerts_enabled: bool
    email_alert_recipients: list[Email]


class PhotoURLResponse(BaseModel):
    """Returned by `GET /v1/patients/{id}/photo-url`. Signed URL is
    short-lived (default 10 min). The caregiver app re-mints on every
    AsyncImage refresh; iOS's NSURLCache handles the bytes-side
    caching."""

    model_config = ConfigDict(frozen=True)

    url: str
    expires_in_seconds: int


class CreatePatientRequest(BaseModel):
    """Body of POST /v1/patients.

    Server-managed fields (state / status_line / last_event_at /
    battery_percent / cameras_paused) cannot be set at creation —
    they default to "calm / All quiet / nulls / false" and only
    transition via wall-phone telemetry or the dedicated PATCH endpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    full_name: Name200
    preferred_name: Name80
    relation: Relation
    date_of_birth: date | None = None
    gender: Gender | None = None
    home_address: Address | None = None


class UpdatePatientRequest(BaseModel):
    """Body of PATCH /v1/patients/{id}. All fields optional.

    Writable:
        full_name, preferred_name, date_of_birth, gender, relation,
        home_address, cameras_paused.

    Server-managed (not writable here):
        state, status_line, last_event_at, battery_percent — these come
        from wall-phone telemetry. The Pydantic model doesn't expose
        them, so even a malicious client can't sneak a payload through.
        `extra="forbid"` rejects unknown keys for the same reason.

    Use `model_dump(exclude_unset=True)` server-side to distinguish
    "field omitted" from explicit `null`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    full_name: Name200 | None = None
    preferred_name: Name80 | None = None
    date_of_birth: date | None = None
    gender: Gender | None = None
    relation: Relation | None = None
    home_address: Address | None = None
    cameras_paused: bool | None = None
    # Per-patient email alert settings. Omitting leaves unchanged;
    # sending `email_alert_recipients` replaces the whole list.
    email_alerts_enabled: bool | None = None
    email_alert_recipients: list[Email] | None = None
