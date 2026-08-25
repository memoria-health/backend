"""Pydantic boundary models for the patient API surface.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Patient.swift` —
same fields, same enum values. Server-managed fields (`state`,
`status_line`, `last_event_at`) are not writable from the iOS app —
see `UpdatePatientRequest`.
"""

from datetime import date, datetime
from enum import Enum
from typing import Annotated
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


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


class Region(str, Enum):
    """Data-residency region for a patient's data. The client's country
    picker (United States / India) maps to these values; the backend routes
    storage (Backblaze bucket, Supabase key prefix) and lift/breach/DSAR
    scoping off the patient's region. See `patients.region` in schema.sql.
    """

    us = "US"
    india = "IN"


Name200 = Annotated[str, Field(min_length=1, max_length=200)]
Name80 = Annotated[str, Field(min_length=1, max_length=80)]
Address = Annotated[str, Field(min_length=1, max_length=500)]
Gender = Annotated[str, Field(min_length=1, max_length=40)]
Email = Annotated[str, Field(min_length=3, max_length=255)]
# ISO 3166-1 alpha-2 country picked at onboarding (e.g. 'US', 'IN', 'GB').
# Recorded verbatim on the row; the data-residency `region` (Region enum) is
# DERIVED from it server-side via region_for_country() and is the only value
# that drives storage routing / legal scoping.
Country = Annotated[str, Field(min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")]


def _known_zone(v: str) -> str:
    """Reject a timezone the server cannot actually resolve. Storing an
    unknown name would not error anywhere — it would silently fall back
    to the caregiver's zone forever, which reads as "the setting does
    nothing" rather than as a bad value."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    try:
        ZoneInfo(v)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"unknown IANA timezone: {v!r}")
    return v


# IANA name, e.g. "America/Los_Angeles". Validated against the server's
# own tz database so what is stored is always resolvable.
TimezoneName = Annotated[
    str, Field(min_length=1, max_length=64), AfterValidator(_known_zone)]


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
    # IANA timezone THIS HOME is in — every day boundary the caregiver sees
    # resolves here first. NULL means "never set": fall back to the
    # caregiver's own zone, which is what all homes did before this existed.
    timezone: str | None
    # Country picked at onboarding (ISO alpha-2). The residency `region` below
    # is derived from it; both are set once and immutable thereafter.
    country: Country
    # Data-residency region for this patient's data ('US' | 'IN'), DERIVED from
    # `country`. Drives storage routing + liftability.
    region: Region
    state: PatientState
    status_line: str
    last_event_at: datetime | None
    cameras_paused: bool
    photo_storage_key: str | None
    created_at: datetime
    updated_at: datetime
    # Per-patient email alerts. enabled = master switch; recipients =
    # ADDITIONAL addresses beyond the owning caregiver's account email
    # (which always receives alerts for this patient when enabled).
    email_alerts_enabled: bool
    email_alert_recipients: list[Email]
    # Data-collection consent. enabled = master research opt-in (default
    # OFF). decided_at = when the caregiver decided; null = never asked,
    # which the web + iOS apps use to show the first-open consent banner.
    data_collection_enabled: bool
    data_collection_decided_at: datetime | None
    # DEPRECATED (2026-07-27): superseded by per-camera
    # cameras.privacy_enabled (room-derived privacy). Read-only echo of
    # the historical column — no longer PATCH-able or enforced; kept on
    # the wire until both clients ship without the toggle.
    privacy_mode_enabled: bool

    # Evaluation mode, surfaced to the caregiver as an Accuracy setting on
    # all three clients. ON by default. Reversed on 2026-08-05 from the
    # operator-only design: a household should be able to see, and turn
    # off, the fact that a reviewer looks at masked footage of their home.
    event_gate_enabled: bool = True

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
    cameras_paused) cannot be set at creation — they default to
    "calm / All quiet / null / false" and only transition via
    wall-phone telemetry or the dedicated PATCH endpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    full_name: Name200
    preferred_name: Name80
    relation: Relation
    date_of_birth: date | None = None
    gender: Gender | None = None
    home_address: Address | None = None
    # IANA timezone of the patient's home. Omitted → the server copies the
    # creating caregiver's zone, which is right for the common case where
    # they live together and a sane starting point when they do not.
    timezone: TimezoneName | None = None
    # Country this home is in (ISO alpha-2). Optional + defaults to 'US' so
    # pre-country clients keep working during rollout. The server DERIVES the
    # data-residency `region` from it (region_for_country) — the client never
    # sends region directly. Immutable after creation (not in
    # UpdatePatientRequest, and the DB trigger blocks client-path changes).
    country: Country = "US"


class UpdatePatientRequest(BaseModel):
    """Body of PATCH /v1/patients/{id}. All fields optional.

    Writable:
        full_name, preferred_name, date_of_birth, gender, relation,
        home_address, cameras_paused.

    Server-managed (not writable here):
        state, status_line, last_event_at — these come from wall-phone
        telemetry. The Pydantic model doesn't expose them, so even a
        malicious client can't sneak a payload through. `extra="forbid"`
        rejects unknown keys for the same reason.

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
    # Move the home to another zone. Every day boundary shifts with it.
    timezone: TimezoneName | None = None
    cameras_paused: bool | None = None
    # Per-patient email alert settings. Omitting leaves unchanged;
    # sending `email_alert_recipients` replaces the whole list.
    email_alerts_enabled: bool | None = None
    email_alert_recipients: list[Email] | None = None
    # Data-collection research opt-in. Sending this (true OR false)
    # records a consent decision: the handler stamps
    # `data_collection_decided_at` server-side. `decided_at` itself is
    # NOT writable here (server-managed) — `extra="forbid"` rejects it.
    data_collection_enabled: bool | None = None
    # Evaluation mode. Holds this patient's alerts for a few seconds so a
    # Memoria reviewer can drop false alarms before the family is told.
    # Caregiver-writable as of 2026-08-05: it used to be operator-only,
    # but the setting is what makes it consented rather than something
    # done to a household without their knowledge. Our own
    # /v1/event-gate/toggle writes the SAME column, so either side can
    # change it — the ops route logs every write for that reason.
    event_gate_enabled: bool | None = None
    # NOTE (2026-07-27): `privacy_mode_enabled` is deliberately ABSENT.
    # Privacy is per-camera now (cameras.privacy_enabled, derived from
    # the room class and never user-settable); a client that still sends
    # the old patient-level flag is rejected 422 by `extra="forbid"`.
