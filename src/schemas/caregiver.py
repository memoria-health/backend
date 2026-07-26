"""Pydantic boundary models for the caregiver API surface.

Wire format is snake_case (FastAPI / Pydantic default). The iOS Codable
models on the other side map to camelCase via explicit `CodingKeys`.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

DisplayName = Annotated[str, Field(min_length=1, max_length=80)]
Email = Annotated[str, Field(min_length=3, max_length=255)]


def _to_e164(value: object) -> object:
    """Normalize common phone input to E.164 so the iOS app can send a plain
    10-digit US number, "(464) 474-4928", "+1 464-474-4928", etc. A bare
    10-digit number is assumed US (+1); "+"-prefixed input keeps its country
    code. Non-strings pass through for Pydantic to reject normally."""
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not s:
        return s
    digits = "".join(c for c in s if c.isdigit())
    if s.startswith("+"):
        return "+" + digits
    if len(digits) == 10:
        return "+1" + digits
    if len(digits) == 11 and digits.startswith("1"):
        return "+" + digits
    return "+" + digits


# E.164: leading +, then 1–15 digits. Input is normalized first (so a plain
# 10-digit number from the iOS app is accepted and stored as +1XXXXXXXXXX).
Phone = Annotated[str, BeforeValidator(_to_e164), Field(pattern=r"^\+[1-9]\d{1,14}$")]


class CaregiverResponse(BaseModel):
    """Full caregiver row returned by GET / POST / PATCH /v1/me."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    display_name: DisplayName
    email: Email                  # lowercased; mirrored from Supabase auth.users
    phone: Phone | None
    created_at: datetime
    updated_at: datetime
    # Web onboarding walkthrough — false until the user dismisses or
    # completes the dashboard modal. The web client reads this on load
    # and PATCHes it true via PATCH /v1/me.
    onboarded_web: bool
    # IANA timezone scheduled detectors evaluate in. Auto-set from the
    # web client's browser timezone.
    timezone: str
    # Per-account visibility of the dashboard's "Live analyses" feed.
    # Defaults true; flip false to hide it for specific accounts.
    show_live_analyses: bool
    # Terms of Service + Privacy Notice acceptance. `tos_accepted_at` is null
    # until the caregiver accepts (the signup gate blocks the app while null);
    # `tos_version` is the document version they accepted. Clients read these
    # to decide whether to show the blocking accept-terms gate (null, or a
    # version older than the current one → re-prompt).
    tos_accepted_at: datetime | None
    tos_version: str | None


class CreateCaregiverRequest(BaseModel):
    """Body of POST /v1/me. Idempotent — if a caregiver row already
    exists for the JWT's `sub`, that row is returned and these fields
    are ignored. Email is taken from the JWT, not this body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: DisplayName
    phone: Phone | None = None
    # The ToS/Privacy version the caregiver accepted at signup. The iOS app
    # gates account creation on acceptance and sends it, so the row is born
    # consented. Optional because the web provisions the row lazily on first
    # load and captures acceptance via the PATCH gate instead. When present,
    # provisioning stamps tos_accepted_at server-side.
    tos_version: str | None = Field(default=None, max_length=40)


class UpdateCaregiverRequest(BaseModel):
    """Body of PATCH /v1/me. All fields optional.

    Callers use `model_dump(exclude_unset=True)` to distinguish "field
    omitted" (leave column unchanged) from explicit `null` (clear the
    column). Email is intentionally NOT writable here — the iOS app
    calls Supabase's `auth.update(user: UserAttributes(email:))` which
    triggers the verification flow; a DB trigger then mirrors the new
    `auth.users.email` back into `caregivers.email`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: DisplayName | None = None
    phone: Phone | None = None
    # Web client PATCHes this true when the user finishes/skips the
    # onboarding walkthrough. Only ever set to true from the client.
    onboarded_web: bool | None = None
    # IANA timezone (e.g. "America/New_York"). Auto-set by the web
    # client from the browser's resolved timezone.
    timezone: str | None = Field(default=None, max_length=64)
    # The ToS/Privacy version the caregiver accepted via the web accept-terms
    # gate (the web provisions the row before acceptance, so it records consent
    # here). When present, the server stamps tos_accepted_at; the timestamp is
    # never client-writable.
    tos_version: str | None = Field(default=None, max_length=40)
