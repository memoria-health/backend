"""Pydantic boundary models for the caregiver API surface.

Wire format is snake_case (FastAPI / Pydantic default). The iOS Codable
models on the other side map to camelCase via explicit `CodingKeys`.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

DisplayName = Annotated[str, Field(min_length=1, max_length=80)]
Email = Annotated[str, Field(min_length=3, max_length=255)]
# E.164: leading +, then 1–15 digits.
Phone = Annotated[str, Field(pattern=r"^\+[1-9]\d{1,14}$")]


class CaregiverResponse(BaseModel):
    """Full caregiver row returned by GET / POST / PATCH /v1/me."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    display_name: DisplayName
    email: Email                  # lowercased; mirrored from Supabase auth.users
    phone: Phone | None
    created_at: datetime
    updated_at: datetime


class CreateCaregiverRequest(BaseModel):
    """Body of POST /v1/me. Idempotent — if a caregiver row already
    exists for the JWT's `sub`, that row is returned and these fields
    are ignored. Email is taken from the JWT, not this body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    display_name: DisplayName
    phone: Phone | None = None


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
