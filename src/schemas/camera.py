"""Pydantic boundary model for cameras.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Camera.swift`.
The iOS app only lists + deletes; v0 has no create body type.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Room = Annotated[str, Field(min_length=1, max_length=80)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]


class CameraResponse(BaseModel):
    """Full camera row."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    patient_id: UUID
    display_id: int
    room: Room
    icon_name: IconName
    online: bool
    # User-facing "off" toggle. When true, the detector worker
    # stays in the room but skips the LLM call (synthetic "skipped"
    # row goes into analyses). Web hides disabled cameras from the
    # live grid + analyses feed.
    disabled: bool
    # `cam_<8 hex>` once the pairing handshake completes; None on rows
    # that hold an open pairing code but haven't been claimed by a
    # wall phone yet. Exposed to the caregiver app so the Live tab
    # can map a Camera row to a subscribed LiveKit track without
    # reconstructing the identity from the UUID prefix client-side.
    livekit_identity: str | None
    last_seen_at: datetime | None
    created_at: datetime


class UpdateCameraRequest(BaseModel):
    """PATCH /v1/cameras/{id} body — partial update of mutable fields.

    Only the user-facing toggles are exposed here. Lifecycle / state-
    machine fields (state, no_people_count, livekit_identity, …) are
    owned by the detector pipeline and the pairing flow respectively.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    disabled: bool | None = None
