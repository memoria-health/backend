"""Pydantic boundary model for cameras.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Camera.swift`.
The iOS app only lists + deletes; v0 has no create body type.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Room = Annotated[str, Field(min_length=1, max_length=80)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]

# Mirrors the DB CHECK constraint on cameras.desired_state. Pydantic
# Literal raises ValidationError if the DB ever returns something
# outside this set — fail-loud on schema drift between DB + code.
DesiredState = Literal["connected", "disconnected"]


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
    # Stronger than `disabled` — when 'disconnected', the wall phone
    # itself leaves the LiveKit room. Reconnect takes ~1-2s after
    # flipping back to 'connected'. See cameras.desired_state in
    # supabase/schema.sql for the underlying CHECK constraint.
    desired_state: DesiredState
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
    # Flip this from the dashboard's 3-dot menu. Phone polls
    # /v1/cameras/me/desired-state every 1s and reconciles by leaving
    # or rejoining the LiveKit room.
    desired_state: DesiredState | None = None


class DesiredStateResponse(BaseModel):
    """GET /v1/cameras/me/desired-state response body. Polled by the
    wall phone every 1s; the phone reconciles its LiveKit connection
    state to match what's returned here.

    `polled_at` is the server's current time — useful for the phone to
    detect a stale or skewed clock vs the backend.
    """

    model_config = ConfigDict(frozen=True)

    desired_state: DesiredState
    polled_at: datetime


class CameraDetectorsResponse(BaseModel):
    """GET /v1/cameras/{id}/detectors body — list of detector IDs
    currently assigned to the camera. Frontend cross-references against
    the patient's full detector list to render toggles.
    """

    model_config = ConfigDict(frozen=True)

    detector_ids: list[UUID]


class SetCameraDetectorsRequest(BaseModel):
    """PUT /v1/cameras/{id}/detectors body — replace the camera's
    assignment set with exactly `detector_ids`.

    Every ID must belong to the same patient as the camera; otherwise
    the request is rejected (no cross-patient leakage). Empty list is
    valid — it means the camera runs no per-tick detectors (only the
    implicit `people_present`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector_ids: list[UUID]
