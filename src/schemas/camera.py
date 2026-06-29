"""Pydantic boundary model for cameras.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Camera.swift`.
The iOS app only lists + deletes; v0 has no create body type.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

Room = Annotated[str, Field(min_length=1, max_length=80)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]

# Mirrors the DB CHECK constraint on cameras.desired_state. Pydantic
# Literal raises ValidationError if the DB ever returns something
# outside this set, fail-loud on schema drift between DB + code.
DesiredState = Literal["connected", "disconnected"]

# Mirrors the DB CHECK constraint on cameras.source_type. 'iphone' is
# the existing NemaraCam wall-phone path. 'pi' is the Raspberry Pi
# MJPEG-relay ingest path. 'gateway' is the mini-PC RTSP gateway — its
# OWN third path: it carries an ingest token like 'pi' but publishes
# DIRECTLY to LiveKit (no relay), so it shares none of the relay logic.
SourceType = Literal["iphone", "pi", "gateway"]

# A Pi stream URL must be an HTTP(S) URL. We do not parse / validate
# routing here; the worker process will discover unreachable URLs
# at connect time. Length cap protects against accidentally storing
# a multi-MB blob.
PiStreamUrl = Annotated[str, Field(min_length=8, max_length=2048)]

# A Pi audio URL has the same shape and constraints. We split the
# type alias for clarity at field declarations even though it is
# identical to PiStreamUrl.
PiAudioUrl = Annotated[str, Field(min_length=8, max_length=2048)]


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
    # LLM-only pause: when true, the tick skips the Gemini call but
    # video keeps streaming and the tile stays visible in the live
    # grid. Lighter than `disabled` (which hides the tile entirely).
    ai_paused: bool
    # Stronger than `disabled`, when 'disconnected', the wall phone
    # itself leaves the LiveKit room. Reconnect takes ~1-2s after
    # flipping back to 'connected'. See cameras.desired_state in
    # supabase/schema.sql for the underlying CHECK constraint.
    desired_state: DesiredState
    # 'iphone' (NemaraCam wall phone) or 'pi' (Raspberry Pi ingest).
    # The caregiver app uses this to hide source-incompatible UI
    # affordances; e.g. Talk mode is iPhone-only until Pi gets a
    # speaker (Phase 6).
    source_type: SourceType
    # The MJPEG-over-HTTP endpoint for Pi cameras. Present only when
    # source_type='pi'. Forbidden when source_type='iphone' (enforced
    # at the DB layer too via CHECK constraint).
    pi_stream_url: PiStreamUrl | None = None
    # Optional PCM-over-HTTP audio endpoint for Pi cameras with a USB
    # microphone. NULL when the camera is video-only (no LiveKit
    # audio track is published, the agitation/speech detectors
    # downstream degrade gracefully). Forbidden on iPhone cameras
    # (their audio rides on the same LiveKit publisher as video).
    pi_audio_url: PiAudioUrl | None = None
    # `cam_<8 hex>` once the pairing handshake completes; None on rows
    # that hold an open pairing code but haven't been claimed by a
    # wall phone yet. Exposed to the caregiver app so the Live tab
    # can map a Camera row to a subscribed LiveKit track without
    # reconstructing the identity from the UUID prefix client-side.
    livekit_identity: str | None
    created_at: datetime
    # True once the wall phone has called /v1/cameras/pair and we've
    # set the api_key_hash. False = "row exists with an open pairing
    # code, no wall phone has claimed it yet." Caregiver clients use
    # this to show a "PAIRING" badge, surface a "Show pairing code"
    # button on the detail page, and skip the LiveKit tile subscription
    # (there's no published track until pairing completes).
    paired: bool
    # True while the Memoria Sensor app is RUNNING on this camera's
    # iPhone (it polls desired-state every 1s when open). false = app
    # closed / phone off → clients show "App closed, open Memoria
    # Sensor" instead of a reconnectable "Disconnected". Only
    # meaningful for source_type='iphone'; Pi cameras stay true (no app
    # to close). Defaulted True so older backends / un-migrated rows /
    # any not-yet-updated construction site degrade gracefully to
    # "present" and never falsely show "app closed".
    app_present: bool = True

    @model_validator(mode="after")
    def _validate_source_pi_url_consistency(self) -> "CameraResponse":
        """Vestigial URL columns: pi_stream_url and pi_audio_url no
        longer carry source-type-XOR semantics after the push refactor
        (Pi POSTs frames outbound; backend never pulls). The columns
        remain nullable for backwards compat but the API does not
        enforce a presence rule on them.
        """
        return self


class UpdateCameraRequest(BaseModel):
    """PATCH /v1/cameras/{id} body — partial update of mutable fields.

    Only the user-facing toggles are exposed here. Lifecycle / state-
    machine fields (state, no_people_count, livekit_identity, …) are
    owned by the detector pipeline and the pairing flow respectively.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    disabled: bool | None = None
    # LLM-only pause. Video keeps streaming; AI tick is skipped.
    ai_paused: bool | None = None
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
    # The caregiver-assigned room/display name for this camera (e.g.
    # "Living Room"). Echoed on every 1s poll so the wall phone can show
    # its human name (and pick up renames live) instead of only the
    # opaque cam_<hex> identity.
    room: str


class UpdateMyDesiredStateRequest(BaseModel):
    """PATCH /v1/cameras/me/desired-state body. The wall phone calls this
    whenever the on-device Start/Stop button is tapped, so the cameras
    row stays in sync with the actual phone state. Without this, the
    web dashboard kept showing stale 'Disconnect' buttons after the
    phone stopped itself, and a subsequent caregiver tap acted on
    out-of-date intent.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    desired_state: DesiredState


class CameraDetectorsResponse(BaseModel):
    """GET /v1/cameras/{id}/detectors body — list of detector IDs
    currently assigned to the camera. Frontend cross-references against
    the patient's full detector list to render toggles.
    """

    model_config = ConfigDict(frozen=True)

    detector_ids: list[UUID]


class SetCameraDetectorsRequest(BaseModel):
    """PUT /v1/cameras/{id}/detectors body, replace the camera's
    assignment set with exactly `detector_ids`.

    Every ID must belong to the same patient as the camera; otherwise
    the request is rejected (no cross-patient leakage). Empty list is
    valid, it means the camera runs no per-tick detectors (only the
    implicit `people_present`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector_ids: list[UUID]


class PiProvisionRequest(BaseModel):
    """Body of POST /v1/patients/{patient_id}/cameras/pi-provision.

    Creates a Pi camera row in one shot. Unlike the iPhone pairing
    flow, there is no two-step exchange: the Pi has no UI to enter a
    code, and the backend worker authenticates to LiveKit using its
    own credentials, so no api_key needs to be handed to the Pi
    device itself.

    Two mutually-exclusive modes to specify the camera's MJPEG URL:

      - `discovery_id`: pick a discovered Pi from the announce table
        (the normal customer flow; the caregiver app gets the id
        from GET /v1/cameras/discovered). The URL fields on the row
        come from the discovery record and the discovery is marked
        claimed.
      - `pi_stream_url` (+ optional `pi_audio_url`): for ops /
        scripted provisioning when the caregiver already knows the
        URL and no announce record exists. Kept so the existing
        admin and test workflows still work.

    Exactly one of the two modes must be set; both or neither raises
    a 422.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    room: Room
    icon_name: IconName = "tv"
    # Mode A (recommended): pair from a discovery. The Pi's hardware ID
    # and human label come from the discovery record.
    discovery_id: UUID | None = None
    # Mode B (manual): direct URLs (vestigial, accepted but unused after
    # the push refactor since the backend no longer pulls from the Pi).
    # Kept for ops/scripted workflows and to avoid breaking old test
    # fixtures.
    pi_stream_url: PiStreamUrl | None = None
    pi_audio_url: PiAudioUrl | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> "PiProvisionRequest":
        has_discovery = self.discovery_id is not None
        has_direct_url = self.pi_stream_url is not None
        if has_discovery and has_direct_url:
            raise ValueError(
                "Specify either discovery_id or pi_stream_url, not both"
            )
        if not has_discovery and not has_direct_url:
            raise ValueError(
                "Must specify either discovery_id or pi_stream_url"
            )
        return self


class PiProvisionResponse(BaseModel):
    """Handed back to the caregiver app after provisioning. Carries the
    LiveKit identity + room so the app can confirm the camera is wired
    before the Pi starts pushing frames.

    NOTE: the camera's ingest bearer token is NOT delivered here. The
    Pi receives it via the next /v1/cameras/pi-announce response (which
    is the only direct Pi-to-backend channel before pairing completes).
    """

    model_config = ConfigDict(frozen=True)

    camera_id: UUID
    patient_id: UUID
    livekit_identity: str
    livekit_room: str
