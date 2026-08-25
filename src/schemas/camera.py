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

# Mirrors the DB CHECK constraint on cameras.room_class (room-derived
# privacy, 2026-07-27). Pydantic Literal fails loud on schema drift.
RoomClass = Literal["bathroom", "bedroom", "other"]

# Mirrors the DB CHECK constraint on cameras.desired_state. Pydantic
# Literal raises ValidationError if the DB ever returns something
# outside this set, fail-loud on schema drift between DB + code.
DesiredState = Literal["connected", "disconnected"]

# Mirrors the DB CHECK constraint on cameras.source_type. 'iphone' is
# the NemaraCam wall-phone path. 'gateway' covers anything that
# publishes DIRECTLY to LiveKit with an ingest token — the mini-PC RTSP
# gateway and the hosted camera bridge.
#
# 'pi' was the Raspberry Pi MJPEG-relay path, removed 2026-08-22. It had
# no ingest endpoint left after 2026-08-21, so a 'pi' row could be
# created and could never deliver a frame; no Pi ever ran in production.
SourceType = Literal["iphone", "gateway"]

class CameraResponse(BaseModel):
    """Full camera row."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    # Compatibility only — the FIRST covered patient, NULL when the sensor
    # covers nobody. Coverage lives in `camera_patients`; nothing should
    # decide anything from this.
    patient_id: UUID | None = None
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
    # 'iphone' (NemaraCam wall phone) or 'gateway' (anything publishing
    # straight to LiveKit). The caregiver app uses this to hide
    # source-incompatible affordances — Talk needs a speaker, which only
    # the wall phone has.
    source_type: SourceType
    # `cam_<8 hex>` once the pairing handshake completes; None on rows
    # that hold an open pairing code but haven't been claimed by a
    # wall phone yet. Exposed to the caregiver app so the Live tab
    # can map a Camera row to a subscribed LiveKit track without
    # reconstructing the identity from the UUID prefix client-side.
    livekit_identity: str | None
    # The LiveKit room this sensor PUBLISHES INTO — the address a viewer
    # must join to see it. Exposed to clients because it is not
    # derivable from the patient being viewed: it is the ACCOUNT's room
    # for a sensor, and a `patient_<uuid>` room for a wall phone paired
    # before 2026-08-22. A client that builds the string itself joins a
    # room the sensor never appears in and shows a black tile.
    #
    # Declared TWICE until 2026-08-22 — once here and once above. The
    # second declaration silently won, so the first comment described
    # behaviour no longer in force and any change to it did nothing.
    livekit_room: str | None = None
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
    # meaningful for source_type='iphone'; gateway sensors stay true (no
    # app to close). Defaulted True so older backends / un-migrated rows /
    # any not-yet-updated construction site degrade gracefully to
    # "present" and never falsely show "app closed".
    app_present: bool = True
    # ── Room-derived privacy (2026-07-27) ────────────────────────────
    # VLM room verdict + when it was finalized. All three are null until
    # the classifier runs (lazily, on next connect); clients key the
    # one-time "sensitive room → privacy is on by default" card off a
    # fresh room_classified_at. NEVER settable via PATCH.
    room_class: RoomClass | None = None
    room_class_confidence: float | None = None
    room_classified_at: datetime | None = None
    # Denormalized privacy verdict: true for bathroom/bedroom (and the
    # classifier's fail-closed fallback). Locked — no user toggle;
    # written only by the classifier / internal reclassify endpoint.
    privacy_enabled: bool = False
    # The facilities audio kill, so a client can read back what it set.
    audio_capture_enabled: bool = True


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
    # Facilities audio kill: False = no audio track is ever published by
    # this sensor, severed at the publisher. Moved off `patients` on
    # 2026-08-22 — it is a statement about a physical room ("no recording
    # in here"), not about a person, and a sensor covering nobody had no
    # patient to read it from. Reaches the bridge within ~0.4s via the
    # roster long-poll fingerprint.
    audio_capture_enabled: bool | None = None
    # Flip this from the dashboard's 3-dot menu. Phone polls
    # /v1/cameras/me/desired-state every 1s and reconciles by leaving
    # or rejoining the LiveKit room.
    desired_state: DesiredState | None = None
    # Cosmetic display/room name. Renaming CLEARS the camera's room
    # classification (room_classified_at + confidence) so the room
    # classifier re-runs — the misclassification escape hatch of the
    # room-privacy plan ⚑3 (a rename signals "this is not the room you
    # think it is").
    room: Room | None = None
    # NOTE: the privacy fields (room_class / room_class_confidence /
    # room_classified_at / privacy_enabled) are deliberately ABSENT —
    # they are NEVER settable via the public PATCH (extra="forbid"
    # rejects them 422). Only the room classifier and the internal
    # reclassify endpoint write them.


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
    # Room-derived privacy verdict (2026-07-27). The publisher reads
    # this on its 1s poll and, when true, restricts its tracks'
    # subscription permissions to detector_worker_* identities (and
    # ignores inbound talk) — the SFU-enforced side of per-camera
    # privacy. Defaults False so an older backend response shape keeps
    # a not-yet-updated sensor publishing normally.
    privacy_enabled: bool = False


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


class AccountCameraResponse(BaseModel):
    """A camera as the ACCOUNT sees it — including ones no patient owns yet."""

    id: UUID
    room: str
    icon_name: str
    online: bool
    source_type: str
    # Coverage, oldest assignment first. Empty until someone assigns it.
    # Equals, with no primary: a primary only ever answered "whose LiveKit
    # room does this publish into", and the room is the account's now.
    patient_ids: list[UUID]
    unassigned: bool


class CoverageRequest(BaseModel):
    """Assign a sensor: name it, pick its icon, and say who it watches.

    An EMPTY `patient_ids` is legal and means "watched by nobody yet".
    The sensor keeps streaming either way -- coverage is attribution, and
    it never starts or stops the video.
    """

    room: str | None = Field(default=None, max_length=80)
    icon_name: str | None = Field(default=None, max_length=80)
    patient_ids: list[UUID] = Field(default_factory=list)
