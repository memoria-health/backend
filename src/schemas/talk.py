"""Pydantic boundary models for caregiver Talk mode (Phase 3).

Talk mode is a full-duplex audio toggle between the caregiver and the
wall phone(s) in a patient's household:

- `POST /v1/patients/{patient_id}/talk/start` — server picks an
  active camera (most recent analyses row with people_present=true in
  the last 30s) and sends a LiveKit data packet to that camera with
  `is_active=true`. Other cameras get `is_active=false`. If no active
  camera is known, the packet is sent to every camera with
  `is_active=true` (fan-out fallback).
- `POST /v1/patients/{patient_id}/talk/end` — server sends
  `is_active=false` to every camera in the room.

Cameras subscribe to the caregiver's audio track via LiveKit
(canSubscribe=true on the camera grant) and gate playback locally on
`is_active`. Routing is enforced camera-side via volume; no
server-side track muting.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TalkStartResponse(BaseModel):
    """Returned to the caregiver app after `/talk/start`.

    The UI uses `active_camera_id` to label the on-air banner —
    "Talking to Kitchen" if known, "Talking to all rooms" if None.
    """

    model_config = ConfigDict(frozen=True)

    active_camera_id: UUID | None
