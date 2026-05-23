"""Pydantic boundary models for caregiver Talk mode (Phase 3).

Talk is a phone-call metaphor driven entirely from the iOS Live tab.
The caregiver picks which camera they're calling (via the carousel);
the server's only job is to flip `is_active=true` on that camera and
`is_active=false` on every sibling. See `src/api/talk.py` for the
endpoint behaviour.
"""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class ActiveCameraResponse(BaseModel):
    """Returned by `GET /v1/patients/{id}/active-camera`. The iOS Live
    tab uses `camera_id` to default the carousel to the right tile.
    `None` means no qualifying analyses row in the last 30s; the
    caller falls back to the first online camera in the patient's
    list.
    """

    model_config = ConfigDict(frozen=True)

    camera_id: UUID | None


class TalkStartRequest(BaseModel):
    """Body of `POST /v1/patients/{id}/talk/start`. `camera_id` is the
    explicit routing target — server sends `is_active=true` to that
    camera and `is_active=false` to every other camera paired to the
    patient. Re-POSTing with a new `camera_id` transfers an active
    call atomically.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    camera_id: UUID
