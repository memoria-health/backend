"""Experimental: caregiver-triggered server-side clip recorder (data collection).

NON-PRODUCTION. Mounted only when `settings.enable_internals` is true
(see src/main.py).

On a record-clip request, kicks off a server-side LiveKit recording (see
src/internals/clip_recorder.py): the backend connects its OWN subscriber
to the patient's room, captures `duration` seconds of the target
camera's video+audio, and writes ONE MP4 (with the real LiveKit audio)
to the dump dir. No on-device recording — the camera never drops off
LiveKit, the audio is never silenced by WebRTC mic contention, and the
clip is a true `duration` seconds, independent of the model's analysis
window.

Fire-and-forget: the recording runs ~`duration` s in the background, so
the HTTP request returns immediately instead of hanging for the whole
clip.

Auth + ownership reuse the Talk-mode helper verbatim: a caregiver can
only address cameras belonging to a patient they own.
"""

import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.api.talk import _resolve_owned_cameras_or_403
from src.internals import clip_recorder

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["internals"])

_MIN_DURATION = 5
_MAX_DURATION = 60
_DEFAULT_DURATION = 15

_CAMERA_NOT_OWNED = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="camera_not_owned_by_patient",
)
_CAMERA_NOT_LIVE = HTTPException(
    status_code=status.HTTP_409_CONFLICT,
    detail="camera_has_no_livekit_identity",
)

# Hold strong refs to in-flight recording tasks so the event loop can't
# GC them mid-run (fire-and-forget). Cleared on completion.
_bg_tasks: set[asyncio.Task] = set()


class RecordClipRequest(BaseModel):
    """Optional body. Defaults to a 15s clip when omitted."""

    duration_seconds: int = Field(default=_DEFAULT_DURATION, ge=1, le=600)


@router.post(
    "/patients/{patient_id}/cameras/{camera_id}/record-clip",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def record_clip(
    patient_id: UUID,
    camera_id: UUID,
    session: SessionDep,
    settings: SettingsDep,
    auth: CaregiverAuth,
    body: RecordClipRequest | None = None,
) -> None:
    """Start a server-side recording of one camera for the next N seconds.

    403 unless the caregiver owns the patient AND `camera_id` is one of
    that patient's paired cameras (same check as `talk/start`). 409 if
    the camera has never paired to LiveKit (no identity to subscribe to).
    """
    cameras = await _resolve_owned_cameras_or_403(
        session, caregiver_id=auth.id, patient_id=patient_id,
    )

    target = next((c for c in cameras if c.id == camera_id), None)
    if target is None:
        log.info(
            "record_clip reject reason=camera_not_owned caregiver=%s patient=%s camera=%s",
            auth.id, patient_id, camera_id,
        )
        raise _CAMERA_NOT_OWNED
    if not target.livekit_identity:
        raise _CAMERA_NOT_LIVE

    requested = body.duration_seconds if body else _DEFAULT_DURATION
    duration = max(_MIN_DURATION, min(requested, _MAX_DURATION))

    task = asyncio.create_task(
        clip_recorder.record_clip(
            settings=settings,
            room_name=f"patient_{patient_id}",
            camera_identity=target.livekit_identity,
            duration=duration,
        )
    )
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)

    log.info(
        "record_clip started server-side recording patient_id=%s camera_id=%s "
        "identity=%s duration=%ds -> %s",
        patient_id, camera_id, target.livekit_identity, duration, clip_recorder.output_dir(),
    )
