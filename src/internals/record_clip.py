"""Experimental: caregiver-triggered 15s clip capture (data collection).

NON-PRODUCTION. Mounted only when `settings.enable_internals` is true
(see src/main.py). Lets a caregiver press a button in the iOS app to
make a specific wall camera record the next N seconds locally and ship
the MP4 to a laptop receiver — deterministic training data for the
detectors.

The camera side is already built and dormant: the wall phone decodes a
`{"type": "record_trigger", "duration_seconds": N}` LiveKit data packet
on the `record_trigger` topic, records the next N s to a local MP4, and
POSTs it to its configured "Experiment Server URL" (the receiver in
clip_receiver.py). This endpoint's only job is to emit that packet to
one camera's LiveKit identity.

Auth + ownership reuse the Talk-mode helpers verbatim so the security
posture is identical: a caregiver can only address cameras belonging to
a patient they own.
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from livekit import api as lkapi
from pydantic import BaseModel, Field

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.api.talk import _ensure_livekit_configured, _resolve_owned_cameras_or_403
from src.db.camera import Camera

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["internals"])

# Mirrors the camera-side clamp at ContentView.swift:372.
_MIN_DURATION = 5
_MAX_DURATION = 300
_DEFAULT_DURATION = 15

_CAMERA_NOT_OWNED = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="camera_not_owned_by_patient",
)


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
    """Tell one camera to record the next N seconds and upload it.

    403 unless the caregiver owns the patient AND `camera_id` is one of
    that patient's paired cameras (same check as `talk/start`). The
    packet is sent to a single target identity, not broadcast.
    """
    cameras = await _resolve_owned_cameras_or_403(
        session, caregiver_id=auth.id, patient_id=patient_id,
    )
    _ensure_livekit_configured(settings)

    target = next((c for c in cameras if c.id == camera_id), None)
    if target is None:
        log.info(
            "record_clip reject reason=camera_not_owned caregiver=%s patient=%s camera=%s",
            auth.id, patient_id, camera_id,
        )
        raise _CAMERA_NOT_OWNED

    requested = body.duration_seconds if body else _DEFAULT_DURATION
    duration = max(_MIN_DURATION, min(requested, _MAX_DURATION))

    await _send_record_packet(
        settings=settings,
        room=f"patient_{patient_id}",
        camera=target,
        duration_seconds=duration,
    )
    log.info(
        "record_clip routed patient_id=%s camera_id=%s identity=%s duration=%ds",
        patient_id, camera_id, target.livekit_identity, duration,
    )


async def _send_record_packet(
    *,
    settings,
    room: str,
    camera: Camera,
    duration_seconds: int,
) -> None:
    """Send a `{type: "record_trigger", duration_seconds}` packet to one
    camera's LiveKit identity. Mirrors `talk._send_talk_packet`: a fresh
    LiveKitAPI client per call — at v0 scale that's fine.
    """
    if camera.livekit_identity is None:
        # `_resolve_owned_cameras_or_403` already excludes these.
        return
    payload = json.dumps(
        {"type": "record_trigger", "duration_seconds": duration_seconds}
    ).encode("utf-8")
    lk = lkapi.LiveKitAPI(
        settings.livekit_url,
        settings.livekit_api_key,
        settings.livekit_api_secret,
    )
    try:
        await lk.room.send_data(
            lkapi.SendDataRequest(
                room=room,
                data=payload,
                kind=lkapi.DataPacket.Kind.RELIABLE,
                destination_identities=[camera.livekit_identity],
                topic="record_trigger",
            )
        )
    finally:
        await lk.aclose()
