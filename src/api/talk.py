"""Caregiver Talk mode endpoints (Phase 3).

Talk in the iOS Live tab is a phone-call metaphor: the caregiver opens
a patient, swipes through a carousel of camera tiles, and taps a
call button that opens full-duplex audio with **the camera the
caregiver explicitly picked**. The server's only routing job is to
flip `is_active` on the chosen camera via a LiveKit data packet and
to mute every sibling — there is no "best guess" branch.

The active-camera *suggestion* (used by the iOS app to default the
carousel to the right tile on Live-tab open) lives at
`GET /v1/patients/{id}/active-camera`. That's a hint, not a routing
decision.

Three endpoints in this router:
  - GET  /v1/patients/{id}/active-camera   — suggestion for the carousel
  - POST /v1/patients/{id}/talk/start      — opens the call to one camera
  - POST /v1/patients/{id}/talk/end        — closes any open call

We do NOT mute on the SFU side — every camera always subscribes to
the caregiver's mic. Gating is camera-local (RemoteTrackPublication
enabled/disabled), driven by `{type: "set_talk_active", is_active}`
data packets. This keeps routing flips sub-millisecond and avoids
deadlocks if a camera misses a server-side mute.
"""

import json
import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from livekit import api as lkapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.camera import Camera
from src.db.patient import Patient
from src.schemas.talk import ActiveCameraResponse, TalkStartRequest
from src.services.active_camera import get_active_camera

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["talk"])

_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="forbidden",
)
_CAMERA_NOT_OWNED = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="camera_not_owned_by_patient",
)
_UNCONFIGURED = HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail="livekit_not_configured",
)


@router.get(
    "/patients/{patient_id}/active-camera",
    response_model=ActiveCameraResponse,
    status_code=status.HTTP_200_OK,
)
async def get_active_camera_suggestion(
    patient_id: UUID,
    session: SessionDep,
    auth: CaregiverAuth,
) -> ActiveCameraResponse:
    """Best-guess camera the patient is currently in front of. The iOS
    Live tab uses this to default the carousel to the right tile;
    None means 'no qualifying analyses row in the last 30s — fall
    back to the first online camera in the patient's list'.
    """
    await _ensure_patient_owned(session, caregiver_id=auth.id, patient_id=patient_id)
    camera_id = await get_active_camera(session, patient_id=patient_id)
    return ActiveCameraResponse(camera_id=camera_id)


@router.post(
    "/patients/{patient_id}/talk/start",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def start_talk(
    patient_id: UUID,
    body: TalkStartRequest,
    session: SessionDep,
    settings: SettingsDep,
    auth: CaregiverAuth,
) -> None:
    """Open (or transfer) a call to one specific camera. Re-call with a
    different `camera_id` to transfer the active call — backend always
    re-broadcasts `is_active` to every camera so the previous target's
    gate closes atomically with the new target's opening.
    """
    cameras = await _resolve_owned_cameras_or_403(
        session, caregiver_id=auth.id, patient_id=patient_id,
    )
    _ensure_livekit_configured(settings)

    target_ids = {c.id for c in cameras}
    if body.camera_id not in target_ids:
        log.info(
            "talk.start reject reason=camera_not_owned caregiver=%s patient=%s camera=%s",
            auth.id, patient_id, body.camera_id,
        )
        raise _CAMERA_NOT_OWNED

    room = f"patient_{patient_id}"
    for cam in cameras:
        await _send_talk_packet(
            settings=settings,
            room=room,
            camera=cam,
            is_active=(cam.id == body.camera_id),
        )
    log.info(
        "talk.start routed patient_id=%s camera_id=%s cameras_notified=%d",
        patient_id, body.camera_id, len(cameras),
    )


@router.post(
    "/patients/{patient_id}/talk/end",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def end_talk(
    patient_id: UUID,
    session: SessionDep,
    settings: SettingsDep,
    auth: CaregiverAuth,
) -> None:
    """Close the call. Idempotent — safe to call when no call is open."""
    cameras = await _resolve_owned_cameras_or_403(
        session, caregiver_id=auth.id, patient_id=patient_id,
    )
    _ensure_livekit_configured(settings)

    room = f"patient_{patient_id}"
    for cam in cameras:
        await _send_talk_packet(
            settings=settings, room=room, camera=cam, is_active=False,
        )
    log.info("talk.end closed patient_id=%s cameras=%d", patient_id, len(cameras))


# ── helpers ────────────────────────────────────────────────────


def _ensure_livekit_configured(settings) -> None:
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        log.error("talk: LIVEKIT_URL/KEY/SECRET not configured")
        raise _UNCONFIGURED


async def _ensure_patient_owned(
    session: AsyncSession,
    *,
    caregiver_id: UUID,
    patient_id: UUID,
) -> None:
    owns = await session.execute(
        select(Patient.id).where(
            Patient.id == patient_id, Patient.caregiver_id == caregiver_id,
        )
    )
    if owns.scalar_one_or_none() is None:
        log.info(
            "talk reject reason=patient_not_owned caregiver=%s patient=%s",
            caregiver_id, patient_id,
        )
        raise _FORBIDDEN


async def _resolve_owned_cameras_or_403(
    session: AsyncSession,
    *,
    caregiver_id: UUID,
    patient_id: UUID,
) -> list[Camera]:
    """403 unless the caregiver owns the patient. Returns every camera
    paired to the patient. Unpaired-placeholder rows
    (`livekit_identity IS NULL`) are excluded since they have no
    LiveKit identity to address.
    """
    await _ensure_patient_owned(session, caregiver_id=caregiver_id, patient_id=patient_id)
    cams = await session.execute(
        select(Camera).where(
            Camera.patient_id == patient_id,
            Camera.livekit_identity.is_not(None),
        )
    )
    return list(cams.scalars().all())


async def _send_talk_packet(
    *,
    settings,
    room: str,
    camera: Camera,
    is_active: bool,
) -> None:
    """Send a `{type: "set_talk_active", is_active}` packet to one
    camera's LiveKit identity. Each call opens + closes a fresh
    LiveKitAPI client — at v0 scale (handful of cameras per patient,
    a few calls per day) connection pooling isn't worth the
    complexity.
    """
    if camera.livekit_identity is None:
        # `_resolve_owned_cameras_or_403` filters these out; the guard
        # is here so future callers can't accidentally crash on it.
        return
    payload = json.dumps({"type": "set_talk_active", "is_active": is_active}).encode("utf-8")
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
                topic="set_talk_active",
            )
        )
    finally:
        await lk.aclose()
