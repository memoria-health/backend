"""Caregiver Talk mode endpoints (Phase 3).

Talk is a server-driven toggle: the caregiver app POSTs to `/talk/start`
and `/talk/end`; this router figures out which wall phone in the
patient's room should play audio, then sends a LiveKit data packet to
the cameras with `{type: "set_talk_active", is_active: bool}`. The
cameras play caregiver audio only when their last received packet had
`is_active=true`.

We do NOT mute on the SFU side — every camera always subscribes to
the caregiver mic. Gating is camera-local (volume = 0 or 1). This
keeps latency low (no track resubscribe round-trip) and avoids
deadlocks if the camera misses a server-side mute.
"""

import json
import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from livekit import api as lkapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.camera import Camera
from src.db.patient import Patient
from src.schemas.talk import TalkStartResponse
from src.services.active_camera import get_active_camera

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["talk"])

_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="forbidden",
)
_NOT_FOUND = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="patient_not_found",
)
_UNCONFIGURED = HTTPException(
    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
    detail="livekit_not_configured",
)


@router.post(
    "/patients/{patient_id}/talk/start",
    response_model=TalkStartResponse,
    status_code=status.HTTP_200_OK,
)
async def start_talk(
    patient_id: UUID,
    session: SessionDep,
    settings: SettingsDep,
    auth: CaregiverAuth,
) -> TalkStartResponse:
    cameras = await _resolve_owned_cameras_or_403(
        session, caregiver_id=auth.id, patient_id=patient_id,
    )
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        log.error("talk.start: LIVEKIT_URL/KEY/SECRET not configured")
        raise _UNCONFIGURED

    active_camera_id = await get_active_camera(session, patient_id=patient_id)
    room = f"patient_{patient_id}"

    if active_camera_id is not None:
        # Route to the one identified camera; mute every other camera
        # explicitly so a previous Talk session can't leave a stale
        # is_active=true on a different phone.
        for cam in cameras:
            is_active = cam.id == active_camera_id
            await _send_talk_packet(
                settings=settings,
                room=room,
                camera=cam,
                is_active=is_active,
            )
        log.info(
            "talk.start routed patient_id=%s active_camera_id=%s",
            patient_id, active_camera_id,
        )
    else:
        # No active camera known — fan out so the caregiver is heard
        # somewhere immediately. Logged separately so we can tell
        # "Talk routed to active room" vs "Talk fanned out" in the
        # diagnostics tail.
        for cam in cameras:
            await _send_talk_packet(
                settings=settings, room=room, camera=cam, is_active=True,
            )
        log.info("talk.start fallback_fanout patient_id=%s", patient_id)

    return TalkStartResponse(active_camera_id=active_camera_id)


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
    cameras = await _resolve_owned_cameras_or_403(
        session, caregiver_id=auth.id, patient_id=patient_id,
    )
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        log.error("talk.end: LIVEKIT_URL/KEY/SECRET not configured")
        raise _UNCONFIGURED

    room = f"patient_{patient_id}"
    for cam in cameras:
        await _send_talk_packet(
            settings=settings, room=room, camera=cam, is_active=False,
        )
    log.info("talk.end closed patient_id=%s cameras=%d", patient_id, len(cameras))


# ── helpers ────────────────────────────────────────────────────


async def _resolve_owned_cameras_or_403(
    session: AsyncSession,
    *,
    caregiver_id: UUID,
    patient_id: UUID,
) -> list[Camera]:
    """403 unless the caregiver owns the patient. Returns every camera
    paired to the patient (including unpaired-placeholder rows are
    excluded — `livekit_identity IS NULL` means the phone never
    completed the pair handshake).
    """
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
    """Send a `{type: 'set_talk_active', is_active}` packet to one
    camera's LiveKit identity. Each call is a one-shot SDK round-trip;
    we don't pool connections at this scale.
    """
    if camera.livekit_identity is None:
        # Defensive — `_resolve_owned_cameras_or_403` already filters
        # these out, so this never fires in practice.
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
