"""Camera pairing exchange endpoint.

The wall phone calls `POST /v1/cameras/pair` with a 6-digit code the
caregiver app minted. Server:
  1. Finds the matching `cameras` row whose pairing_code matches AND
     hasn't expired.
  2. Generates a 32-byte api_key, bcrypts it (at settings.bcrypt_rounds),
     stores the hash on the row.
  3. Clears the pairing fields (single-use).
  4. Returns the api_key + everything the phone needs to start
     publishing to LiveKit (room, identity, token endpoint).

Unauthenticated — the code IS the proof of authorization. Once
exchanged the code is gone; a second `pair` call with the same code
returns 401.
"""

import secrets
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import bcrypt
from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select

from src.api.deps import SessionDep, SettingsDep
from src.db.camera import Camera
from src.schemas.pairing import PairRequest, PairResponse

router = APIRouter(prefix="/v1", tags=["camera"])


@router.post(
    "/cameras/pair",
    response_model=PairResponse,
    status_code=status.HTTP_200_OK,
)
async def pair_camera(
    body: PairRequest,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> PairResponse:
    now = datetime.now(timezone.utc)

    stmt = (
        select(Camera)
        .where(Camera.pairing_code == body.pairing_code)
        .where(Camera.pairing_expires_at.is_not(None))
        .where(Camera.pairing_expires_at > now)
    )
    cam = (await session.execute(stmt)).scalar_one_or_none()
    if cam is None:
        # Opaque 401 — don't leak whether the code is unknown vs. expired.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_or_expired_pairing_code",
        )

    api_key = secrets.token_hex(32)  # 64 hex chars
    api_key_hash = bcrypt.hashpw(
        api_key.encode("utf-8"),
        bcrypt.gensalt(rounds=settings.bcrypt_rounds),
    ).decode("utf-8")

    cam.api_key_hash = api_key_hash
    cam.pairing_code = None
    cam.pairing_expires_at = None
    cam.last_seen_at = now
    await session.commit()

    backend_base_url = _backend_base_url(request)

    return PairResponse(
        api_key=api_key,
        camera_id=cam.id,
        patient_id=cam.patient_id,
        livekit_url=settings.livekit_url,
        livekit_identity=cam.livekit_identity or f"cam_{cam.id.hex[:8]}",
        livekit_room=cam.livekit_room or f"patient_{cam.patient_id}",
        backend_base_url=backend_base_url,
    )


def _backend_base_url(request: Request) -> str:
    """Derive the absolute URL the wall phone should use for follow-up
    calls (LiveKit token refresh, clip ingest). Strips path + query.
    """
    parts = urlsplit(str(request.url))
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
