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
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select

from src.api.deps import SessionDep, SettingsDep
from src.api.rate_limit import RateLimit, RateLimitPolicy
from src.db.camera import Camera
from src.schemas.pairing import PairRequest, PairResponse

router = APIRouter(prefix="/v1", tags=["camera"])

# Pairing codes are six digits (100000-999999). Unlimited guessing
# exhausts that space, and the demo code below never expires, so this
# endpoint is the single most brute-forceable thing in the API. 10 per
# minute per caller leaves a real pairing (one or two attempts, typed
# by hand from a screen) untouched while making enumeration useless.
PAIR_RATE_LIMIT = RateLimitPolicy(name="cameras_pair", limit=10, window_seconds=60)

# Stable identity marker for the App Review demo camera. The demo-pair
# path matches the camera by THIS, not by pairing_code — the caregiver
# app rotates an unpaired camera's pairing_code whenever its detail screen
# is opened, which would otherwise silently break the 424242 demo. The
# seed (scripts/seed/seed_demo_camera.py) stamps this identity on the demo row.
DEMO_CAMERA_IDENTITY = "cam_demo"


@router.post(
    "/cameras/pair",
    response_model=PairResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(RateLimit(PAIR_RATE_LIMIT))],
)
async def pair_camera(
    body: PairRequest,
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
) -> PairResponse:
    now = datetime.now(UTC)

    # App Review demo path: a configured, never-expiring, reusable code.
    # Lets an Apple reviewer pair a fresh NemaraCam install against the
    # seeded demo camera without a live 10-minute pairing window. We mint
    # a fresh api_key each time but deliberately leave the pairing fields
    # intact so the code stays reusable across review installs. Disabled
    # (and falls through to the normal path) whenever the setting is unset.
    demo = settings.demo_pairing_code
    if demo and body.pairing_code == demo:
        return await _pair_demo_camera(session, settings, request, now)

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


async def _pair_demo_camera(
    session: SessionDep,
    settings: SettingsDep,
    request: Request,
    now: datetime,
) -> PairResponse:
    """Pair the seeded demo camera, identified by its stable
    `DEMO_CAMERA_IDENTITY` (not by pairing_code, which the caregiver app
    rotates). Mirrors the normal pairing response so NemaraCam behaves
    identically after pairing. Each call re-mints the api_key (the
    previously-paired phone's key is invalidated, which is fine for review).
    """
    cam = (
        await session.execute(
            select(Camera)
            .where(Camera.livekit_identity == DEMO_CAMERA_IDENTITY)
            .limit(1)
        )
    ).scalar_one_or_none()
    if cam is None:
        # The setting is on but no camera carries the demo identity — the
        # demo camera wasn't seeded. Surface a distinct 404 so it's obvious
        # in logs (vs the opaque 401 the normal path returns). Run
        # scripts/seed/seed_demo_camera.py to provision it.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="demo_camera_not_provisioned",
        )

    api_key = secrets.token_hex(32)
    cam.api_key_hash = bcrypt.hashpw(
        api_key.encode("utf-8"),
        bcrypt.gensalt(rounds=settings.bcrypt_rounds),
    ).decode("utf-8")
    await session.commit()

    return PairResponse(
        api_key=api_key,
        camera_id=cam.id,
        patient_id=cam.patient_id,
        livekit_url=settings.livekit_url,
        livekit_identity=cam.livekit_identity or f"cam_{cam.id.hex[:8]}",
        livekit_room=cam.livekit_room or f"patient_{cam.patient_id}",
        backend_base_url=_backend_base_url(request),
    )


def _backend_base_url(request: Request) -> str:
    """Derive the absolute URL the wall phone should use for follow-up
    calls (LiveKit token refresh, clip ingest). Strips path + query.
    """
    parts = urlsplit(str(request.url))
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
