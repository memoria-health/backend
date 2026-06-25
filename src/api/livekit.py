"""LiveKit token minting.

Dual auth: a Bearer is either a camera api-key OR a caregiver
Supabase JWT. The body's `role` decides which path is checked:
  - `role='camera'`    → Bearer must be a valid camera api-key,
                         and `room`/`identity` must match the camera's
                         own `livekit_room`/`livekit_identity`.
  - `role='viewer'`    → Bearer must be a valid caregiver JWT, and
                         the room must belong to one of their patients.
  - `role='broadcaster'` → same as viewer, plus canPublish=true for
                         audio (used by Phase 3 Talk mode).

We hand-roll the dual-auth dispatch rather than reusing the existing
`CameraAuth`/`CaregiverAuth` deps because FastAPI's dep system
short-circuits the first failure; here we WANT to try both.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from livekit.api import AccessToken, VideoGrants
from sqlalchemy import select

from src.api.deps import SessionDep, SettingsDep, camera_auth, caregiver_auth
from src.db.camera import Camera
from src.db.patient import Patient
from src.schemas.livekit_io import MintTokenRequest, MintTokenResponse

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["livekit"])

_bearer_scheme = HTTPBearer(auto_error=False)

_INVALID = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="invalid_auth",
    headers={"WWW-Authenticate": "Bearer"},
)
_FORBIDDEN = HTTPException(
    status_code=status.HTTP_403_FORBIDDEN,
    detail="forbidden",
)

# Default token TTL — 1 hour, same as the dashboard. Camera-side
# refresh logic kicks in at TTL/2 (Phase 4).
_TTL_SECONDS = 3600


@router.post(
    "/livekit/token",
    response_model=MintTokenResponse,
    status_code=status.HTTP_200_OK,
)
async def mint_livekit_token(
    body: MintTokenRequest,
    session: SessionDep,
    settings: SettingsDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(_bearer_scheme),
    ],
) -> MintTokenResponse:
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        log.error("livekit.mint_token: LIVEKIT_URL/KEY/SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="livekit_not_configured",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise _INVALID

    if body.role == "camera":
        cam = await _resolve_camera_or_401(session, credentials)
        await _ensure_camera_room_match(cam, body)
        grants = VideoGrants(
            room=body.room,
            room_join=True,
            can_publish=True,
            can_subscribe=True,           # camera subscribes to caregiver Talk audio (Phase 3)
            can_publish_data=True,
        )
        identity = body.identity
    else:
        # caregiver-JWT path; reuses the same verifier as the rest of
        # the backend (handles HS256 local + ES256 prod).
        principal = await caregiver_auth(settings, credentials)
        privacy = await _ensure_caregiver_can_join(
            session, caregiver_id=principal.id, room=body.room
        )
        if privacy:
            # PRIVACY MODE: the caregiver may still JOIN the room (so the
            # app stays connected for insights + UI), but is granted
            # neither subscribe (no live video/audio in) nor publish (no
            # Talk audio out). Baking this into the JWT is the durable
            # guarantee: the iOS/web SDKs reuse their token verbatim on
            # auto-reconnect, so a severed caregiver can't wash the block
            # out by reconnecting. The detector bot mints its OWN token
            # independently and is unaffected — analysis keeps running.
            grants = VideoGrants(
                room=body.room,
                room_join=True,
                can_publish=False,
                can_subscribe=False,
                can_publish_data=False,
            )
        elif body.role == "viewer":
            grants = VideoGrants(
                room=body.room,
                room_join=True,
                can_publish=False,
                can_subscribe=True,
                can_publish_data=False,
            )
        else:  # broadcaster — caregiver Talk mode
            grants = VideoGrants(
                room=body.room,
                room_join=True,
                can_publish=True,      # mic only, in practice
                can_subscribe=True,
                can_publish_data=True,
            )
        identity = body.identity

    # `with_ttl` takes a `datetime.timedelta` — passing an int silently
    # corrupts the token (datetime + int raises TypeError inside to_jwt).
    at = (
        AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_ttl(timedelta(seconds=_TTL_SECONDS))
    )
    jwt = at.to_jwt()
    expires_at = (
        datetime.now(timezone.utc) + timedelta(seconds=_TTL_SECONDS)
    ).isoformat()
    return MintTokenResponse(
        token=jwt,
        url=settings.livekit_url,
        expires_at=expires_at,
    )


async def _resolve_camera_or_401(
    session: SessionDep,
    credentials: HTTPAuthorizationCredentials,
) -> Camera:
    """Like `camera_auth` but local to this router (avoids the
    short-circuit of trying caregiver-JWT first)."""
    try:
        return await camera_auth(session, credentials)
    except HTTPException as exc:
        # Preserve the 404 `camera_not_found` so a DELETED camera's
        # token-mint is distinguishable from a transient auth failure —
        # the wall phone treats 404 as "I've been unpaired" (see
        # DesiredStateMonitor). Every other auth failure collapses to the
        # router's 401.
        if exc.status_code == status.HTTP_404_NOT_FOUND:
            raise
        raise _INVALID from None


async def _ensure_camera_room_match(cam: Camera, body: MintTokenRequest) -> None:
    """A camera token may only be minted for the camera's own room +
    identity. Anything else is a privilege-escalation attempt.
    """
    if body.room != cam.livekit_room:
        log.info(
            "livekit.mint_token reject reason=camera_room_mismatch "
            "cam_room=%s requested=%s",
            cam.livekit_room, body.room,
        )
        raise _FORBIDDEN
    if cam.livekit_identity and body.identity != cam.livekit_identity:
        log.info(
            "livekit.mint_token reject reason=camera_identity_mismatch "
            "cam_identity=%s requested=%s",
            cam.livekit_identity, body.identity,
        )
        raise _FORBIDDEN


async def _ensure_caregiver_can_join(
    session: SessionDep,
    *,
    caregiver_id,
    room: str,
) -> bool:
    """Caregivers may only join rooms named `patient_<id>` for patients
    they own. The room name is the load-bearing identifier.

    Returns the patient's ``privacy_mode_enabled`` flag so the caller can
    gate the token's subscribe/publish grants in the same DB round-trip.
    """
    if not room.startswith("patient_"):
        raise _FORBIDDEN
    try:
        patient_id = UUID(room.removeprefix("patient_"))
    except ValueError:
        raise _FORBIDDEN from None

    stmt = select(Patient.privacy_mode_enabled).where(
        Patient.id == patient_id, Patient.caregiver_id == caregiver_id,
    )
    privacy = (await session.execute(stmt)).scalar_one_or_none()
    if privacy is None:
        log.info(
            "livekit.mint_token reject reason=patient_not_owned "
            "caregiver=%s patient=%s",
            caregiver_id, patient_id,
        )
        raise _FORBIDDEN
    return bool(privacy)
