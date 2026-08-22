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
  - `role='gateway'`   → Bearer must be a valid gateway site token, and
                         the camera named by `identity` must belong to
                         THAT gateway (2026-08-03).

The `gateway` role exists so the box never holds the LiveKit API
secret. It used to sign its own tokens with the raw key+secret in
`cameras.json`, which is how the PRODUCTION LiveKit credential ended up
in three places at once — the box, a committed ops doc, and a leaked
.env. With this path the secret lives only in the backend's env, so
rotating it is a single change instead of a coordinated fleet update.

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
from src.schemas.livekit_io import MintTokenRequest, MintTokenResponse
from src.services import livekit_rooms

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
    elif body.role == "gateway":
        # The box authenticates with its ONE stable site token — the same
        # credential it already uses for /v1/gateway/roster — and gets a
        # token scoped to a single camera it owns. Grants deliberately
        # mirror the `camera` role: the gateway publishes AS that camera,
        # so anything the camera app could do here, the box must too.
        cam = await _resolve_gateway_camera_or_401(session, credentials, body)
        grants = VideoGrants(
            room=body.room,
            room_join=True,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        identity = body.identity
    else:
        # caregiver-JWT path; reuses the same verifier as the rest of
        # the backend (handles HS256 local + ES256 prod).
        #
        # NOTE (2026-07-27): the patient-level privacy gate that once
        # zeroed subscribe/publish here is GONE — privacy is per-camera
        # now (cameras.privacy_enabled). Caregivers always get room-level
        # subscribe; per-TRACK enforcement is publisher-side track
        # subscription permissions (camera app / gateway restrict private
        # cameras' tracks to detector_worker_* identities), with the
        # server-side per-camera sever in livekit_privacy.py as
        # defense-in-depth. Room-level grants cannot express "this track
        # but not that one", so the token deliberately stays permissive.
        principal = await caregiver_auth(settings, credentials)
        await _ensure_caregiver_can_join(
            session, caregiver_id=principal.id, room=body.room
        )
        if body.role == "viewer":
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
        # SECURITY: force a `caregiver_`-prefixed identity. The live
        # per-camera privacy cut (`sever_caregiver_media`) unsubscribes
        # only `caregiver_*` participants, so a tampered/legacy client
        # that joined with a non-prefixed identity (e.g. "viewer-1")
        # would otherwise slip past the instant cut and keep live media
        # until publisher-side permissions land. Real clients already send
        # `caregiver_<id>_ios`/`_web`, so their device-distinct identity
        # (needed for simultaneous web+iOS viewing) is preserved as-is.
        identity = body.identity
        if not identity.startswith("caregiver_"):
            log.warning(
                "livekit.mint_token: caregiver identity %r lacks caregiver_ "
                "prefix — overriding to canonical caregiver_<id>", identity,
            )
            identity = f"caregiver_{principal.id}"

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


async def _resolve_gateway_camera_or_401(
    session: SessionDep,
    credentials: HTTPAuthorizationCredentials,
    body: MintTokenRequest,
) -> Camera:
    """Authenticate a gateway site token and resolve the ONE camera it is
    asking to publish as.

    Three things must all hold, and each is a privilege-escalation stop:
      1. the Bearer hashes to a row in `public.gateways`;
      2. `identity` names a camera whose `gateway_id` is THAT gateway —
         so one box can never mint a token for another box's camera;
      3. `room` is that camera's own `livekit_room`.

    Failures collapse to 401 without saying which of the three missed —
    a caller probing for "does camera X exist" learns nothing.

    NOTE: the same token->gateway lookup is currently inlined in
    `cameras_ingest.gateway_roster`/`_wait`/`_provision_camera`. Worth
    extracting to one shared helper; not done here because that means
    touching three live prod paths in the same change as this one.
    """
    from sqlalchemy import text

    from src.services.fleet_auth import hash_ingest_token

    if credentials.credentials is None or not credentials.credentials.strip():
        raise _INVALID
    digest = hash_ingest_token(credentials.credentials.strip())
    gw = (await session.execute(
        text("SELECT id FROM public.gateways WHERE token_hash = :d"),
        {"d": digest},
    )).first()
    if gw is None:
        raise _INVALID
    gateway_id = gw[0]

    cam = (await session.execute(
        select(Camera).where(Camera.livekit_identity == body.identity)
    )).scalar_one_or_none()
    if cam is None or cam.gateway_id is None or cam.gateway_id != gateway_id:
        raise _INVALID
    if cam.livekit_room != body.room:
        raise _INVALID
    return cam


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
) -> None:
    """Caregivers may only join a room that resolves to something of theirs.

    RESOLVED, not parsed. This used to take the uuid out of
    `patient_<uuid>` and check access to that patient, which was correct
    only while every room was a patient's. An account room
    (`account_<caregiver_id>`, 2026-08-22) parses just as cleanly and
    resolves to no patient at all, so the caregiver was refused their own
    streaming sensor — the room name had quietly become the thing
    granting access, rather than the thing being checked.

    `livekit_rooms.caregiver_may_join` answers both shapes by lookup, and
    returns the same False for unknown, malformed and unauthorized, so a
    caller probing room names learns nothing from the difference.
    """
    if not await livekit_rooms.caregiver_may_join(
        session, caregiver_id=caregiver_id, room=room,
    ):
        log.info(
            "livekit.mint_token reject reason=room_not_accessible "
            "caregiver=%s room=%s",
            caregiver_id, room,
        )
        raise _FORBIDDEN
