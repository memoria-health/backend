"""Shared FastAPI dependencies.

Composition root for things every protected endpoint needs:
- `SettingsDep` — frozen Settings injection.
- `CaregiverAuth` — verified Supabase JWT principal. Endpoints declare
  `auth: CaregiverAuth` to require a signed-in caregiver; the dep
  rejects with 401 if anything is wrong with the bearer token.
- `SessionDep` — one async SQLAlchemy session per request.

All `JwtVerificationError`s are flattened to the same opaque
`{"detail": "invalid_auth"}` 401 so iOS can't distinguish missing-token
from expired-token from wrong-signature (security best practice). The
real reason is logged for our own diagnostics.
"""

import asyncio
import logging
from typing import Annotated

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.session import get_session
from src.errors import JwtVerificationError
from src.schemas.auth import SupabasePrincipal
from src.services.auth import verify_supabase_jwt
from src.settings import Settings, get_settings

log = logging.getLogger(__name__)

# `auto_error=False` so we return our own 401 (with WWW-Authenticate header)
# instead of FastAPI's default 403 Forbidden.
_bearer_scheme = HTTPBearer(auto_error=False)

_INVALID_AUTH_RESPONSE = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="invalid_auth",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_settings_dep() -> Settings:
    """FastAPI-compatible Settings injector."""
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def caregiver_auth(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> SupabasePrincipal:
    """Verify the bearer token and return the SupabasePrincipal.

    Raises HTTPException(401) on any failure — missing header, wrong
    scheme, malformed/expired/invalid-signature token, or missing claims.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        log.info("auth.reject reason=missing_or_malformed_bearer")
        raise _INVALID_AUTH_RESPONSE

    try:
        principal = verify_supabase_jwt(
            credentials.credentials,
            supabase_url=settings.supabase_url,
            jwt_secret=settings.supabase_jwt_secret,
        )
    except JwtVerificationError as exc:
        log.info("auth.reject reason=%s", exc)
        raise _INVALID_AUTH_RESPONSE from exc

    log.debug("auth.accept caregiver_id=%s", principal.id)
    return principal


CaregiverAuth = Annotated[SupabasePrincipal, Depends(caregiver_auth)]


# ── Camera api-key auth (Phase 0+) ────────────────────────────────
# The wall phone gets a long-lived api_key from the pairing flow and
# uses it as a Bearer token on every subsequent request. The server
# stores only its bcrypt hash on `cameras.api_key_hash`.
#
# Bcrypt verification is O(2^rounds); a naive `select all then verify`
# would scan every camera row per request. We trade some PII-leak risk
# for performance by also storing a short prefix of the api_key
# unhashed (the first 8 chars) on the cameras row as `api_key_prefix`
# in a future revision; for now N is small (a few cameras per
# patient × a few patients per backend) so the linear scan is fine.
#
# Returning the Camera ORM lets the route handler use cam.patient_id
# without an extra lookup.

async def camera_auth(
    session: SessionDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> Camera:
    """Verify the bearer token against `cameras.api_key_hash` and
    return the matching Camera row. Raises 401 on any failure.

    Phase 0 implementation: linear-scan all rows with a non-null
    `api_key_hash`. Phase 4+ may add a prefix index for O(1) lookup.
    """
    if credentials is None or credentials.scheme.lower() != "bearer":
        log.info("camera_auth.reject reason=missing_or_malformed_bearer")
        raise _INVALID_AUTH_RESPONSE

    token = credentials.credentials.encode("utf-8")

    stmt = select(Camera).where(Camera.api_key_hash.is_not(None))
    rows = (await session.execute(stmt)).scalars().all()
    loop = asyncio.get_running_loop()
    for cam in rows:
        if cam.api_key_hash is None:
            continue
        hash_bytes = cam.api_key_hash.encode("utf-8")
        # bcrypt.checkpw is CPU-bound (intentionally — the whole point
        # is to be slow). Offload to the default executor so we don't
        # block the asyncio event loop during the linear scan.
        try:
            matched = await loop.run_in_executor(
                None, bcrypt.checkpw, token, hash_bytes,
            )
        except ValueError:
            # malformed hash on disk — skip and keep scanning
            continue
        if matched:
            log.debug("camera_auth.accept camera_id=%s", cam.id)
            return cam

    log.info("camera_auth.reject reason=no_matching_camera")
    raise _INVALID_AUTH_RESPONSE


CameraAuth = Annotated[Camera, Depends(camera_auth)]
