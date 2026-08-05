"""Shared FastAPI dependencies.

Composition root for things every protected endpoint needs:
- `SettingsDep` — frozen Settings injection.
- `CaregiverAuth` — verified Supabase JWT principal. Endpoints declare
  `auth: CaregiverAuth` to require a signed-in caregiver; the dep
  rejects with 401 if anything is wrong with the bearer token.
- `OperatorAuth` — static bearer token for OUR OWN cross-tenant ops
  endpoints. A separate credential from `CaregiverAuth` on purpose:
  customer tokens must never satisfy an operator surface.
- `SessionDep` — one async SQLAlchemy session per request.

All `JwtVerificationError`s are flattened to the same opaque
`{"detail": "invalid_auth"}` 401 so iOS can't distinguish missing-token
from expired-token from wrong-signature (security best practice). The
real reason is logged for our own diagnostics.
"""

import asyncio
import logging
import secrets
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

# Distinct from 401: the bearer was well-formed but no `cameras` row
# has a matching `api_key_hash`. Almost always means the camera was
# deleted server-side (caregiver removed it, or DB was reset). The
# wall phone uses this code to decide it's actually been unpaired,
# vs a transient 401 (which it retries through). See
# DesiredStateMonitor.swift for the consumer.
_NO_MATCHING_CAMERA_RESPONSE = HTTPException(
    status_code=status.HTTP_404_NOT_FOUND,
    detail="camera_not_found",
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


async def operator_auth(
    settings: SettingsDep,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer_scheme)],
) -> None:
    """Gate operator-only endpoints on a static bearer token.

    Deliberately NOT `CaregiverAuth`. An operator surface reads across
    every household in the product — narratives, rooms, clips — so it
    must not be reachable by a customer token at all. Accepting a
    Supabase JWT and then checking an email allow-list would make the
    boundary a string comparison inside a token issued to customers;
    this makes it a separate credential that no customer ever holds.

    Fails closed when unset: with no token configured the endpoint is
    unreachable rather than open. That matters because the natural
    mistake — deploying without `NEMARA_OPS_API_TOKEN` — would
    otherwise publish every household's events to anyone.

    The token is compared with `compare_digest` to keep the check
    constant-time, and never logged.
    """
    expected = settings.ops_api_token
    if not expected:
        log.warning("auth.operator_reject reason=ops_api_token_unset")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ops_api_not_configured",
        )
    if credentials is None or credentials.scheme.lower() != "bearer":
        log.info("auth.operator_reject reason=missing_or_malformed_bearer")
        raise _INVALID_AUTH_RESPONSE
    if not secrets.compare_digest(credentials.credentials, expected):
        log.warning("auth.operator_reject reason=bad_token")
        raise _INVALID_AUTH_RESPONSE


# No principal to return — the token identifies "an operator", not a
# person. If operators ever become a team, this becomes a lookup that
# returns who, and the endpoints keep their signature.
OperatorAuth = Annotated[None, Depends(operator_auth)]


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
    return the matching Camera row.

    Two distinct failure modes, with different status codes so the
    wall phone (NemaraCam) can distinguish "auth blip, retry" from
    "I've actually been deleted, reset to pairing":

      - 401 invalid_auth: missing/malformed bearer header. The phone
        treats this as transient and keeps polling.
      - 404 camera_not_found: bearer well-formed but no row has a
        matching api_key_hash. The phone treats this as definitive
        "row was deleted" and clears local credentials.

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
    raise _NO_MATCHING_CAMERA_RESPONSE


CameraAuth = Annotated[Camera, Depends(camera_auth)]
