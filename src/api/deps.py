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

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

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
