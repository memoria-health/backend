"""Centralized nemara-backend exception types.

All custom exceptions raised inside `src/` live here. Same shape as
wolf's `errors.py` — keeps `services/`, `repos/`, `api/` free of
ad-hoc exception classes.
"""


class NemaraError(Exception):
    """Base for all nemara-raised exceptions. Never raised directly."""


# ─── Auth ────────────────────────────────────────────────────────


class JwtVerificationError(NemaraError):
    """A JWT couldn't be verified or has invalid claims.

    Raised by `src.services.auth.verify_supabase_jwt`. Caller decides
    how to surface (401 at the API layer, log + reject elsewhere).
    """


# ─── Storage ─────────────────────────────────────────────────────


class StorageError(NemaraError):
    """A Supabase Storage operation failed.

    Raised by `src.services.storage` for both payload validation
    (empty / oversized / wrong content type) and provider-side
    failures (timeout, non-2xx, transport error). The API layer maps
    most of these to 4xx/5xx; the caller decides which.
    """


# ─── Auth admin ──────────────────────────────────────────────────


class AuthAdminError(NemaraError):
    """A Supabase Auth Admin (GoTrue) API call failed.

    Raised by `src.services.auth_admin` when deleting an auth user
    returns a non-2xx (and non-404) status, or the HTTP transport
    fails. The account-deletion endpoint maps this to a 5xx so the
    client can retry rather than believing the account is gone.
    """
