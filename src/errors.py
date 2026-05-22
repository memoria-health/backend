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
