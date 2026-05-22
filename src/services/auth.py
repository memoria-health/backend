"""JWT verification for caregiver auth (Supabase realm).

Ported from `elephantLabs-wolf/src/services/auth.py`. We accept both
ES256 (production — Supabase 2.x default; public keys via JWKS) and
HS256 (legacy / local-dev with the shared `supabase_jwt_secret`). The
verifier reads the JWT's `alg` header and dispatches accordingly.

First request per process pays the JWKS fetch; subsequent requests hit
PyJWT's in-memory cache.
"""

from functools import lru_cache

import jwt as pyjwt

# `PyJWKClient` is exposed at `jwt.PyJWKClient` at runtime (since pyjwt
# 2.4) but the bundled types-pyjwt stubs are older and don't expose it.
# Import via the implementation submodule.
from jwt.jwks_client import PyJWKClient
from pydantic import ValidationError

from src.errors import JwtVerificationError
from src.schemas.auth import SupabasePrincipal

# Algorithms accepted across Supabase versions.
JWT_ALGORITHMS = ["ES256", "HS256"]

# Supabase sets `aud="authenticated"` for signed-in users; reject anything else.
EXPECTED_AUDIENCE = "authenticated"


@lru_cache(maxsize=8)
def _jwks_client(supabase_url: str) -> "PyJWKClient":
    """One JWKS client per Supabase URL. PyJWKClient maintains its own
    in-memory key cache; the `lru_cache` here just avoids construction
    cost on every request."""
    jwks_uri = supabase_url.rstrip("/") + "/auth/v1/.well-known/jwks.json"
    return PyJWKClient(jwks_uri)


def verify_supabase_jwt(
    token: str,
    *,
    supabase_url: str | None = None,
    jwt_secret: str | None = None,
) -> SupabasePrincipal:
    """Verify a Supabase-signed JWT and extract the principal.

    Dispatches on the JWT's `alg` header:
    - `ES256` → fetch the matching public key from JWKS at `supabase_url`.
    - `HS256` → verify with `jwt_secret`.

    Callers may pass both; whichever the token needs gets used. At
    least one must be set, and the one corresponding to the token's
    `alg` must be set or verification fails.

    Raises:
        JwtVerificationError: token is malformed, signed wrong, expired,
        has the wrong audience, or is missing required claims.
    """
    if supabase_url is None and jwt_secret is None:
        raise JwtVerificationError("must pass at least one of supabase_url or jwt_secret")

    try:
        # Inspect alg without verifying — pyjwt validates the alg again
        # below against `algorithms=JWT_ALGORITHMS`.
        alg = pyjwt.get_unverified_header(token).get("alg")
    except pyjwt.InvalidTokenError as exc:
        raise JwtVerificationError(f"invalid token header: {exc}") from exc

    try:
        if alg == "ES256":
            if supabase_url is None:
                raise JwtVerificationError("ES256 token but no supabase_url configured")
            signing_key: object = _jwks_client(supabase_url).get_signing_key_from_jwt(token).key
        elif alg == "HS256":
            if jwt_secret is None:
                raise JwtVerificationError("HS256 token but no jwt_secret configured")
            signing_key = jwt_secret
        else:
            raise JwtVerificationError(f"unsupported alg: {alg!r}")

        claims = pyjwt.decode(
            token,
            signing_key,  # type: ignore[arg-type]
            algorithms=JWT_ALGORITHMS,
            audience=EXPECTED_AUDIENCE,
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise JwtVerificationError("token expired") from exc
    except pyjwt.InvalidTokenError as exc:
        raise JwtVerificationError(f"invalid token: {exc}") from exc
    except pyjwt.PyJWKClientError as exc:  # type: ignore[attr-defined]
        raise JwtVerificationError(f"jwks fetch failed: {exc}") from exc

    sub = claims.get("sub")
    email = claims.get("email")
    if not isinstance(sub, str) or not sub:
        raise JwtVerificationError("missing or invalid 'sub' claim")
    if not isinstance(email, str) or not email:
        raise JwtVerificationError("missing or invalid 'email' claim")

    try:
        return SupabasePrincipal(id=sub, email=email.lower())
    except ValidationError as exc:
        raise JwtVerificationError(f"invalid claims: {exc}") from exc
