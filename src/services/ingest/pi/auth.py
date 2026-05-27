"""Pi-camera ingest bearer-token mint + verify.

The token authenticates the Pi when it POSTs frames to
`/v1/cameras/me/frames`. Lifecycle:

  1. `mint_ingest_token()` at provision time produces
     `(plaintext, sha256_hex)`. Backend stores the hash on the
     `cameras` row and returns the plaintext to the Pi (via the next
     announce response, the only Pi-facing channel before the camera
     has any other credentials).
  2. Pi includes `Authorization: Bearer <plaintext>` on every frame-
     push connection. The push endpoint hashes the presented token
     and looks up the matching `cameras.ingest_token_hash`.
  3. To rotate, mint a new token and overwrite the hash. The Pi
     picks up the new value on its next announce response.

The plaintext is 32 random bytes URL-safe-base64 encoded (43 chars
without padding). Long enough to make brute-force lookup against the
hashed column infeasible without DB access. The hash is SHA-256 hex
(64 chars), unsalted: because the token is high-entropy random and
only one is ever issued per camera, salting buys nothing the entropy
doesn't already provide and would require a column-scan lookup
instead of an indexed equality match.

Constant-time comparison guards against timing attacks even though
the surface area is tiny (the only attacker channel is the push
endpoint itself, rate-limited at the transport layer).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

# Length of the raw plaintext token (URL-safe-base64 encoded).
# `token_urlsafe(n)` yields ~ceil(n * 4 / 3) chars; 32 bytes -> 43 chars.
_TOKEN_BYTES = 32

# Length of the SHA-256 hex digest stored in the DB.
INGEST_TOKEN_HASH_LENGTH = 64


def mint_ingest_token() -> tuple[str, str]:
    """Return (plaintext_token, sha256_hex_hash).

    The plaintext is returned to the Pi ONCE in plaintext and must
    not be persisted server-side (only the hash). Caller is
    responsible for writing the hash to `cameras.ingest_token_hash`
    and surfacing the plaintext to the Pi via the announce response.
    """
    plaintext = secrets.token_urlsafe(_TOKEN_BYTES)
    digest = hash_ingest_token(plaintext)
    return plaintext, digest


def hash_ingest_token(plaintext: str) -> str:
    """Hash a plaintext bearer token to its server-side digest form.

    Exported separately so the push endpoint can hash the presented
    token and compare against the stored hash without re-minting.
    """
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def verify_ingest_token(presented: str, stored_hash: str) -> bool:
    """Constant-time comparison of a presented plaintext token
    against a stored SHA-256 hex hash.

    Returns False if either value is empty or malformed; never
    raises. Use in the push endpoint's auth check.
    """
    if not presented or not stored_hash:
        return False
    if len(stored_hash) != INGEST_TOKEN_HASH_LENGTH:
        return False
    presented_hash = hash_ingest_token(presented)
    return hmac.compare_digest(presented_hash, stored_hash)
