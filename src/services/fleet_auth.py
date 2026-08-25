"""Bearer credentials for the services that talk to this backend.

Camera bridges, gateway boxes, and anything else in the fleet. A service
holds a plaintext token; the backend stores only its sha256 and looks it
up on every poll, so a leaked database does not leak service credentials.

Lived in `services/ingest/pi/` until 2026-08-21, which was doubly wrong:
there is no Raspberry Pi in the product, and this is not ingest — it is
the credential system every fleet endpoint depends on (`/gateway/roster`,
`/gateway/enroll`, `/fleet/poll`, `/livekit/token`). The name mattered
because "delete the pi code, it is unused" is a reasonable-sounding
instruction that would have taken camera auth down with it.
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
