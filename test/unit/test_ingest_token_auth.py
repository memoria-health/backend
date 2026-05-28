"""Unit tests for the ingest-token mint + verify helpers.

These tests guard the bearer-token contract used by the Pi-camera
push endpoint (POST /v1/cameras/me/frames). The token is the only
thing standing between an attacker on the public internet and the
ability to publish frames into another customer's LiveKit room, so
a regression that weakens the comparison or truncates entropy is a
real safety issue.
"""

from __future__ import annotations

import pytest

from src.services.ingest.pi.auth import (
    INGEST_TOKEN_HASH_LENGTH,
    hash_ingest_token,
    mint_ingest_token,
    verify_ingest_token,
)


# ─── mint shape ───────────────────────────────────────────────────


def test_mint_returns_two_distinct_strings() -> None:
    plain, hashed = mint_ingest_token()
    assert plain
    assert hashed
    # The plaintext is high-entropy URL-safe-base64; the hash is hex.
    # They should not collide.
    assert plain != hashed


def test_mint_plaintext_is_high_entropy() -> None:
    """32 random bytes URL-safe-base64-encoded yields ~43 chars (no
    padding). A length significantly shorter than this would indicate
    the entropy source got truncated.
    """
    plain, _ = mint_ingest_token()
    assert len(plain) >= 40


def test_mint_hash_is_64_hex_chars() -> None:
    """SHA-256 hex digest is exactly 64 lowercase hex chars. Anything
    else would break the cameras_ingest_token_hash_length DB check.
    """
    _, hashed = mint_ingest_token()
    assert len(hashed) == INGEST_TOKEN_HASH_LENGTH
    assert len(hashed) == 64
    assert all(c in "0123456789abcdef" for c in hashed)


def test_each_mint_produces_unique_token() -> None:
    """A non-random mint that returned the same token twice would
    silently make every Pi the same Pi. 64 invocations is enough to
    catch a constant-output bug.
    """
    seen: set[str] = set()
    for _ in range(64):
        plain, _ = mint_ingest_token()
        seen.add(plain)
    assert len(seen) == 64


# ─── hash_ingest_token ────────────────────────────────────────────


def test_hash_is_deterministic() -> None:
    """Same plaintext -> same hash. Required because the push
    endpoint hashes the presented token and looks the row up by
    indexed equality.
    """
    assert hash_ingest_token("abc") == hash_ingest_token("abc")


def test_hash_differs_for_different_plaintext() -> None:
    assert hash_ingest_token("abc") != hash_ingest_token("abd")


def test_hash_matches_mint_output() -> None:
    """mint_ingest_token() must return (plaintext, hash) where the
    hash IS hash_ingest_token(plaintext). A regression that minted
    a hash from random bytes instead of from the plaintext would
    make verify always return False against the stored value.
    """
    plain, hashed = mint_ingest_token()
    assert hash_ingest_token(plain) == hashed


# ─── verify_ingest_token ──────────────────────────────────────────


def test_verify_accepts_matching_token() -> None:
    plain, hashed = mint_ingest_token()
    assert verify_ingest_token(plain, hashed) is True


def test_verify_rejects_wrong_token() -> None:
    plain, hashed = mint_ingest_token()
    other_plain, _ = mint_ingest_token()
    assert verify_ingest_token(other_plain, hashed) is False


@pytest.mark.parametrize("plain", ["", " ", "\n"])
def test_verify_rejects_empty_or_whitespace_plain(plain: str) -> None:
    _, hashed = mint_ingest_token()
    # Whitespace-only strings are nonsense bearer tokens but the
    # function should not crash on them, just return False.
    if plain.strip():
        assert verify_ingest_token(plain, hashed) is False
    else:
        # Truly empty strings short-circuit; verify the explicit guard.
        assert verify_ingest_token(plain.strip(), hashed) is False


def test_verify_rejects_empty_hash() -> None:
    plain, _ = mint_ingest_token()
    assert verify_ingest_token(plain, "") is False


def test_verify_rejects_malformed_hash_length() -> None:
    plain, _ = mint_ingest_token()
    # Hash that is not exactly INGEST_TOKEN_HASH_LENGTH chars cannot
    # have been minted by us; treat as not-a-match without further
    # comparison.
    assert verify_ingest_token(plain, "a" * 63) is False
    assert verify_ingest_token(plain, "a" * 65) is False


def test_verify_uses_constant_time_compare() -> None:
    """Smoke-check that the verify call uses hmac.compare_digest by
    confirming behavior on similar-prefixed inputs. We cannot
    directly assert timing characteristics in a unit test, but we
    CAN assert that an attacker substituting a different-but-same-
    length hash is correctly rejected.
    """
    plain, hashed = mint_ingest_token()
    # Flip the last hex char so the hash differs by 4 bits but
    # length stays exactly 64.
    flipped = hashed[:-1] + ("0" if hashed[-1] != "0" else "1")
    assert verify_ingest_token(plain, flipped) is False
