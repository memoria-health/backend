"""Guards for the production attack surface.

Every assertion here corresponds to a finding in
`security/SECURITY-AUDIT-2026-08-03-CROSS-SYSTEM.md`. They are unit
tests on purpose: they build the app and inspect routes/middleware
without touching the database, so they run in CI on a laptop with no
Supabase and catch a regression the moment someone re-mounts a
removed router or flips a docs flag.

The pattern for "off in production" mirrors `_require_internal_key`
in src/api/internal/cameras.py: local and test stay convenient, every
other env fails closed.
"""

from __future__ import annotations

import pytest

from src.main import create_app
from src.settings import get_settings


def _app_for_env(monkeypatch: pytest.MonkeyPatch, env: str):
    """Build an app as if it were running under NEMARA_ENV=<env>.

    Real env vars beat the `.env` file in pydantic-settings and
    `get_settings()` builds a fresh Settings per call (see conftest),
    so setting the var is enough — but `get_settings` is lru_cached in
    some call paths, so clear it to be certain the app sees `env`.
    """
    monkeypatch.setenv("NEMARA_ENV", env)
    cache_clear = getattr(get_settings, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    return create_app()


def _paths(app) -> set[str]:
    return {getattr(r, "path", None) for r in app.routes}


# ── Pi discovery: deleted, not merely disabled ──────────────────────


@pytest.mark.parametrize("env", ["local", "test", "staging", "prod"])
def test_pi_announce_route_does_not_exist(monkeypatch, env):
    """The unauthenticated Pi announce endpoint is gone in EVERY env.

    It accepted anonymous writes keyed on a caller-supplied
    hardware_id and handed back a plaintext ingest bearer token, so an
    attacker who won the race against a real Pi could push frames as
    that camera. The whole Pi path is now deleted — endpoints, tables,
    columns and the `source_type` value — so this guards against it
    being reintroduced rather than against it still being live.
    Parametrised across envs because an env-gated version of this route
    would still be one config mistake away from serving.
    """
    app = _app_for_env(monkeypatch, env)
    paths = _paths(app)
    assert "/v1/cameras/pi-announce" not in paths
    assert "/v1/cameras/discovered" not in paths


# ── M2: OpenAPI docs must not be public in production ───────────────


@pytest.mark.parametrize("env", ["staging", "prod"])
def test_api_docs_disabled_outside_local_and_test(monkeypatch, env):
    """Production served /openapi.json unauthenticated — 53 paths and
    68 operations, the complete API surface handed to an attacker.
    """
    app = _app_for_env(monkeypatch, env)
    assert app.docs_url is None
    assert app.redoc_url is None
    assert app.openapi_url is None
    paths = _paths(app)
    assert "/docs" not in paths
    assert "/openapi.json" not in paths


@pytest.mark.parametrize("env", ["local", "test"])
def test_api_docs_available_locally(monkeypatch, env):
    """Developers keep /docs on a laptop; that convenience is the whole
    reason the flag is env-scoped rather than removed."""
    app = _app_for_env(monkeypatch, env)
    assert app.docs_url == "/docs"
    assert app.openapi_url == "/openapi.json"


# ── Network layer: security headers ─────────────────────────────────


def test_security_headers_present_on_responses(monkeypatch):
    """Production answered with no HSTS, CSP, X-Content-Type-Options or
    X-Frame-Options. /health is unauthenticated and needs no DB, so it
    is the right probe for a header-middleware test.
    """
    app = _app_for_env(monkeypatch, "prod")
    from src.api.security_headers import SECURITY_HEADERS

    # The middleware is registered on the app...
    assert any(
        "SecurityHeaders" in str(m.cls) for m in app.user_middleware
    ), "SecurityHeadersMiddleware is not registered"

    # ...and every header we promise is non-empty.
    for name, value in SECURITY_HEADERS.items():
        assert value, f"{name} configured empty"
    assert "Strict-Transport-Security" in SECURITY_HEADERS
    assert "X-Content-Type-Options" in SECURITY_HEADERS
    assert "X-Frame-Options" in SECURITY_HEADERS
    assert "Content-Security-Policy" in SECURITY_HEADERS


# ── M1: rate limiting ───────────────────────────────────────────────


def test_sliding_window_allows_then_blocks():
    from src.api.rate_limit import InMemorySlidingWindow

    lim = InMemorySlidingWindow(limit=3, window_seconds=60)
    assert lim.allow("k", now=1000.0)
    assert lim.allow("k", now=1000.1)
    assert lim.allow("k", now=1000.2)
    assert not lim.allow("k", now=1000.3), "4th hit inside the window must block"


def test_sliding_window_recovers_after_window():
    from src.api.rate_limit import InMemorySlidingWindow

    lim = InMemorySlidingWindow(limit=2, window_seconds=10)
    assert lim.allow("k", now=100.0)
    assert lim.allow("k", now=101.0)
    assert not lim.allow("k", now=102.0)
    # once the window has slid past both hits, the caller is clean again
    assert lim.allow("k", now=112.5)


def test_sliding_window_keys_are_independent():
    from src.api.rate_limit import InMemorySlidingWindow

    lim = InMemorySlidingWindow(limit=1, window_seconds=60)
    assert lim.allow("a", now=1.0)
    assert not lim.allow("a", now=2.0)
    assert lim.allow("b", now=2.0), "one caller must not exhaust another's budget"


def test_sliding_window_evicts_idle_keys():
    """An attacker rotating source IPs must not grow the limiter without
    bound — that turns a rate limiter into a memory-exhaustion vector.
    """
    from src.api.rate_limit import InMemorySlidingWindow

    lim = InMemorySlidingWindow(limit=1, window_seconds=10)
    for i in range(500):
        lim.allow(f"ip-{i}", now=float(i))
    # Long after every window has expired, one more call sweeps the dead keys.
    lim.allow("fresh", now=10_000.0)
    assert lim.tracked_keys() <= 2, (
        f"expected idle keys to be evicted, still tracking {lim.tracked_keys()}"
    )


def test_client_key_prefers_proxy_appended_forwarded_for():
    """Railway's edge APPENDS the real client IP to X-Forwarded-For, so
    the right-most entry is the trustworthy one. Taking the left-most
    would let an attacker spoof a fresh identity per request and bypass
    the limit entirely.
    """
    from src.api.rate_limit import client_key

    # A single entry is the edge's own attribution — trust it over the
    # socket peer, which behind a proxy is just the proxy.
    assert client_key("203.0.113.9", "10.0.0.1") == "203.0.113.9"
    # An attacker prepending a fake hop must not mint a new identity:
    # the edge appends the address it actually saw, so right-most wins.
    assert client_key("spoofed, 203.0.113.9", "10.0.0.1") == "203.0.113.9"
    assert client_key("a, b, 203.0.113.9", "10.0.0.1") == "203.0.113.9"
    # No header (direct connection, local dev): fall back to the peer.
    assert client_key(None, "1.2.3.4") == "1.2.3.4"
    assert client_key("", "1.2.3.4") == "1.2.3.4"
    assert client_key("  ,  ", "1.2.3.4") == "1.2.3.4"
    # Nothing identifiable: one shared bucket, i.e. fail closed.
    assert client_key(None, None) == "unknown"
