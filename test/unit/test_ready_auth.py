"""/ready is an authenticated ops surface outside local/test.

It makes outbound calls and names our dependencies, so it must not be
open on the internet. `/health` stays open — that is the one a platform
healthcheck points at. Added 2026-07-29 alongside the not_configured
signal for a deploy that never set NEMARA_IDENTITY_STATE_URL.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import src.settings as settings_mod
from src.api.health import ready


def _settings(**over):
    base = {
        "internal_service_key": "",
        "env": "prod",
        "livekit_url": "",
        "identity_state_url": "http://host.docker.internal:8012",
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def fake_settings(monkeypatch):
    def _install(**over):
        s = _settings(**over)
        monkeypatch.setattr(settings_mod, "get_settings", lambda: s)
        return s
    return _install


async def test_prod_without_key_configured_is_refused(fake_settings):
    """Fails CLOSED: an unconfigured prod rejects rather than opens."""
    fake_settings(internal_service_key="", env="prod")
    with pytest.raises(HTTPException) as exc:
        await ready(x_internal_key="")
    assert exc.value.status_code == 403
    assert exc.value.detail == "internal_key_not_configured"


async def test_wrong_key_is_refused(fake_settings):
    fake_settings(internal_service_key="right", env="prod")
    with pytest.raises(HTTPException) as exc:
        await ready(x_internal_key="wrong")
    assert exc.value.status_code == 403
    assert exc.value.detail == "invalid_internal_key"


async def test_refused_before_any_db_connection(fake_settings, monkeypatch):
    """The rejection must cost nothing — no pool connection taken.

    This endpoint holding connections across its outbound probes is how
    the 2026-07-28 login outage started; an unauthorized caller must not
    be able to reach that code path at all.
    """
    fake_settings(internal_service_key="right", env="prod")

    def _boom():
        raise AssertionError("sessionmaker touched before auth")

    monkeypatch.setattr("src.db.session.get_sessionmaker", _boom)
    with pytest.raises(HTTPException):
        await ready(x_internal_key="")


async def test_local_needs_no_key(fake_settings, monkeypatch):
    """A laptop with no key set still gets the diagnosis layer."""
    fake_settings(internal_service_key="", env="local")
    monkeypatch.setattr(
        "src.db.session.get_sessionmaker", lambda: _ok_sessionmaker(),
    )
    resp = await ready(x_internal_key="")
    # Dev-default identity URL + env=local → the URL is legitimate here,
    # so it is probed (and fails, nothing listening) rather than being
    # reported as a misconfiguration.
    assert resp.checks["db"] == "ok"
    assert resp.checks["identity"] != "not_configured"


async def test_prod_dev_identity_url_reads_not_configured(
    fake_settings, monkeypatch,
):
    """A deploy that never set the identity URL must say so by name.

    Probing host.docker.internal from Railway just times out on every
    call, so /ready would read 'fail' forever and be ignored. The remedy
    is a deploy env var, not an incident, and the word should say that.
    """
    fake_settings(internal_service_key="k", env="prod")
    monkeypatch.setattr(
        "src.db.session.get_sessionmaker", lambda: _ok_sessionmaker(),
    )
    resp = await ready(x_internal_key="k")
    assert resp.checks["identity"] == "not_configured"
    assert resp.status == "degraded"


def _ok_sessionmaker():
    class _Result:
        def one(self):
            return (1,)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, *a, **k):
            return _Result()

    return lambda: _Session()
