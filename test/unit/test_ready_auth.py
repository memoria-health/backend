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


async def test_prod_with_no_registered_node_is_not_benign(
    fake_settings, monkeypatch,
):
    """Prod with an empty roster means NOBODY is watching any home.

    This used to read 'not_configured', on the reasoning that a deploy
    which never set the identity URL wanted an env var rather than an
    incident. Under the roster that reasoning inverts: production nodes
    register themselves, so an empty roster is not a missing setting, it
    is an absent fleet — every home blind, and the backend failing open
    over it. It must not be reported with a benign-sounding word.
    """
    fake_settings(internal_service_key="k", env="prod")
    monkeypatch.setattr(
        "src.db.session.get_sessionmaker", lambda: _ok_sessionmaker(),
    )
    resp = await ready(x_internal_key="k")
    assert resp.checks["identity"] == "no_nodes_registered"
    assert resp.status == "degraded"


async def test_all_nodes_unreachable_reads_zero_of_n(
    fake_settings, monkeypatch,
):
    """Registered but unreachable is the tailnet/ACL failure, and it must count.

    Nothing is listening on these ports, so both probes fail. The single-URL
    check this replaces would have reported on at most one node.
    """
    fake_settings(internal_service_key="k", env="prod")
    monkeypatch.setattr(
        "src.db.session.get_sessionmaker",
        lambda: _ok_sessionmaker(nodes=[
            ("pod-a", "http://127.0.0.1", 65001),
            ("pod-b", "http://127.0.0.1", 65002),
        ]),
    )
    resp = await ready(x_internal_key="k")
    assert resp.checks["identity"] == "degraded_0_of_2_nodes"
    # Each failing node is named, so the reader knows WHICH to go look at.
    assert resp.checks["identity_node:pod-a"] == "fail"
    assert resp.checks["identity_node:pod-b"] == "fail"
    assert resp.status == "degraded"


async def test_only_broken_nodes_get_their_own_line(fake_settings, monkeypatch):
    """A per-node key appears only for a node that is NOT ok.

    Otherwise the check grows a line per home and stops being readable at
    the ten-node scale this design is for.
    """
    fake_settings(internal_service_key="k", env="prod")
    monkeypatch.setattr(
        "src.db.session.get_sessionmaker",
        lambda: _ok_sessionmaker(nodes=[("pod-a", "http://127.0.0.1", 65001)]),
    )
    resp = await ready(x_internal_key="k")
    named = {k for k in resp.checks if k.startswith("identity_node:")}
    # This node fails (nothing is listening), so it IS named...
    assert named == {"identity_node:pod-a"}
    # ...and every named node is a failing one, never a healthy one.
    assert all(resp.checks[k] != "ok" for k in named)


def _ok_sessionmaker(nodes=()):
    """A sessionmaker whose SELECT 1 succeeds and whose roster returns `nodes`.

    `nodes` is what `_identity_node_targets` reads: rows of
    (node_id, base_url, lowest_assigned_port).
    """
    class _Result:
        def one(self):
            return (1,)

        def all(self):
            return list(nodes)

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def execute(self, *a, **k):
            return _Result()

    return lambda: _Session()
