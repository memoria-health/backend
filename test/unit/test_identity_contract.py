"""The /state seam (gpu/identity/contract.py) — version + auth doors.

The contract module is the ONLY file shared between the backend and
the GPU-host identity service. These tests pin its two guarantees:
auth is constant-time-boring and fail-closed when configured, and a
version-mismatched service reads as "no verdict" (fail-open), never a
silent misparse.
"""

import asyncio
from types import SimpleNamespace

from identity.contract import STATE_CONTRACT, auth_ok
from src.services.reid.identity_client import IdentityClient


def test_auth_disabled_when_no_token_configured():
    assert auth_ok(None, "") is True
    assert auth_ok("Bearer whatever", "") is True


def test_auth_requires_bearer_when_configured():
    assert auth_ok(None, "s3cret") is False
    assert auth_ok("s3cret", "s3cret") is False          # missing scheme
    assert auth_ok("Bearer wrong", "s3cret") is False
    assert auth_ok("Bearer s3cret", "s3cret") is True


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload):
        self._payload = payload
        self.seen_headers = None

    async def get(self, url, headers=None):
        self.seen_headers = headers
        return _Resp(self._payload)


def _sidecar(payload, token=""):
    sc = IdentityClient(settings=SimpleNamespace(
        identity_state_url="http://x:1", identity_provider="sidecar",
        identity_state_token=token,
    ))
    sc._client = _Client(payload)
    return sc


def test_wrong_contract_reads_as_no_verdict():
    sc = _sidecar({"contract": "state-v99", "ready": True, "cameras": {}})
    assert asyncio.run(sc._state()) is None


def test_missing_contract_reads_as_no_verdict():
    sc = _sidecar({"ready": True, "cameras": {}})
    assert asyncio.run(sc._state()) is None


def test_matching_contract_flows_and_bearer_is_sent():
    payload = {"contract": STATE_CONTRACT, "ready": True, "cameras": {}}
    sc = _sidecar(payload, token="tok")
    assert asyncio.run(sc._state()) == payload
    assert sc._client.seen_headers == {"Authorization": "Bearer tok"}


def test_no_token_sends_no_auth_header():
    sc = _sidecar({"contract": STATE_CONTRACT, "ready": True, "cameras": {}})
    asyncio.run(sc._state())
    assert sc._client.seen_headers == {}
