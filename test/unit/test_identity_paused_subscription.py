"""Pausing AI must stop the identity service pulling video too.

The detector worker and this camera's identity service hold SEPARATE
LiveKit subscriptions to the same camera. Severing only the backend's
made pausing halve the bill instead of ending it: measured 2026-08-24
with AI paused, the backend was at 0.029 GB/hr while the node was still
pulling ~1,100 KB/s.

`--only-cam` is fixed at launch and pause state is not, so this is
polled. It fails OPEN — an unreachable database must never blind a
camera whose user has NOT paused it, and "everything looks paused" is
exactly the shape a failed query would take.
"""

from __future__ import annotations

import pytest

from identity import service


@pytest.fixture(autouse=True)
def _clean():
    service._paused_cache.update(t=0.0, v=set())
    yield
    service._paused_cache.update(t=0.0, v=set())


@pytest.mark.asyncio
async def test_a_db_failure_pauses_nothing(monkeypatch):
    """THE safety property. A camera goes blind only because its user
    said so, never because a query failed."""
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(service, "get_sessionmaker", boom)

    assert await service.paused_cameras() == set()


@pytest.mark.asyncio
async def test_a_db_failure_does_not_retry_every_pass(monkeypatch):
    """The walk runs every 10s per process. A dead database must not
    turn that into a query storm on top of whatever already broke."""
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise RuntimeError("db down")

    monkeypatch.setattr(service, "get_sessionmaker", boom)
    for _ in range(5):
        await service.paused_cameras()

    assert calls["n"] == 1, "the failure must be cached like a result"


@pytest.mark.asyncio
async def test_the_result_is_cached(monkeypatch):
    calls = {"n": 0}

    class _S:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def execute(self, *a):
            calls["n"] += 1
            class R:
                def all(self_inner): return [("650ca94d",)]
            return R()

    monkeypatch.setattr(service, "get_sessionmaker", lambda: (lambda: _S()))
    first = await service.paused_cameras()
    second = await service.paused_cameras()

    assert first == {"650ca94d"} == second
    assert calls["n"] == 1
