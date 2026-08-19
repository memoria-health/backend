"""IdentityRouter — per-CAMERA endpoint resolution (2026-08-11,
re-keyed 2026-08-20).

The bug this class exists to prevent is silent: with a single global
identity URL and more than one GPU node, every home on any other node gets
IdentityResult.empty() forever — no error, no log, no verdict. So the cases
below assert on ROUTING, not just on "it returned something".

Routing keys on the CAMERA because a camera watching two residents
publishes into one room and so lives in one process. Keying on the tick's
patient would send the second resident's ticks to a process that has never
seen that camera — the same silent no-verdict failure, one level down.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.identity.identity_router import IdentityRouter
from src.services.identity.result import IdentityResult


class _FakeResult:
    """Marker so a test can tell WHICH node answered."""

    def __init__(self, tag: str) -> None:
        self.tag = tag


class _FakeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url
        self.calls: list[dict] = []
        self.closed = False

    async def label_clip(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResult(self.base_url)

    def timing(self, camera_id):
        return {"e2e_ms": 1.0, "node": self.base_url}

    async def aclose(self):
        self.closed = True


def _router(monkeypatch, route_map: dict, *, fail: bool = False) -> IdentityRouter:
    """Router with only its two edges stubbed: the roster lookup and the
    per-node client constructor.

    Deliberately does NOT stub `_client_for` — that method owns the
    `_clients` registry which `timing()` and `aclose()` both read, so
    replacing it would test the stub rather than the router.
    """
    monkeypatch.setattr(
        "src.services.identity.identity_router.IdentityClient",
        lambda **kw: _FakeClient(kw["base_url"]),
    )
    r = IdentityRouter(
        sessionmaker=None,
        settings=SimpleNamespace(
            identity_state_url="http://unused:8012",
            identity_state_token="",
            reid_trace_enabled=False,
            reid_trace_path="out/x.jsonl",
        ),
    )

    async def _lookup(camera_id):
        return None if fail else route_map.get(str(camera_id))

    r._base_url_for = _lookup           # type: ignore[assignment]
    return r


@pytest.mark.asyncio
async def test_each_camera_reaches_its_own_node(monkeypatch) -> None:
    """The whole point: two homes on two nodes each get their own answer.

    Before the router, the second one silently got nothing at all.
    """
    a, b = uuid4(), uuid4()
    r = _router(monkeypatch, {str(a): "http://node-a:8012", str(b): "http://node-b:8012"})

    ra = await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=a)
    rb = await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=b)

    assert ra.tag == "http://node-a:8012"
    assert rb.tag == "http://node-b:8012"


@pytest.mark.asyncio
async def test_shared_camera_answers_both_residents_from_one_node(
    monkeypatch,
) -> None:
    """A shared lounge camera lives in ONE process. Ticks for BOTH
    residents must reach that process — routing by the tick's patient
    would send the second resident somewhere that has never seen the
    camera, get no verdict, and fail open forever without an error."""
    lounge = uuid4()
    alice, bob = uuid4(), uuid4()
    r = _router(monkeypatch, {str(lounge): "http://node-a:8012"})

    ra = await r.label_clip(mp4_bytes=b"", patient_id=alice, camera_id=lounge)
    rb = await r.label_clip(mp4_bytes=b"", patient_id=bob, camera_id=lounge)

    assert ra.tag == rb.tag == "http://node-a:8012"
    # ...and the patient still travels, so the node can answer per person.
    asked = [c["patient_id"] for c in r._clients["http://node-a:8012"].calls]
    assert asked == [alice, bob]


@pytest.mark.asyncio
async def test_one_client_reused_per_node_not_per_patient(monkeypatch) -> None:
    """Two cameras on the SAME node share one connection pool."""
    a, b = uuid4(), uuid4()
    r = _router(monkeypatch, {str(a): "http://node-a:8012", str(b): "http://node-a:8012"})

    await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=a)
    await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=b)

    assert len(r._clients) == 1, "opened a second client for the same node"
    assert len(r._clients["http://node-a:8012"].calls) == 2


@pytest.mark.asyncio
async def test_unassigned_camera_fails_open(monkeypatch) -> None:
    """A camera no node is holding must fail OPEN, like an unreachable service.

    Fail-closed here would silence safety detectors for an unplaced home.
    """
    r = _router(monkeypatch, {})
    out = await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=uuid4())

    assert isinstance(out, IdentityResult)
    assert out.reid_ran is False, "must look like 'no verdict', not a verdict"


@pytest.mark.asyncio
async def test_lookup_failure_fails_open(monkeypatch) -> None:
    """A database blip must not stop detection."""
    r = _router(monkeypatch, {}, fail=True)
    out = await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=uuid4())
    assert isinstance(out, IdentityResult)
    assert out.reid_ran is False


@pytest.mark.asyncio
async def test_timing_follows_the_camera_to_its_node(monkeypatch) -> None:
    """`timing()` only receives a camera_id, so the router has to remember
    which node last served it — otherwise the dashboard reports another
    home's latency."""
    cam_a, cam_b = uuid4(), uuid4()
    r = _router(monkeypatch, {
        str(cam_a): "http://node-a:8012", str(cam_b): "http://node-b:8012",
    })

    await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=cam_a)
    await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=cam_b)

    assert r.timing(cam_a)["node"] == "http://node-a:8012"
    assert r.timing(cam_b)["node"] == "http://node-b:8012"
    # A camera never seen: no guessing.
    assert r.timing(uuid4()) is None


@pytest.mark.asyncio
async def test_aclose_closes_every_node_client(monkeypatch) -> None:
    a, b = uuid4(), uuid4()
    r = _router(monkeypatch, {str(a): "http://node-a:8012", str(b): "http://node-b:8012"})
    await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=a)
    await r.label_clip(mp4_bytes=b"", patient_id=uuid4(), camera_id=b)

    await r.aclose()
    assert all(c.closed for c in r._clients.values())
