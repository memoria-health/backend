"""Integration tests for `POST /v1/livekit/webhooks`.

The route does three things worth asserting:

  1. **Signature validation.** A missing or forged `Authorization`
     header returns 401 without touching the worker.
  2. **Event dispatch.** A signature-valid `participant_joined` for a
     `cam_<...>` identity in a `patient_<...>` room calls
     `worker.ensure_room_joined(room_name)`. `participant_left`
     mirrors with `ensure_room_left_if_empty`.
  3. **Filtering.** Events for non-patient rooms or non-`cam_`
     identities are accepted-but-no-op (LiveKit gets 204 so it doesn't
     retry, but the worker is untouched).

The detector worker itself is replaced with a recording stand-in so
these tests don't need LiveKit Cloud or a real RoomWorker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from httpx import AsyncClient

from src.settings import Settings, get_settings


@pytest.fixture(autouse=True)
def livekit_configured_app(app):
    """The webhook route 503s without LIVEKIT_API_KEY/SECRET. Local
    dev typically doesn't set those, so every test in this module
    overrides Settings via the FastAPI dep system. autouse so every
    test gets the override without naming it.
    """
    from src.api.deps import get_settings_dep

    real = get_settings()
    configured = Settings(**{
        **real.model_dump(),
        "livekit_url": real.livekit_url or "wss://test.livekit.cloud",
        "livekit_api_key": real.livekit_api_key or "fake-api-key",
        "livekit_api_secret": real.livekit_api_secret or "fake-api-secret",
    })
    app.dependency_overrides[get_settings_dep] = lambda: configured
    yield
    app.dependency_overrides.pop(get_settings_dep, None)


@dataclass
class _RecordingWorker:
    """Replaces `DetectorWorker` on `app.state.detector_worker`.
    Captures every call so we can assert routing + filtering.
    """
    joined: list[str] = field(default_factory=list)
    left_checked: list[str] = field(default_factory=list)

    async def ensure_room_joined(self, room_name: str) -> None:
        self.joined.append(room_name)

    async def ensure_room_left_if_empty(self, room_name: str) -> None:
        self.left_checked.append(room_name)


@pytest.fixture
def installed_worker(app):
    """Hang a recording worker on `app.state.detector_worker` so the
    route can find it.
    """
    worker = _RecordingWorker()
    app.state.detector_worker = worker
    yield worker
    delattr(app.state, "detector_worker")


@pytest.fixture
def fake_receiver(monkeypatch):
    """Replace `WebhookReceiver` with a stub that bypasses signature
    verification — used by the routing/filtering tests. The
    signature-failure test does NOT use this fixture.
    """
    captured: dict = {}

    class _FakeReceiver:
        def __init__(self, *_args, **_kw): ...
        def receive(self, body: str, auth_token: str):
            payload = json.loads(body)
            captured["body"] = body
            captured["auth_token"] = auth_token
            return SimpleNamespace(
                event=payload["event"],
                room=SimpleNamespace(name=payload.get("room", {}).get("name", "")),
                participant=SimpleNamespace(
                    identity=payload.get("participant", {}).get("identity", ""),
                ),
            )
    monkeypatch.setattr("src.api.livekit_webhooks.lkapi.WebhookReceiver", _FakeReceiver)
    return captured


def _payload(event: str, room_name: str = "", identity: str = "") -> str:
    return json.dumps({
        "event": event,
        "room": {"name": room_name},
        "participant": {"identity": identity},
    })


# ─── routing ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_participant_joined_for_camera_calls_ensure_joined(
    client: AsyncClient, installed_worker, fake_receiver,
) -> None:
    body = _payload(
        "participant_joined",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="cam_ce110001",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Authorization": "Bearer fake.jwt", "Content-Type": "application/json"},
    )
    assert r.status_code == 204
    assert installed_worker.joined == [
        "patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
    ]
    assert installed_worker.left_checked == []


@pytest.mark.asyncio
async def test_participant_left_for_camera_calls_ensure_left(
    client: AsyncClient, installed_worker, fake_receiver,
) -> None:
    body = _payload(
        "participant_left",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="cam_ce110001",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Authorization": "Bearer fake.jwt", "Content-Type": "application/json"},
    )
    assert r.status_code == 204
    assert installed_worker.left_checked == [
        "patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
    ]
    assert installed_worker.joined == []


# ─── filtering ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_patient_room_is_no_op(
    client: AsyncClient, installed_worker, fake_receiver,
) -> None:
    body = _payload("participant_joined", room_name="other_room", identity="cam_x")
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Authorization": "Bearer fake.jwt", "Content-Type": "application/json"},
    )
    assert r.status_code == 204
    assert installed_worker.joined == []
    assert installed_worker.left_checked == []


@pytest.mark.asyncio
async def test_caregiver_identity_in_patient_room_is_no_op(
    client: AsyncClient, installed_worker, fake_receiver,
) -> None:
    """A caregiver entering Talk mode shows up in the room too. The
    webhook receives a `participant_joined` event for the
    `caregiver_<...>` identity — must NOT cause the worker to
    (re)join the room, which it's already in.
    """
    body = _payload(
        "participant_joined",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="caregiver_a4f1c2d8-5c7e-4dab-9c2f-72e1f3b0aa01",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Authorization": "Bearer fake.jwt", "Content-Type": "application/json"},
    )
    assert r.status_code == 204
    assert installed_worker.joined == []


@pytest.mark.asyncio
async def test_unrelated_event_type_is_no_op(
    client: AsyncClient, installed_worker, fake_receiver,
) -> None:
    body = _payload(
        "room_started",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Authorization": "Bearer fake.jwt", "Content-Type": "application/json"},
    )
    assert r.status_code == 204
    assert installed_worker.joined == []
    assert installed_worker.left_checked == []


# ─── signature failures ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_authorization_header_returns_401(
    client: AsyncClient, installed_worker,
) -> None:
    """No fake_receiver fixture — the real signature path runs and
    must reject. Worker must not be touched.
    """
    body = _payload(
        "participant_joined",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="cam_ce110001",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 401
    assert installed_worker.joined == []


@pytest.mark.asyncio
async def test_invalid_signature_returns_401(
    client: AsyncClient, installed_worker,
) -> None:
    """A bogus Authorization header fails JWT verification ⇒ 401."""
    body = _payload(
        "participant_joined",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="cam_ce110001",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={
            "Authorization": "Bearer not.a.real.jwt",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 401
    assert installed_worker.joined == []


# ─── worker not started ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_204_when_worker_not_installed(
    client: AsyncClient, fake_receiver, app,
) -> None:
    """If lifespan never started the worker (`worker_enabled=false`),
    the route accepts the webhook silently so LiveKit doesn't retry.
    """
    if hasattr(app.state, "detector_worker"):
        delattr(app.state, "detector_worker")
    body = _payload(
        "participant_joined",
        room_name="patient_5b1c0312-44a0-4f9d-9a3a-2d4f44a1f0bb",
        identity="cam_ce110001",
    )
    r = await client.post(
        "/v1/livekit/webhooks",
        content=body,
        headers={"Authorization": "Bearer fake.jwt", "Content-Type": "application/json"},
    )
    assert r.status_code == 204
