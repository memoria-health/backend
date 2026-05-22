"""Integration tests for caregiver Talk mode.

POST /v1/patients/{id}/talk/start and /talk/end. The interesting
behaviours are:

- ownership check (cross-caregiver → 403)
- active-camera routing (most recent analyses row wins)
- fan-out fallback when no active row exists
- end_talk sends is_active=false to every paired camera
- 503 when LiveKit env not configured

The LiveKit data-send is mocked via monkeypatch — we capture the
arguments to assert routing decisions without hitting LiveKit Cloud.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.settings import Settings, get_settings
from test.conftest import mint_test_jwt


# ─── Mock harness ─────────────────────────────────────────────────


@dataclass
class _SentPacket:
    room: str
    destination_identities: list[str]
    payload: dict[str, Any]
    topic: str


@dataclass
class _DataSendMock:
    """Captures every send_data call. The endpoint uses
    `lkapi.LiveKitAPI(...).room.send_data(SendDataRequest(...))`; we
    monkeypatch `LiveKitAPI` so each constructor returns a fresh
    `_FakeAPI` that records into this mock.
    """

    sent: list[_SentPacket] = field(default_factory=list)
    closed: int = 0

    def install(self, monkeypatch) -> None:
        # The endpoint constructs `lkapi.LiveKitAPI(url, key, secret)`.
        # Replace the class with a factory that returns our fake.
        mock_self = self

        class _FakeRoomService:
            async def send_data(self, req):
                mock_self.sent.append(_SentPacket(
                    room=req.room,
                    destination_identities=list(req.destination_identities),
                    payload=json.loads(bytes(req.data).decode("utf-8")),
                    topic=req.topic,
                ))

        class _FakeAPI:
            def __init__(self, *_args, **_kw):
                self.room = _FakeRoomService()

            async def aclose(self):
                mock_self.closed += 1

        monkeypatch.setattr("src.api.talk.lkapi.LiveKitAPI", _FakeAPI)


@pytest_asyncio.fixture
async def data_send_mock(monkeypatch) -> _DataSendMock:
    mock = _DataSendMock()
    mock.install(monkeypatch)
    return mock


@pytest.fixture
def livekit_configured_app(app, monkeypatch):
    """The Talk endpoint short-circuits to 503 unless the LiveKit
    triple (url, api_key, api_secret) is configured. Tests run against
    a Settings instance that may or may not have those set; override
    to ensure the endpoint reaches the send_data path.
    """
    from src.api.deps import get_settings_dep

    real = get_settings()
    configured = Settings(**{
        **real.model_dump(),
        "livekit_url": real.livekit_url or "wss://test.livekit.cloud",
        "livekit_api_key": real.livekit_api_key or "fake-api-key",
        "livekit_api_secret": real.livekit_api_secret or "fake-api-secret",
        "bcrypt_rounds": 4,
    })
    app.dependency_overrides[get_settings_dep] = lambda: configured
    yield app
    app.dependency_overrides.pop(get_settings_dep, None)


# ─── Helpers ──────────────────────────────────────────────────────


async def _provision_caregiver_and_patient(
    client: AsyncClient, auth_user,
) -> tuple[dict[str, str], UUID]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "Margaret O'Brien", "preferred_name": "Maggie", "relation": "parent",
    })).json()
    return headers, UUID(p["id"])


async def _pair_camera(
    client: AsyncClient, headers: dict[str, str], pid: UUID, room: str,
) -> tuple[UUID, str]:
    mint = (await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers, json={"room": room, "icon_name": "tv"},
    )).json()
    pair = (await client.post(
        "/v1/cameras/pair", json={"pairing_code": mint["pairing_code"]},
    )).json()
    return UUID(pair["camera_id"]), pair["livekit_identity"]


# ─── Tests ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_routes_to_active_camera(
    livekit_configured_app, client: AsyncClient, auth_user,
    session: AsyncSession, data_send_mock,
) -> None:
    """Two cameras paired. A recent analyses row says cam_a has people
    visible. talk/start should send is_active=true to cam_a, false to
    cam_b.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    cam_a, identity_a = await _pair_camera(client, headers, pid, "Kitchen")
    cam_b, identity_b = await _pair_camera(client, headers, pid, "Bedroom")

    uid, _ = auth_user
    session.add(Analysis(
        id=uuid4(),
        caregiver_id=uid,
        patient_id=pid,
        camera_id=cam_a,
        provider="gemini",
        description="person walking",
        detector_fields={},
        people_present=True,
        motion_in_clip=True,
        state_after="open",
        captured_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        created_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    ))
    await session.commit()

    r = await client.post(f"/v1/patients/{pid}/talk/start", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["active_camera_id"] == str(cam_a)

    # Two packets sent — one is_active=true to cam_a, one is_active=false to cam_b.
    assert len(data_send_mock.sent) == 2
    by_identity = {p.destination_identities[0]: p for p in data_send_mock.sent}
    assert by_identity[identity_a].payload == {"type": "set_talk_active", "is_active": True}
    assert by_identity[identity_a].room == f"patient_{pid}"
    assert by_identity[identity_a].topic == "set_talk_active"
    assert by_identity[identity_b].payload == {"type": "set_talk_active", "is_active": False}
    assert by_identity[identity_b].room == f"patient_{pid}"


@pytest.mark.asyncio
async def test_start_fans_out_when_no_active_known(
    livekit_configured_app, client: AsyncClient, auth_user, data_send_mock,
) -> None:
    """No qualifying analyses row → every paired camera gets
    is_active=true so the caregiver is heard somewhere immediately.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    _, identity_a = await _pair_camera(client, headers, pid, "Kitchen")
    _, identity_b = await _pair_camera(client, headers, pid, "Bedroom")

    r = await client.post(f"/v1/patients/{pid}/talk/start", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json()["active_camera_id"] is None

    assert len(data_send_mock.sent) == 2
    for packet in data_send_mock.sent:
        assert packet.payload == {"type": "set_talk_active", "is_active": True}
        assert packet.room == f"patient_{pid}"
        assert packet.topic == "set_talk_active"
    identities = {p.destination_identities[0] for p in data_send_mock.sent}
    assert identities == {identity_a, identity_b}


@pytest.mark.asyncio
async def test_end_sends_is_active_false_to_all(
    livekit_configured_app, client: AsyncClient, auth_user, data_send_mock,
) -> None:
    """`/talk/end` always sends is_active=false to every paired camera
    so a previous active session can't leave a stale gate open.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    _, identity_a = await _pair_camera(client, headers, pid, "Kitchen")
    _, identity_b = await _pair_camera(client, headers, pid, "Bedroom")

    r = await client.post(f"/v1/patients/{pid}/talk/end", headers=headers)
    assert r.status_code == 204

    assert len(data_send_mock.sent) == 2
    identities = {p.destination_identities[0] for p in data_send_mock.sent}
    assert identities == {identity_a, identity_b}
    for packet in data_send_mock.sent:
        assert packet.payload == {"type": "set_talk_active", "is_active": False}


@pytest.mark.asyncio
async def test_start_returns_403_for_unowned_patient(
    livekit_configured_app, client: AsyncClient, auth_user_factory,
    data_send_mock,
) -> None:
    """Caregiver B asking to Talk to A's patient → 403, no packet
    sent. This is the load-bearing isolation check.
    """
    uid_a, email_a = await auth_user_factory()
    headers_a = {"Authorization": f"Bearer {mint_test_jwt(uid_a, email_a)}"}
    await client.post("/v1/me", headers=headers_a, json={"display_name": "A"})
    pid_a = UUID((await client.post("/v1/patients", headers=headers_a, json={
        "full_name": "A Patient", "preferred_name": "A", "relation": "parent",
    })).json()["id"])

    uid_b, email_b = await auth_user_factory()
    headers_b = {"Authorization": f"Bearer {mint_test_jwt(uid_b, email_b)}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "B"})

    r = await client.post(f"/v1/patients/{pid_a}/talk/start", headers=headers_b)
    assert r.status_code == 403
    assert data_send_mock.sent == []


@pytest.mark.asyncio
async def test_503_when_livekit_not_configured(
    app, client: AsyncClient, auth_user, monkeypatch, data_send_mock,
) -> None:
    """Without LIVEKIT_URL/KEY/SECRET the endpoint short-circuits to
    503 — and no data packet is sent.
    """
    from src.api.deps import get_settings_dep

    real = get_settings()
    unconfigured = Settings(**{
        **real.model_dump(),
        "livekit_url": "",
        "livekit_api_key": "",
        "livekit_api_secret": "",
        "bcrypt_rounds": 4,
    })
    app.dependency_overrides[get_settings_dep] = lambda: unconfigured

    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    await _pair_camera(client, headers, pid, "Kitchen")

    r = await client.post(f"/v1/patients/{pid}/talk/start", headers=headers)
    assert r.status_code == 503
    assert data_send_mock.sent == []

    app.dependency_overrides.pop(get_settings_dep, None)
