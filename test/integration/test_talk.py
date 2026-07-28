"""Integration tests for caregiver Talk mode.

Three endpoints:
  - GET  /v1/patients/{id}/active-camera   — suggestion for the iOS carousel
  - POST /v1/patients/{id}/talk/start      — opens a call to an explicit camera
  - POST /v1/patients/{id}/talk/end        — closes any open call

The LiveKit data-send is mocked via monkeypatch — we capture every
`SendDataRequest` to assert routing decisions without hitting LiveKit
Cloud.
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

from sqlalchemy import update

from src.db.analysis import Analysis
from src.db.camera import Camera
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
    """Captures every `send_data` call. The endpoint constructs
    `lkapi.LiveKitAPI(url, key, secret)` per packet; we monkeypatch the
    class to return a fake whose `room.send_data` appends here.
    """

    sent: list[_SentPacket] = field(default_factory=list)
    closed: int = 0

    def install(self, monkeypatch) -> None:
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
    """Talk endpoints short-circuit to 503 unless LiveKit env is set.
    Override Settings so the tests reach the send_data path even when
    the local env is bare.
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


# ─── GET /active-camera ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_active_camera_returns_freshest_qualifying(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """When the patient has a recent `analyses.people_present=true`
    row, the endpoint echoes that camera_id.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    cam_a, _ = await _pair_camera(client, headers, pid, "Kitchen")

    uid, _ = auth_user
    session.add(Analysis(
        id=uuid4(),
        caregiver_id=uid,
        patient_id=pid,
        camera_id=cam_a,
        provider="gemini",
        description="walking",
        detector_fields={},
        people_present=True,
        motion_in_clip=True,
        state_after="open",
        captured_at=datetime.now(timezone.utc) - timedelta(seconds=5),
        created_at=datetime.now(timezone.utc) - timedelta(seconds=5),
    ))
    await session.commit()

    r = await client.get(f"/v1/patients/{pid}/active-camera", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"camera_id": str(cam_a)}


@pytest.mark.asyncio
async def test_get_active_camera_returns_null_when_no_qualifying_row(
    client: AsyncClient, auth_user,
) -> None:
    """No analyses row → `camera_id: null`. iOS falls back to the
    first online camera in the patient's list.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    await _pair_camera(client, headers, pid, "Kitchen")

    r = await client.get(f"/v1/patients/{pid}/active-camera", headers=headers)
    assert r.status_code == 200, r.text
    assert r.json() == {"camera_id": None}


@pytest.mark.asyncio
async def test_get_active_camera_returns_403_for_unowned_patient(
    client: AsyncClient, auth_user_factory,
) -> None:
    """Caregiver B querying A's patient → 403."""
    uid_a, email_a = await auth_user_factory()
    headers_a = {"Authorization": f"Bearer {mint_test_jwt(uid_a, email_a)}"}
    await client.post("/v1/me", headers=headers_a, json={"display_name": "A"})
    pid_a = UUID((await client.post("/v1/patients", headers=headers_a, json={
        "full_name": "A Patient", "preferred_name": "A", "relation": "parent",
    })).json()["id"])

    uid_b, email_b = await auth_user_factory()
    headers_b = {"Authorization": f"Bearer {mint_test_jwt(uid_b, email_b)}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "B"})

    r = await client.get(f"/v1/patients/{pid_a}/active-camera", headers=headers_b)
    assert r.status_code == 403


# ─── POST /talk/start ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_routes_to_specified_camera(
    livekit_configured_app, client: AsyncClient, auth_user, data_send_mock,
) -> None:
    """Caregiver picks cam_a explicitly. Server sends `is_active=true`
    to cam_a, `is_active=false` to cam_b. The `analyses` table is
    not consulted — routing is request-driven.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    cam_a, identity_a = await _pair_camera(client, headers, pid, "Kitchen")
    _, identity_b = await _pair_camera(client, headers, pid, "Bedroom")

    r = await client.post(
        f"/v1/patients/{pid}/talk/start",
        headers=headers,
        json={"camera_id": str(cam_a)},
    )
    assert r.status_code == 204, r.text

    assert len(data_send_mock.sent) == 2
    by_identity = {p.destination_identities[0]: p for p in data_send_mock.sent}
    assert by_identity[identity_a].payload == {"type": "set_talk_active", "is_active": True}
    assert by_identity[identity_a].room == f"patient_{pid}"
    assert by_identity[identity_a].topic == "set_talk_active"
    assert by_identity[identity_b].payload == {"type": "set_talk_active", "is_active": False}
    assert by_identity[identity_b].room == f"patient_{pid}"


@pytest.mark.asyncio
async def test_start_returns_422_when_camera_id_missing(
    livekit_configured_app, client: AsyncClient, auth_user, data_send_mock,
) -> None:
    """`camera_id` is required. Empty body → 422, no packet sent."""
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    await _pair_camera(client, headers, pid, "Kitchen")

    r = await client.post(f"/v1/patients/{pid}/talk/start", headers=headers, json={})
    assert r.status_code == 422
    assert data_send_mock.sent == []


@pytest.mark.asyncio
async def test_start_returns_403_when_camera_not_owned_by_patient(
    livekit_configured_app, client: AsyncClient, auth_user_factory,
    data_send_mock,
) -> None:
    """A camera_id belonging to a different patient (or random uuid) →
    403, no packet sent. Defends against an iOS-side bug picking up a
    stale camera_id from another patient's roster.
    """
    uid_a, email_a = await auth_user_factory()
    headers_a = {"Authorization": f"Bearer {mint_test_jwt(uid_a, email_a)}"}
    await client.post("/v1/me", headers=headers_a, json={"display_name": "A"})
    pid_a = UUID((await client.post("/v1/patients", headers=headers_a, json={
        "full_name": "A1", "preferred_name": "A1", "relation": "parent",
    })).json()["id"])
    pid_other = UUID((await client.post("/v1/patients", headers=headers_a, json={
        "full_name": "A2", "preferred_name": "A2", "relation": "parent",
    })).json()["id"])
    other_cam, _ = await _pair_camera(client, headers_a, pid_other, "Kitchen")

    # Try to start a call on patient A1's room with patient A2's camera.
    r = await client.post(
        f"/v1/patients/{pid_a}/talk/start",
        headers=headers_a,
        json={"camera_id": str(other_cam)},
    )
    assert r.status_code == 403
    assert data_send_mock.sent == []


@pytest.mark.asyncio
async def test_start_returns_403_for_unowned_patient(
    livekit_configured_app, client: AsyncClient, auth_user_factory,
    data_send_mock,
) -> None:
    """Caregiver B starting a call on A's patient → 403, no packet."""
    uid_a, email_a = await auth_user_factory()
    headers_a = {"Authorization": f"Bearer {mint_test_jwt(uid_a, email_a)}"}
    await client.post("/v1/me", headers=headers_a, json={"display_name": "A"})
    pid_a = UUID((await client.post("/v1/patients", headers=headers_a, json={
        "full_name": "A Patient", "preferred_name": "A", "relation": "parent",
    })).json()["id"])
    cam_a, _ = await _pair_camera(client, headers_a, pid_a, "Kitchen")

    uid_b, email_b = await auth_user_factory()
    headers_b = {"Authorization": f"Bearer {mint_test_jwt(uid_b, email_b)}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "B"})

    r = await client.post(
        f"/v1/patients/{pid_a}/talk/start",
        headers=headers_b,
        json={"camera_id": str(cam_a)},
    )
    assert r.status_code == 403
    assert data_send_mock.sent == []


@pytest.mark.asyncio
async def test_start_returns_403_when_target_camera_is_private(
    livekit_configured_app, client: AsyncClient, auth_user,
    session: AsyncSession, data_send_mock,
) -> None:
    """A camera with `privacy_enabled=true` (bathroom/bedroom room
    class) can never become the active talk target — 403, no packet
    sent to ANY camera. iOS hides the call button for private tiles,
    but a stale client could still POST an old camera_id, so the
    server enforces it too.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    cam_a, _ = await _pair_camera(client, headers, pid, "Bathroom")
    await _pair_camera(client, headers, pid, "Kitchen")

    # privacy_enabled is written only by the room classifier — flip it
    # directly in the DB, as the classifier would.
    await session.execute(
        update(Camera).where(Camera.id == cam_a).values(privacy_enabled=True)
    )
    await session.commit()

    r = await client.post(
        f"/v1/patients/{pid}/talk/start",
        headers=headers, json={"camera_id": str(cam_a)},
    )
    assert r.status_code == 403
    assert r.json()["detail"] == "talk_not_available_for_private_room_camera"
    assert data_send_mock.sent == []


@pytest.mark.asyncio
async def test_start_to_non_private_camera_still_works_with_private_sibling(
    livekit_configured_app, client: AsyncClient, auth_user,
    session: AsyncSession, data_send_mock,
) -> None:
    """Happy path is unchanged when the TARGET is non-private: the call
    opens on the target and the private sibling still receives the
    `is_active=false` mute (muting private cameras is always safe —
    only activation is gated).
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    cam_a, identity_a = await _pair_camera(client, headers, pid, "Kitchen")
    cam_b, identity_b = await _pair_camera(client, headers, pid, "Bathroom")

    await session.execute(
        update(Camera).where(Camera.id == cam_b).values(privacy_enabled=True)
    )
    await session.commit()

    r = await client.post(
        f"/v1/patients/{pid}/talk/start",
        headers=headers, json={"camera_id": str(cam_a)},
    )
    assert r.status_code == 204, r.text

    assert len(data_send_mock.sent) == 2
    by_identity = {p.destination_identities[0]: p for p in data_send_mock.sent}
    assert by_identity[identity_a].payload == {"type": "set_talk_active", "is_active": True}
    assert by_identity[identity_b].payload == {"type": "set_talk_active", "is_active": False}


@pytest.mark.asyncio
async def test_start_transfers_call_when_called_again_with_different_camera(
    livekit_configured_app, client: AsyncClient, auth_user, data_send_mock,
) -> None:
    """Re-POSTing with a different `camera_id` transfers the call.
    Each call broadcasts to every camera, so the previous target's
    gate closes atomically with the new target's opening.
    """
    headers, pid = await _provision_caregiver_and_patient(client, auth_user)
    cam_a, identity_a = await _pair_camera(client, headers, pid, "Kitchen")
    cam_b, identity_b = await _pair_camera(client, headers, pid, "Bedroom")

    await client.post(
        f"/v1/patients/{pid}/talk/start",
        headers=headers, json={"camera_id": str(cam_a)},
    )
    data_send_mock.sent.clear()

    r = await client.post(
        f"/v1/patients/{pid}/talk/start",
        headers=headers, json={"camera_id": str(cam_b)},
    )
    assert r.status_code == 204, r.text
    assert len(data_send_mock.sent) == 2
    by_identity = {p.destination_identities[0]: p for p in data_send_mock.sent}
    assert by_identity[identity_a].payload == {"type": "set_talk_active", "is_active": False}
    assert by_identity[identity_b].payload == {"type": "set_talk_active", "is_active": True}


# ─── POST /talk/end ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_end_sends_is_active_false_to_all(
    livekit_configured_app, client: AsyncClient, auth_user, data_send_mock,
) -> None:
    """`/talk/end` sends `is_active=false` to every paired camera so a
    previous call can't leave a stale gate open.
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


# ─── 503 when LiveKit unconfigured ────────────────────────────────


@pytest.mark.asyncio
async def test_503_when_livekit_not_configured(
    app, client: AsyncClient, auth_user, monkeypatch, data_send_mock,
) -> None:
    """Without LIVEKIT_URL/KEY/SECRET the endpoint short-circuits to
    503 and no packet is sent.
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
    cam_a, _ = await _pair_camera(client, headers, pid, "Kitchen")

    r = await client.post(
        f"/v1/patients/{pid}/talk/start",
        headers=headers, json={"camera_id": str(cam_a)},
    )
    assert r.status_code == 503
    assert data_send_mock.sent == []

    app.dependency_overrides.pop(get_settings_dep, None)
