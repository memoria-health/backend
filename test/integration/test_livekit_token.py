"""Integration tests for POST /v1/livekit/token.

Dual-auth: camera api_key OR caregiver JWT. The role decides which
path is checked and which grants are issued.

We dial bcrypt rounds down to 4 (same trick as test_pairing) and set
dummy LIVEKIT_URL/KEY/SECRET via the Settings dependency override.
JWTs are decoded with `verify_signature=False` so we can inspect
the LiveKit grants without re-implementing the HS256 verify.
"""

from __future__ import annotations

from uuid import UUID

import jwt as pyjwt
import pytest
from httpx import AsyncClient

from src.settings import Settings, get_settings
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


# Dummy LiveKit creds for tests. Real keys aren't needed — we only
# decode the JWTs locally (no LiveKit server roundtrip).
_LIVEKIT_URL = "wss://test.livekit.cloud"
_LIVEKIT_KEY = "test-key"
_LIVEKIT_SECRET = "test-secret-must-be-long-enough-for-hs256-signing"


@pytest.fixture
def livekit_app(app):
    """Inject the test LiveKit creds + fast bcrypt rounds into Settings."""
    from src.api.deps import get_settings_dep

    real = get_settings()
    fast = Settings(**{
        **real.model_dump(),
        "bcrypt_rounds": 4,
        "livekit_url": _LIVEKIT_URL,
        "livekit_api_key": _LIVEKIT_KEY,
        "livekit_api_secret": _LIVEKIT_SECRET,
    })
    app.dependency_overrides[get_settings_dep] = lambda: fast
    yield app
    app.dependency_overrides.pop(get_settings_dep, None)


@pytest.fixture
def livekit_disabled_app(app):
    """LiveKit creds left blank → endpoint must 503."""
    from src.api.deps import get_settings_dep

    real = get_settings()
    fast = Settings(**{
        **real.model_dump(),
        "bcrypt_rounds": 4,
        "livekit_url": "",
        "livekit_api_key": "",
        "livekit_api_secret": "",
    })
    app.dependency_overrides[get_settings_dep] = lambda: fast
    yield app
    app.dependency_overrides.pop(get_settings_dep, None)


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _pair_camera(
    client: AsyncClient, headers: dict[str, str], patient_id: str,
    *, room: str = "Living room", icon_name: str = "tv",
) -> dict:
    """Mint a pairing code + exchange it. Returns the pair response
    (camera_id, api_key, livekit_identity, livekit_room, ...).
    """
    mint = (await client.post(
        f"/v1/patients/{patient_id}/cameras/pairing-code",
        headers=headers, json={"room": room, "icon_name": icon_name},
    )).json()
    return (await client.post(
        "/v1/cameras/pair", json={"pairing_code": mint["pairing_code"]},
    )).json()


def _decode_lk(token: str) -> dict:
    return pyjwt.decode(token, options={"verify_signature": False})


# ─── Camera role ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_camera_role_token_grants(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """A camera api-key requesting role='camera' for its own room +
    identity → 200, JWT grants canPublish=True canSubscribe=True.
    """
    headers, pid = await _setup_patient(client, auth_user)
    pair = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/livekit/token",
        headers={"Authorization": f"Bearer {pair['api_key']}"},
        json={
            "room": pair["livekit_room"],
            "identity": pair["livekit_identity"],
            "role": "camera",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"] == _LIVEKIT_URL
    decoded = _decode_lk(body["token"])

    assert decoded["sub"] == pair["livekit_identity"]
    assert decoded["iss"] == _LIVEKIT_KEY
    assert decoded["video"]["room"] == pair["livekit_room"]
    assert decoded["video"]["roomJoin"] is True
    assert decoded["video"]["canPublish"] is True
    assert decoded["video"]["canSubscribe"] is True
    assert decoded["video"].get("canPublishData") is True
    # Sanity: exp > nbf (TTL applied).
    assert decoded["exp"] > decoded["nbf"]


@pytest.mark.asyncio
async def test_camera_cannot_mint_token_for_other_room(
    livekit_app, client: AsyncClient, auth_user_factory,
) -> None:
    """Camera A requesting a token for B's room → 403."""
    a_uid, a_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    pa = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]
    pair_a = await _pair_camera(client, h_a, pa)

    b_uid, b_email = await auth_user_factory()
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pb = (await client.post("/v1/patients", headers=h_b, json=_PATIENT_BODY)).json()["id"]
    pair_b = await _pair_camera(client, h_b, pb)

    # A's api_key requesting B's room.
    r = await client.post(
        "/v1/livekit/token",
        headers={"Authorization": f"Bearer {pair_a['api_key']}"},
        json={
            "room": pair_b["livekit_room"],
            "identity": pair_a["livekit_identity"],
            "role": "camera",
        },
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_camera_cannot_mint_token_for_other_identity(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """Camera A requesting role=camera with a different identity → 403."""
    headers, pid = await _setup_patient(client, auth_user)
    pair = await _pair_camera(client, headers, pid)

    r = await client.post(
        "/v1/livekit/token",
        headers={"Authorization": f"Bearer {pair['api_key']}"},
        json={
            "room": pair["livekit_room"],
            "identity": "cam_someoneelse",   # spoofed
            "role": "camera",
        },
    )
    assert r.status_code == 403


# ─── Viewer / broadcaster (caregiver JWT) ─────────────────────────


@pytest.mark.asyncio
async def test_caregiver_viewer_token(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """Caregiver JWT requesting role=viewer for their own patient's
    room → 200, can_publish=False, can_subscribe=True.
    """
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={
            "room": f"patient_{pid}",
            "identity": "viewer-caregiver-1",
            "role": "viewer",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"] == _LIVEKIT_URL
    decoded = _decode_lk(body["token"])

    assert decoded["sub"] == "viewer-caregiver-1"
    assert decoded["iss"] == _LIVEKIT_KEY
    assert decoded["video"]["room"] == f"patient_{pid}"
    assert decoded["video"]["roomJoin"] is True
    assert decoded["video"].get("canPublish", False) is False
    assert decoded["video"]["canSubscribe"] is True
    assert decoded["video"].get("canPublishData", False) is False


@pytest.mark.asyncio
async def test_caregiver_cannot_get_viewer_token_for_others_patient(
    livekit_app, client: AsyncClient, auth_user_factory,
) -> None:
    """Caregiver B requesting viewer token for A's room → 403."""
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]

    r = await client.post(
        "/v1/livekit/token",
        headers=h_b,
        json={
            "room": f"patient_{pid}",
            "identity": "viewer-b",
            "role": "viewer",
        },
    )
    assert r.status_code == 403


@pytest.mark.asyncio
async def test_caregiver_broadcaster_token(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """role=broadcaster on own patient → can_publish=True can_subscribe=True."""
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={
            "room": f"patient_{pid}",
            "identity": "broadcaster-caregiver-1",
            "role": "broadcaster",
        },
    )
    assert r.status_code == 200, r.text
    decoded = _decode_lk(r.json()["token"])
    assert decoded["video"]["room"] == f"patient_{pid}"
    assert decoded["video"]["roomJoin"] is True
    assert decoded["video"]["canPublish"] is True
    assert decoded["video"]["canSubscribe"] is True
    assert decoded["video"].get("canPublishData") is True


# ─── Privacy mode severs the caregiver's media grants ────────────


async def _set_privacy(client: AsyncClient, headers: dict[str, str], pid: str) -> None:
    r = await client.patch(
        f"/v1/patients/{pid}", headers=headers, json={"privacy_mode_enabled": True},
    )
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_viewer_token_severed_under_privacy(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """Privacy on → a viewer token may JOIN (for insights/UI) but is
    granted neither subscribe (no live in) nor publish (no Talk out).
    Baked into the JWT so it survives reconnects."""
    headers, pid = await _setup_patient(client, auth_user)
    await _set_privacy(client, headers, pid)

    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={"room": f"patient_{pid}", "identity": "viewer-1", "role": "viewer"},
    )
    assert r.status_code == 200, r.text
    v = _decode_lk(r.json()["token"])["video"]
    assert v["roomJoin"] is True  # still in the room for insights
    assert v.get("canSubscribe", False) is False  # no live video/audio in
    assert v.get("canPublish", False) is False  # no Talk audio out
    assert v.get("canPublishData", False) is False


@pytest.mark.asyncio
async def test_broadcaster_token_severed_under_privacy(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """Privacy on → even a broadcaster (Talk) token is fully severed."""
    headers, pid = await _setup_patient(client, auth_user)
    await _set_privacy(client, headers, pid)

    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={"room": f"patient_{pid}", "identity": "bc-1", "role": "broadcaster"},
    )
    assert r.status_code == 200, r.text
    v = _decode_lk(r.json()["token"])["video"]
    assert v["roomJoin"] is True
    assert v.get("canSubscribe", False) is False
    assert v.get("canPublish", False) is False


# ─── Room-name validation ────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_room_format_rejected(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """A caregiver requesting a viewer token for a non-`patient_<uuid>`
    room → 403 (the room name is the load-bearing identifier).
    """
    headers, _ = await _setup_patient(client, auth_user)

    # Wrong prefix.
    r1 = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={"room": "totally_bogus_room", "identity": "x", "role": "viewer"},
    )
    assert r1.status_code == 403

    # patient_ prefix but the UUID is malformed.
    r2 = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={"room": "patient_not-a-uuid", "identity": "x", "role": "viewer"},
    )
    assert r2.status_code == 403


# ─── Auth errors ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_missing_auth_header_rejected(
    livekit_app, client: AsyncClient,
) -> None:
    """No bearer at all → 401."""
    r = await client.post(
        "/v1/livekit/token",
        json={
            "room": "patient_00000000-0000-0000-0000-000000000000",
            "identity": "x",
            "role": "viewer",
        },
    )
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_503_when_livekit_not_configured(
    livekit_disabled_app, client: AsyncClient, auth_user,
) -> None:
    """LIVEKIT_URL/KEY/SECRET unset → 503 before any auth dispatch."""
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={
            "room": f"patient_{pid}",
            "identity": "viewer-1",
            "role": "viewer",
        },
    )
    assert r.status_code == 503
    assert r.json()["detail"] == "livekit_not_configured"


# ─── Camera role rejected for caregiver JWT ───────────────────────


@pytest.mark.asyncio
async def test_camera_role_with_caregiver_jwt_rejected(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """Caregiver JWT requesting role='camera' goes down the camera-auth
    path. The JWT is a well-formed bearer but matches no camera
    `api_key_hash`, so `camera_auth` raises 404 `camera_not_found` (the
    documented "well-formed bearer, no matching camera" contract that the
    wall phone uses to detect deletion), and `_resolve_camera_or_401`
    preserves it. Either way the caregiver JWT cannot mint a camera token.
    """
    headers, pid = await _setup_patient(client, auth_user)

    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={
            "room": f"patient_{pid}",
            "identity": "cam_anything",
            "role": "camera",
        },
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "camera_not_found"


# ─── Sanity: token round-trips to a UUID-format livekit_room ──────


@pytest.mark.asyncio
async def test_caregiver_viewer_token_decodes_to_patient_uuid_room(
    livekit_app, client: AsyncClient, auth_user,
) -> None:
    """Smoke check that the decoded JWT's room == `patient_<patient_uuid>`."""
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        "/v1/livekit/token",
        headers=headers,
        json={
            "room": f"patient_{pid}",
            "identity": "viewer",
            "role": "viewer",
        },
    )
    assert r.status_code == 200
    decoded = _decode_lk(r.json()["token"])
    # Room is `patient_<uuid>` and the suffix parses.
    assert decoded["video"]["room"].startswith("patient_")
    UUID(decoded["video"]["room"].removeprefix("patient_"))
