"""Integration tests for the camera pairing handshake.

Covers both legs of the flow:
  - Caregiver mints a code:  POST /v1/patients/{id}/cameras/pairing-code
  - Wall phone exchanges it: POST /v1/cameras/pair

Bcrypt rounds are dialed down to 4 via Settings monkeypatch so we
don't spend ~100ms per test on the hash + verify roundtrip.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import bcrypt
import pytest
from httpx import AsyncClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.settings import Settings, get_settings
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}

_PAIRING_BODY = {
    "room": "Living room",
    "icon_name": "tv",
}


@pytest.fixture
def fast_bcrypt(monkeypatch):
    """bcrypt rounds=10 is intentionally slow (~80ms). Drop to 4 in
    tests so the pair endpoint runs in low single-digit milliseconds.

    Settings is `frozen=True` so we can't mutate the cached instance;
    instead we override `get_settings_dep` to return a fresh Settings
    with rounds=4.
    """
    real = get_settings()
    fast = Settings(**{**real.model_dump(), "bcrypt_rounds": 4})

    monkeypatch.setattr(
        "src.api.deps.get_settings", lambda: fast,
    )
    monkeypatch.setattr(
        "src.api.deps.get_settings_dep", lambda: fast,
    )
    # The pairing/clips/livekit routes use SettingsDep (which calls
    # get_settings_dep via FastAPI Depends) — overriding the dep is
    # how we wire the fast settings through.
    return fast


@pytest.fixture
def fast_bcrypt_app(app, fast_bcrypt):
    """Wire the fast-bcrypt Settings into FastAPI's dependency override
    so the running route sees rounds=4 instead of rounds=10.
    """
    from src.api.deps import get_settings_dep

    app.dependency_overrides[get_settings_dep] = lambda: fast_bcrypt
    yield app
    app.dependency_overrides.pop(get_settings_dep, None)


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


# ─── Mint pairing code ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mint_pairing_code_inserts_camera_row(
    client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    """Minting a code creates a placeholder cameras row with every
    field we'll need before the wall phone exchanges the code.
    """
    headers, pid = await _setup_patient(client, auth_user)
    before = datetime.now(timezone.utc)
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers,
        json=_PAIRING_BODY,
    )
    after = datetime.now(timezone.utc)
    assert r.status_code == 200

    body = r.json()
    assert re.fullmatch(r"^[1-9]\d{5}$", body["pairing_code"]), body
    camera_id = UUID(body["camera_id"])
    expires_at = datetime.fromisoformat(body["expires_at"])
    # ~10 minutes from now (allow generous slack for test timing).
    assert before + timedelta(seconds=590) <= expires_at <= after + timedelta(seconds=605)

    cam = await session.get(Camera, camera_id)
    assert cam is not None
    assert cam.id == camera_id
    assert cam.patient_id == UUID(pid)
    assert cam.display_id == 1
    assert cam.room == "Living room"
    assert cam.icon_name == "tv"
    assert cam.online is False
    assert cam.last_seen_at is None
    assert cam.pairing_code == body["pairing_code"]
    assert cam.pairing_expires_at is not None
    assert int(cam.pairing_expires_at.timestamp()) == int(expires_at.timestamp())
    assert cam.api_key_hash is None
    assert cam.livekit_identity is not None
    assert cam.livekit_identity.startswith("cam_")
    assert cam.livekit_identity == f"cam_{cam.id.hex[:8]}"
    assert cam.livekit_room == f"patient_{cam.patient_id}"
    assert cam.state == "open"
    assert cam.no_people_count == 0
    assert cam.state_changed_at is not None
    assert cam.close_threshold == 6
    assert cam.battery_percent is None


@pytest.mark.asyncio
async def test_mint_pairing_code_uses_next_display_id(
    client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    """Two consecutive mints for the same patient → display_id 1 then 2."""
    headers, pid = await _setup_patient(client, auth_user)
    r1 = await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers,
        json=_PAIRING_BODY,
    )
    r2 = await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers,
        json={"room": "Kitchen", "icon_name": "fork.knife"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200

    c1 = await session.get(Camera, UUID(r1.json()["camera_id"]))
    c2 = await session.get(Camera, UUID(r2.json()["camera_id"]))
    assert c1 is not None and c2 is not None
    assert c1.display_id == 1
    assert c2.display_id == 2
    # And the codes are distinct.
    assert r1.json()["pairing_code"] != r2.json()["pairing_code"]


@pytest.mark.asyncio
async def test_mint_pairing_code_caregiver_b_cannot_mint_for_a(
    client: AsyncClient, auth_user_factory
) -> None:
    """Caregiver B trying to mint a code for patient owned by A is a
    404 (don't leak existence).
    """
    a_uid, a_email = await auth_user_factory()
    b_uid, b_email = await auth_user_factory()
    h_a = {"Authorization": f"Bearer {mint_test_jwt(a_uid, a_email)}"}
    h_b = {"Authorization": f"Bearer {mint_test_jwt(b_uid, b_email)}"}
    await client.post("/v1/me", headers=h_a, json={"display_name": "A"})
    await client.post("/v1/me", headers=h_b, json={"display_name": "B"})
    pid = (await client.post("/v1/patients", headers=h_a, json=_PATIENT_BODY)).json()["id"]

    r = await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=h_b,
        json=_PAIRING_BODY,
    )
    assert r.status_code == 404


# ─── Exchange code for api_key ───────────────────────────────────


@pytest.mark.asyncio
async def test_pair_exchanges_code_for_api_key(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Happy path: caregiver mints → wall phone pairs → response
    carries the api_key, identity, room, livekit metadata. The
    cameras row reflects the exchange (api_key_hash set, pairing_*
    cleared).
    """
    headers, pid = await _setup_patient(client, auth_user)
    mint = (await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers,
        json=_PAIRING_BODY,
    )).json()
    code = mint["pairing_code"]
    camera_id = UUID(mint["camera_id"])

    r = await client.post(
        "/v1/cameras/pair",
        json={"pairing_code": code},
    )
    assert r.status_code == 200, r.text
    body = r.json()

    api_key = body["api_key"]
    assert re.fullmatch(r"^[0-9a-f]{64}$", api_key), api_key
    assert UUID(body["camera_id"]) == camera_id
    assert UUID(body["patient_id"]) == UUID(pid)
    # In tests LIVEKIT_URL may be empty; just assert the field is a string.
    assert isinstance(body["livekit_url"], str)
    assert body["livekit_token_endpoint"].endswith("/v1/livekit/token")
    assert body["livekit_identity"] == f"cam_{camera_id.hex[:8]}"
    assert body["livekit_room"] == f"patient_{pid}"
    assert body["backend_base_url"].startswith("http://")
    assert "/v1/cameras/pair" not in body["backend_base_url"]
    assert body["livekit_token_endpoint"].startswith(body["backend_base_url"])

    # DB-side: api_key bcrypts against the stored hash; pairing fields
    # are NULL after exchange.
    session.expire_all()
    cam = await session.get(Camera, camera_id)
    assert cam is not None
    assert cam.api_key_hash is not None
    assert bcrypt.checkpw(api_key.encode("utf-8"), cam.api_key_hash.encode("utf-8"))
    assert cam.pairing_code is None
    assert cam.pairing_expires_at is None
    assert cam.last_seen_at is not None
    # State machine fields untouched by the exchange.
    assert cam.state == "open"
    assert cam.no_people_count == 0


@pytest.mark.asyncio
async def test_pair_rejects_unknown_code(
    fast_bcrypt_app, client: AsyncClient
) -> None:
    """A random 6-digit code that's not in the DB → 401."""
    r = await client.post(
        "/v1/cameras/pair",
        json={"pairing_code": "987654"},
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_or_expired_pairing_code"


@pytest.mark.asyncio
async def test_pair_rejects_expired_code(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A code whose pairing_expires_at is in the past → 401, even
    though the row still exists in the DB.
    """
    headers, pid = await _setup_patient(client, auth_user)
    # Insert a pre-expired pairing row directly via SQL.
    cid = uuid4()
    now = datetime.now(timezone.utc)
    expired_code = "100001"
    await session.execute(
        text(
            """
            INSERT INTO cameras (
                id, patient_id, display_id, room, icon_name, online,
                created_at, pairing_code, pairing_expires_at,
                livekit_identity, livekit_room,
                state, no_people_count, state_changed_at, close_threshold
            ) VALUES (
                :id, :pid, 1, 'Living room', 'tv', false,
                :now, :code, :exp, :ident, :room,
                'open', 0, :now, 6
            )
            """
        ),
        {
            "id": str(cid),
            "pid": pid,
            "now": now,
            "code": expired_code,
            "exp": now - timedelta(seconds=1),
            "ident": f"cam_{cid.hex[:8]}",
            "room": f"patient_{pid}",
        },
    )
    await session.commit()

    r = await client.post(
        "/v1/cameras/pair",
        json={"pairing_code": expired_code},
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_or_expired_pairing_code"

    # And the row's api_key_hash is still NULL — no side effect.
    session.expire_all()
    cam = await session.get(Camera, cid)
    assert cam is not None
    assert cam.api_key_hash is None
    assert cam.pairing_code == expired_code


@pytest.mark.asyncio
async def test_pair_is_single_use(
    fast_bcrypt_app, client: AsyncClient, auth_user
) -> None:
    """Pairing exchanges the code — replaying the same code returns 401."""
    headers, pid = await _setup_patient(client, auth_user)
    mint = (await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers,
        json=_PAIRING_BODY,
    )).json()
    code = mint["pairing_code"]

    r1 = await client.post("/v1/cameras/pair", json={"pairing_code": code})
    assert r1.status_code == 200

    r2 = await client.post("/v1/cameras/pair", json={"pairing_code": code})
    assert r2.status_code == 401
    assert r2.json()["detail"] == "invalid_or_expired_pairing_code"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_code",
    [
        "abcdef",        # letters
        "12345",         # 5 digits
        "1234567",       # 7 digits
        "12 456",        # whitespace
        "",              # empty
    ],
)
async def test_pair_rejects_invalid_format(
    fast_bcrypt_app, client: AsyncClient, bad_code: str
) -> None:
    """Pydantic's `pattern=r"^\\d{6}$"` rejects anything non-6-digit
    with 422 before the route handler runs.
    """
    r = await client.post(
        "/v1/cameras/pair",
        json={"pairing_code": bad_code},
    )
    assert r.status_code == 422


# ─── Display-id / camera_id sanity ───────────────────────────────


@pytest.mark.asyncio
async def test_pair_does_not_increment_other_patient_display_ids(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The display_id sequence is scoped per-patient, not global. Two
    patients each minting their first code both get display_id=1.

    Catches a regression where MAX(display_id) was computed across all
    cameras instead of filtered to the target patient.
    """
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p1 = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()["id"]
    p2 = (await client.post("/v1/patients", headers=headers, json={
        "full_name": "Henry P.",
        "preferred_name": "Hank",
        "relation": "parent",
    })).json()["id"]

    r1 = (await client.post(
        f"/v1/patients/{p1}/cameras/pairing-code",
        headers=headers,
        json=_PAIRING_BODY,
    )).json()
    r2 = (await client.post(
        f"/v1/patients/{p2}/cameras/pairing-code",
        headers=headers,
        json=_PAIRING_BODY,
    )).json()

    c1 = await session.get(Camera, UUID(r1["camera_id"]))
    c2 = await session.get(Camera, UUID(r2["camera_id"]))
    assert c1 is not None and c2 is not None
    assert c1.display_id == 1
    assert c2.display_id == 1
    assert c1.patient_id != c2.patient_id


@pytest.mark.asyncio
async def test_camera_auth_only_matches_correct_camera(
    fast_bcrypt_app, client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Linear-scan auth (deps.camera_auth) must only resolve to the
    camera whose hash actually matches — never bleed into other paired
    cameras. We pair two cameras and verify each api_key matches
    exactly its own row.
    """
    headers, pid = await _setup_patient(client, auth_user)
    mint1 = (await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers, json=_PAIRING_BODY,
    )).json()
    mint2 = (await client.post(
        f"/v1/patients/{pid}/cameras/pairing-code",
        headers=headers, json={"room": "Kitchen", "icon_name": "fork.knife"},
    )).json()

    pair1 = (await client.post(
        "/v1/cameras/pair", json={"pairing_code": mint1["pairing_code"]},
    )).json()
    pair2 = (await client.post(
        "/v1/cameras/pair", json={"pairing_code": mint2["pairing_code"]},
    )).json()

    # Pull both cameras' hashes and confirm key1 only matches cam1, key2 only cam2.
    rows = (await session.execute(select(Camera).where(Camera.patient_id == UUID(pid)))).scalars().all()
    by_id = {c.id: c for c in rows}
    cam1 = by_id[UUID(pair1["camera_id"])]
    cam2 = by_id[UUID(pair2["camera_id"])]
    assert cam1.api_key_hash is not None
    assert cam2.api_key_hash is not None
    assert cam1.api_key_hash != cam2.api_key_hash

    assert bcrypt.checkpw(
        pair1["api_key"].encode(), cam1.api_key_hash.encode()
    )
    assert not bcrypt.checkpw(
        pair1["api_key"].encode(), cam2.api_key_hash.encode()
    )
    assert bcrypt.checkpw(
        pair2["api_key"].encode(), cam2.api_key_hash.encode()
    )
    assert not bcrypt.checkpw(
        pair2["api_key"].encode(), cam1.api_key_hash.encode()
    )
