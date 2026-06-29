"""Integration tests for account deletion (DELETE /v1/me) and the
App Review demo pairing code path.

Both rely on a local Supabase (`supabase start`) + applied schema, same
as the rest of the integration suite — `auth.users` rows are real, so
DELETE /v1/me exercises the genuine auth-admin delete + FK cascade rather
than a mock.
"""

from __future__ import annotations

import re
from uuid import UUID

import bcrypt
import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.pairing import DEMO_CAMERA_IDENTITY
from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.patient import Patient
from src.settings import Settings, get_settings
from test.conftest import mint_test_jwt

_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}
_PAIRING_BODY = {"room": "Living room", "icon_name": "tv"}
_DEMO_CODE = "424242"


@pytest.fixture
def demo_settings_app(app):
    """Override Settings with a demo pairing code + fast bcrypt, wired
    through FastAPI's dependency override so the pair route sees them.
    Mirrors `test_pairing.fast_bcrypt_app`.
    """
    real = get_settings()
    fast = Settings(
        **{**real.model_dump(), "bcrypt_rounds": 4, "demo_pairing_code": _DEMO_CODE}
    )
    from src.api.deps import get_settings_dep

    app.dependency_overrides[get_settings_dep] = lambda: fast
    yield app
    app.dependency_overrides.pop(get_settings_dep, None)


async def _setup_caregiver_patient(
    client: AsyncClient, auth_user
) -> tuple[dict[str, str], UUID, str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Jane"})
    pid = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()["id"]
    return headers, uid, pid


# ─── DELETE /v1/me ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_me_cascades_account(
    client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    """DELETE /v1/me removes the caregiver and everything under it via
    the auth.users → caregivers → patients → cameras cascade."""
    headers, uid, pid = await _setup_caregiver_patient(client, auth_user)
    mint = (
        await client.post(
            f"/v1/patients/{pid}/cameras/pairing-code", headers=headers, json=_PAIRING_BODY
        )
    ).json()
    camera_id = UUID(mint["camera_id"])

    r = await client.delete("/v1/me", headers=headers)
    assert r.status_code == 204, r.text

    # Caregiver, patient, camera rows are all gone.
    session.expire_all()
    assert await session.get(Caregiver, uid) is None
    assert await session.get(Patient, UUID(pid)) is None
    assert await session.get(Camera, camera_id) is None

    # The JWT still verifies (HS256, unexpired), but there's no caregiver
    # row anymore → GET /v1/me is 404, not 200.
    g = await client.get("/v1/me", headers=headers)
    assert g.status_code == 404


@pytest.mark.asyncio
async def test_delete_me_requires_auth(client: AsyncClient) -> None:
    """No bearer token → 401, before any DB work."""
    r = await client.delete("/v1/me")
    assert r.status_code == 401


# ─── Demo pairing code ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_demo_pairing_code_is_reusable(
    demo_settings_app, client: AsyncClient, auth_user, session: AsyncSession
) -> None:
    """The configured demo code pairs the demo camera repeatedly without
    expiring or being cleared — each pair mints a fresh api_key."""
    headers, uid, pid = await _setup_caregiver_patient(client, auth_user)
    # Seed the demo camera: mint a normal row, then stamp it with the stable
    # demo identity (what scripts/seed_demo_camera.py does). The demo-pair
    # path matches on livekit_identity == DEMO_CAMERA_IDENTITY — NOT on
    # pairing_code (the caregiver app rotates an unpaired camera's code). We
    # also set the demo code on the row so the "pairing fields survive each
    # pair" assertions below stay meaningful.
    mint = (
        await client.post(
            f"/v1/patients/{pid}/cameras/pairing-code", headers=headers, json=_PAIRING_BODY
        )
    ).json()
    camera_id = UUID(mint["camera_id"])
    await session.execute(
        text(
            "UPDATE cameras SET pairing_code = :c, livekit_identity = :ident "
            "WHERE id = :id"
        ),
        {"c": _DEMO_CODE, "ident": DEMO_CAMERA_IDENTITY, "id": str(camera_id)},
    )
    await session.commit()

    r1 = await client.post("/v1/cameras/pair", json={"pairing_code": _DEMO_CODE})
    assert r1.status_code == 200, r1.text
    r2 = await client.post("/v1/cameras/pair", json={"pairing_code": _DEMO_CODE})
    assert r2.status_code == 200, r2.text

    k1, k2 = r1.json()["api_key"], r2.json()["api_key"]
    assert re.fullmatch(r"^[0-9a-f]{64}$", k1)
    assert k1 != k2  # fresh key each pair

    # The code survived both pairs (still reusable), expiry was NOT cleared,
    # and the stored hash matches the LATEST key.
    session.expire_all()
    cam = await session.get(Camera, camera_id)
    assert cam is not None
    assert cam.pairing_code == _DEMO_CODE
    assert cam.pairing_expires_at is not None
    assert bcrypt.checkpw(k2.encode(), cam.api_key_hash.encode())


@pytest.mark.asyncio
async def test_demo_code_not_provisioned_returns_404(
    demo_settings_app, client: AsyncClient
) -> None:
    """Demo code is set but no camera row holds it → distinct 404
    (vs the opaque 401 the normal path returns)."""
    r = await client.post("/v1/cameras/pair", json={"pairing_code": _DEMO_CODE})
    assert r.status_code == 404
    assert r.json()["detail"] == "demo_camera_not_provisioned"


@pytest.mark.asyncio
async def test_demo_code_off_by_default(client: AsyncClient) -> None:
    """Without demo_pairing_code configured, the demo value is just an
    unknown code → 401, no special-casing."""
    r = await client.post("/v1/cameras/pair", json={"pairing_code": _DEMO_CODE})
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_or_expired_pairing_code"
