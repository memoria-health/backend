"""Integration tests for the Pi push-ingest endpoint.

POST /v1/cameras/me/frames

These tests cover the auth contract end-to-end (mint a token via the
provision endpoint, then call the push endpoint with it) and the
shape of the announce-side one-shot token delivery. They do NOT
exercise the LiveKit publish path, because no LiveKit server is
available in the test environment; the unit-test suite in
test_pi_relay_publisher.py covers that path against in-process
fakes.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_sessionmaker
from test.conftest import mint_test_jwt


@pytest_asyncio.fixture(autouse=True)
async def _clean_pi_discoveries() -> None:
    """pi_discoveries doesn't cascade-delete with auth.users, so the
    standard conftest cleanup misses it. TRUNCATE before each test
    so a stale row from a prior test cannot satisfy the upsert
    branch by hardware_id collision.
    """
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(text("TRUNCATE TABLE pi_discoveries CASCADE"))
        await s.commit()

_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(
    client: AsyncClient, auth_user,
) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


async def _provision_pi_via_discovery(
    client: AsyncClient, headers: dict[str, str], pid: str,
) -> str:
    """Run the full discovery -> provision path. Returns the new
    camera_id. Leaves a pending_ingest_token on the discovery row
    that the next announce will consume.
    """
    hardware_id = "aa:bb:cc:dd:ee:ff"
    announce_body = {
        "hardware_id": hardware_id,
        "device_label": "Test Camera AABBCC",
        "current_url": "http://10.0.0.99/",
    }
    r = await client.post("/v1/cameras/pi-announce", json=announce_body)
    assert r.status_code == 200, r.text
    discovery_id = r.json()["discovery_id"]

    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Living Room", "discovery_id": discovery_id},
    )
    assert r.status_code == 201, r.text
    return r.json()["camera_id"]


# ─── one-shot token delivery via announce response ────────────────


@pytest.mark.asyncio
async def test_announce_after_pairing_delivers_token_once(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The very next /pi-announce after a /pi-provision returns the
    plaintext ingest token in the response. The announce immediately
    after that returns ingest_token=None (consumed).
    """
    headers, pid = await _setup_patient(client, auth_user)
    await _provision_pi_via_discovery(client, headers, pid)

    # Second announce (post-provision) — receives the token.
    r1 = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "aa:bb:cc:dd:ee:ff",
            "device_label": "Test Camera AABBCC",
            "current_url": "http://10.0.0.99/",
        },
    )
    assert r1.status_code == 200, r1.text
    body1 = r1.json()
    assert body1["paired"] is True
    assert body1["ingest_token"] is not None
    assert isinstance(body1["ingest_token"], str)
    assert len(body1["ingest_token"]) >= 40  # high-entropy plaintext
    assert body1["ingest_url"].endswith("/v1/cameras/me/frames")
    delivered_token = body1["ingest_token"]

    # Third announce — token already consumed, response carries
    # ingest_token=None but still reports paired=True.
    r2 = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "aa:bb:cc:dd:ee:ff",
            "device_label": "Test Camera AABBCC",
            "current_url": "http://10.0.0.99/",
        },
    )
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["paired"] is True
    assert body2["ingest_token"] is None
    assert body2["ingest_url"] is None

    # DB invariant: pending_ingest_token nulled, hash on cameras row
    # remains; verifying the delivered plaintext should match the
    # stored hash so the push endpoint will accept it.
    import hashlib

    digest = hashlib.sha256(delivered_token.encode("utf-8")).hexdigest()
    row = (
        await session.execute(
            text(
                "SELECT ingest_token_hash FROM cameras "
                "WHERE livekit_room = :room"
            ),
            {"room": f"patient_{pid}"},
        )
    ).one()
    assert row.ingest_token_hash == digest


@pytest.mark.asyncio
async def test_announce_before_pairing_returns_no_token(
    client: AsyncClient,
) -> None:
    """A Pi that has announced but not been paired yet gets
    paired=False and no token in the response."""
    r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "ff:ff:ff:ff:ff:ff",
            "device_label": "Unpaired Cam FFFFFF",
            "current_url": "http://10.0.0.77/",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["paired"] is False
    assert body["ingest_token"] is None
    assert body["ingest_url"] is None


# ─── push endpoint auth contract ──────────────────────────────────


@pytest.mark.asyncio
async def test_push_without_bearer_token_is_401(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/v1/cameras/me/frames",
        content=b"",
        headers={
            "Content-Type": 'multipart/x-mixed-replace; boundary="frame"',
        },
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "missing_bearer_token"


@pytest.mark.asyncio
async def test_push_with_unknown_bearer_token_is_401(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/v1/cameras/me/frames",
        content=b"",
        headers={
            "Authorization": "Bearer not-a-real-token-but-the-right-shape",
            "Content-Type": 'multipart/x-mixed-replace; boundary="frame"',
        },
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid_ingest_token"


@pytest.mark.asyncio
async def test_push_with_empty_bearer_is_401(
    client: AsyncClient,
) -> None:
    r = await client.post(
        "/v1/cameras/me/frames",
        content=b"",
        headers={
            "Authorization": "Bearer    ",
            "Content-Type": 'multipart/x-mixed-replace; boundary="frame"',
        },
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "empty_bearer_token"


@pytest.mark.asyncio
async def test_push_with_wrong_content_type_is_400(
    client: AsyncClient, auth_user,
) -> None:
    """A correctly authenticated request with the wrong Content-Type
    must fail with a clean 400, not silently start reading garbage as
    JPEG bytes.
    """
    headers, pid = await _setup_patient(client, auth_user)
    await _provision_pi_via_discovery(client, headers, pid)
    # Consume the token via an announce so we have the plaintext.
    r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "aa:bb:cc:dd:ee:ff",
            "device_label": "Test Camera AABBCC",
            "current_url": "http://10.0.0.99/",
        },
    )
    token = r.json()["ingest_token"]

    r = await client.post(
        "/v1/cameras/me/frames",
        content=b"",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "expected_multipart_mixed_replace"


@pytest.mark.asyncio
async def test_push_with_multipart_missing_boundary_is_400(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    await _provision_pi_via_discovery(client, headers, pid)
    r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "aa:bb:cc:dd:ee:ff",
            "device_label": "Test Camera AABBCC",
            "current_url": "http://10.0.0.99/",
        },
    )
    token = r.json()["ingest_token"]

    r = await client.post(
        "/v1/cameras/me/frames",
        content=b"",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "multipart/x-mixed-replace",
        },
    )
    assert r.status_code == 400
    assert r.json()["detail"] == "missing_boundary_parameter"
