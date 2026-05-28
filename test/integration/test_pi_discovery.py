"""Integration tests for the Pi discovery API surface.

Covers the announce → list → claim handshake end to end:
  - POST /v1/cameras/pi-announce upserts a discovery row (no auth)
  - GET /v1/cameras/discovered lists unclaimed recent rows (caregiver auth)
  - POST /v1/cameras/pi-provision with discovery_id claims it

Cleanup: pi_discoveries has no FK to auth.users so the standard
conftest cleanup chain does not reach it. We TRUNCATE the table
before each test in this file (autouse fixture below) to keep tests
isolated from prior runs and from each other.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_sessionmaker
from test.conftest import mint_test_jwt


@pytest_asyncio.fixture(autouse=True)
async def _clean_pi_discoveries() -> None:
    """Truncate pi_discoveries before each test so state from prior
    runs (and from other tests within the same session) does not
    leak in. Discoveries have no cascade path from auth.users so
    the standard conftest cleanup doesn't catch them.
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


async def _setup_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, p["id"]


# ─── pi-announce ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_announce_inserts_a_new_discovery(
    client: AsyncClient, session: AsyncSession,
) -> None:
    """First announcement for a hardware_id creates a row."""
    body = {
        "hardware_id": "b8:27:eb:aa:bb:cc",
        "device_label": "Nemara Camera AABBCC",
        "current_url": "http://10.0.0.44:8000/stream.mjpg",
    }
    r = await client.post("/v1/cameras/pi-announce", json=body)
    assert r.status_code == 200, r.text
    discovery_id = r.json()["discovery_id"]

    # DB shape
    row = (
        await session.execute(
            text(
                "SELECT hardware_id, device_label, current_url, "
                "current_audio_url, claimed_at "
                "FROM pi_discoveries WHERE id = :id"
            ),
            {"id": discovery_id},
        )
    ).one()
    assert row.hardware_id == "b8:27:eb:aa:bb:cc"
    assert row.device_label == "Nemara Camera AABBCC"
    assert row.current_url == "http://10.0.0.44:8000/stream.mjpg"
    assert row.current_audio_url is None
    assert row.claimed_at is None


@pytest.mark.asyncio
async def test_announce_with_audio_url_stores_both(
    client: AsyncClient, session: AsyncSession,
) -> None:
    body = {
        "hardware_id": "b8:27:eb:11:22:33",
        "device_label": "Nemara Camera 112233",
        "current_url": "http://10.0.0.55:8000/stream.mjpg",
        "current_audio_url": "http://10.0.0.55:8001/audio.pcm",
    }
    r = await client.post("/v1/cameras/pi-announce", json=body)
    assert r.status_code == 200
    discovery_id = r.json()["discovery_id"]
    row = (
        await session.execute(
            text(
                "SELECT current_url, current_audio_url "
                "FROM pi_discoveries WHERE id = :id"
            ),
            {"id": discovery_id},
        )
    ).one()
    assert row.current_url == "http://10.0.0.55:8000/stream.mjpg"
    assert row.current_audio_url == "http://10.0.0.55:8001/audio.pcm"


@pytest.mark.asyncio
async def test_announce_re_announce_updates_row_not_duplicates(
    client: AsyncClient, session: AsyncSession,
) -> None:
    """Same hardware_id announced twice keeps a single row with the
    latest URL. Without this, a Pi that got a new DHCP lease would
    create a phantom duplicate.
    """
    body1 = {
        "hardware_id": "b8:27:eb:de:ad:be",
        "device_label": "Nemara Camera DEADBE",
        "current_url": "http://10.0.0.44:8000/stream.mjpg",
    }
    r1 = await client.post("/v1/cameras/pi-announce", json=body1)
    assert r1.status_code == 200
    first_id = r1.json()["discovery_id"]

    body2 = {
        **body1,
        "current_url": "http://10.0.0.99:8000/stream.mjpg",  # new IP
    }
    r2 = await client.post("/v1/cameras/pi-announce", json=body2)
    assert r2.status_code == 200
    assert r2.json()["discovery_id"] == first_id  # same row id

    # Only one row exists for this hardware_id.
    count = (
        await session.execute(
            text(
                "SELECT count(*) FROM pi_discoveries "
                "WHERE hardware_id = :hwid"
            ),
            {"hwid": "b8:27:eb:de:ad:be"},
        )
    ).scalar_one()
    assert count == 1

    row = (
        await session.execute(
            text(
                "SELECT current_url FROM pi_discoveries WHERE id = :id"
            ),
            {"id": first_id},
        )
    ).one()
    assert row.current_url == "http://10.0.0.99:8000/stream.mjpg"


@pytest.mark.asyncio
async def test_announce_is_unauthenticated(client: AsyncClient) -> None:
    """Pis have no credentials before pairing. The announce
    endpoint must accept anonymous POSTs.
    """
    body = {
        "hardware_id": "b8:27:eb:00:00:01",
        "device_label": "Nemara Camera 000001",
        "current_url": "http://10.0.0.44:8000/stream.mjpg",
    }
    r = await client.post("/v1/cameras/pi-announce", json=body)
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_announce_rejects_invalid_body(client: AsyncClient) -> None:
    # Missing required field
    r = await client.post(
        "/v1/cameras/pi-announce",
        json={"hardware_id": "x", "device_label": "Y"},
    )
    assert r.status_code == 422
    # URL too short
    r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "x",
            "device_label": "Y",
            "current_url": "http://",
        },
    )
    assert r.status_code == 422


# ─── /cameras/discovered ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_discovered_lists_unclaimed_recent_rows(
    client: AsyncClient, auth_user,
) -> None:
    headers, _ = await _setup_patient(client, auth_user)
    # Announce two distinct Pis.
    await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "b8:27:eb:de:ad:01",
            "device_label": "Nemara Camera DEAD01",
            "current_url": "http://10.0.0.44:8000/stream.mjpg",
        },
    )
    await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "b8:27:eb:de:ad:02",
            "device_label": "Nemara Camera DEAD02",
            "current_url": "http://10.0.0.55:8000/stream.mjpg",
        },
    )
    r = await client.get("/v1/cameras/discovered", headers=headers)
    assert r.status_code == 200
    discovered = r.json()["pis"]
    hwids = {pi["hardware_id"] for pi in discovered}
    assert {"b8:27:eb:de:ad:01", "b8:27:eb:de:ad:02"} <= hwids


@pytest.mark.asyncio
async def test_discovered_unauthenticated_returns_401(client: AsyncClient) -> None:
    r = await client.get("/v1/cameras/discovered")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_discovered_omits_stale_rows(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A row with last_seen_at older than the discovery window must
    NOT appear, even though it is unclaimed. Without this, devices
    that fell off the network days ago would pollute the picker.
    """
    headers, _ = await _setup_patient(client, auth_user)
    # Announce, then manually backdate the last_seen_at.
    await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "b8:27:eb:00:00:02",
            "device_label": "Stale Camera",
            "current_url": "http://10.0.0.66:8000/stream.mjpg",
        },
    )
    stale_cutoff = datetime.now(timezone.utc) - timedelta(minutes=10)
    await session.execute(
        text(
            "UPDATE pi_discoveries SET last_seen_at = :ts "
            "WHERE hardware_id = :hwid"
        ),
        {"ts": stale_cutoff, "hwid": "b8:27:eb:00:00:02"},
    )
    await session.commit()

    r = await client.get("/v1/cameras/discovered", headers=headers)
    assert r.status_code == 200
    hwids = {pi["hardware_id"] for pi in r.json()["pis"]}
    assert "b8:27:eb:00:00:02" not in hwids


@pytest.mark.asyncio
async def test_discovered_omits_claimed_rows(
    client: AsyncClient, auth_user,
) -> None:
    """Once a Pi is paired to a patient, it stops appearing in the
    discovered list.
    """
    headers, pid = await _setup_patient(client, auth_user)
    announce_r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "b8:27:eb:00:00:03",
            "device_label": "About to be claimed",
            "current_url": "http://10.0.0.77:8000/stream.mjpg",
        },
    )
    discovery_id = announce_r.json()["discovery_id"]

    # Claim via pi-provision with discovery_id.
    provision_r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Living Room", "discovery_id": discovery_id},
    )
    assert provision_r.status_code == 201, provision_r.text

    r = await client.get("/v1/cameras/discovered", headers=headers)
    hwids = {pi["hardware_id"] for pi in r.json()["pis"]}
    assert "b8:27:eb:00:00:03" not in hwids


# ─── pi-provision via discovery_id ───────────────────────────────


@pytest.mark.asyncio
async def test_provision_with_discovery_id_copies_urls(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Provisioning from a discovery copies BOTH current_url and
    current_audio_url onto the cameras row (still happens for
    backwards compat / diagnostics even though they're vestigial).
    Also: the discovery is marked claimed and gets a one-shot
    pending_ingest_token for the next announce to consume.
    """
    headers, pid = await _setup_patient(client, auth_user)
    announce_r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "b8:27:eb:00:00:04",
            "device_label": "Nemara Camera 000004",
            "current_url": "http://10.0.0.88:8000/stream.mjpg",
            "current_audio_url": "http://10.0.0.88:8001/audio.pcm",
        },
    )
    discovery_id = announce_r.json()["discovery_id"]
    provision_r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Kitchen", "discovery_id": discovery_id},
    )
    assert provision_r.status_code == 201
    cam = provision_r.json()
    # Push-refactor: pi_stream_url / pi_audio_url no longer surface
    # in the provision response. The URLs do still land on the row.
    assert "pi_stream_url" not in cam
    assert "pi_audio_url" not in cam
    camera_id = cam["camera_id"]
    row = (
        await session.execute(
            text(
                "SELECT pi_stream_url, pi_audio_url FROM cameras "
                "WHERE id = :id"
            ),
            {"id": camera_id},
        )
    ).one()
    assert row.pi_stream_url == "http://10.0.0.88:8000/stream.mjpg"
    assert row.pi_audio_url == "http://10.0.0.88:8001/audio.pcm"

    # Verify the discovery is marked claimed, linked, and has a
    # pending_ingest_token stashed for next-announce delivery.
    drow = (
        await session.execute(
            text(
                "SELECT claimed_at, claimed_camera_id, pending_ingest_token "
                "FROM pi_discoveries WHERE id = :id"
            ),
            {"id": discovery_id},
        )
    ).one()
    assert drow.claimed_at is not None
    assert str(drow.claimed_camera_id) == camera_id
    assert drow.pending_ingest_token is not None
    assert len(drow.pending_ingest_token) >= 40


@pytest.mark.asyncio
async def test_provision_rejects_already_claimed_discovery(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    announce_r = await client.post(
        "/v1/cameras/pi-announce",
        json={
            "hardware_id": "b8:27:eb:00:00:05",
            "device_label": "Nemara Camera 000005",
            "current_url": "http://10.0.0.99:8000/stream.mjpg",
        },
    )
    discovery_id = announce_r.json()["discovery_id"]
    r1 = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Bedroom", "discovery_id": discovery_id},
    )
    assert r1.status_code == 201
    r2 = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Hallway", "discovery_id": discovery_id},
    )
    assert r2.status_code == 409
    assert r2.json()["detail"] == "discovery_already_claimed"


@pytest.mark.asyncio
async def test_provision_rejects_unknown_discovery(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={
            "room": "Living Room",
            "discovery_id": "00000000-0000-0000-0000-000000000000",
        },
    )
    assert r.status_code == 404
    assert r.json()["detail"] == "discovery_not_found"


@pytest.mark.asyncio
async def test_provision_rejects_both_discovery_and_url(
    client: AsyncClient, auth_user,
) -> None:
    """The Pydantic validator on PiProvisionRequest enforces
    exactly-one-mode; both at once should 422.
    """
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={
            "room": "Living Room",
            "discovery_id": "00000000-0000-0000-0000-000000000000",
            "pi_stream_url": "http://10.0.0.44:8000/stream.mjpg",
        },
    )
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_provision_rejects_neither_discovery_nor_url(
    client: AsyncClient, auth_user,
) -> None:
    headers, pid = await _setup_patient(client, auth_user)
    r = await client.post(
        f"/v1/patients/{pid}/cameras/pi-provision",
        headers=headers,
        json={"room": "Living Room"},
    )
    assert r.status_code == 422
