"""Integration tests for the gateway roster: per-box camera ownership
+ reverse-SSH tunnel_port assignment (branch + main union, 2026-07-19).

Each gateway camera is owned by a specific box (cameras.gateway_id), so
a caregiver with more than one gateway gets only its own cameras — plus
the transitional NULL-owner fallback. Each box's FIRST roster poll also
lazily assigns it a UNIQUE reverse-SSH relay port (folded into the
liveness UPDATE).

Requires the local DB to have `scripts/apply_gateway_delta.sql`,
`scripts/apply_gateway_ownership_delta.sql` AND
`scripts/apply_gateway_tunnel_port_delta.sql` applied, same as the
other delta-gated suites.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.ingest.pi.auth import hash_ingest_token
from test.conftest import mint_test_jwt

_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(
    client: AsyncClient, auth_user,
) -> tuple[dict[str, str], str, str]:
    """Create the caregiver + one patient. Returns (headers, email, patient_id)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    return headers, email, p["id"]


async def _provision(
    client: AsyncClient, email: str, token: str, room: str,
    *, headers: dict | None = None, patient_id: str | None = None,
) -> dict:
    """Provision a gateway camera as the box holding `token`.

    Provisioning leaves the camera on the ACCOUNT with no patient, and the
    roster skips unassigned cameras, so pass `headers`/`patient_id` when the
    case needs one actually in service.
    """
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {token}"},
        json={"email": email, "room": room},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    if headers and patient_id:
        for cam in body["cameras"]:
            a = await client.post(
                f"/v1/cameras/{cam['camera_id']}/coverage",
                headers=headers, json={"primary_patient_id": patient_id},
            )
            assert a.status_code == 200, a.text
            cam["livekit_room"] = f"patient_{patient_id}"
    return body


async def _roster_identities(client: AsyncClient, token: str) -> set[str]:
    r = await client.get(
        "/v1/gateway/roster",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    return {c["livekit_identity"] for c in r.json()["cameras"]}


@pytest.mark.asyncio
async def test_provision_stamps_owning_gateway(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A provisioned camera records the box that provisioned it."""
    headers, email, pid = await _setup_patient(client, auth_user)
    resp = await _provision(client, email, "gw-token-A", room="Living Room", headers=headers, patient_id=pid)
    camera_id = resp["cameras"][0]["camera_id"]

    gw_id = (await session.execute(
        text("SELECT id FROM public.gateways WHERE token_hash = :h"),
        {"h": hash_ingest_token("gw-token-A")},
    )).scalar_one()
    stamped = (await session.execute(
        text("SELECT gateway_id FROM public.cameras WHERE id = :id"),
        {"id": camera_id},
    )).scalar_one()
    assert str(stamped) == str(gw_id)


@pytest.mark.asyncio
async def test_roster_isolates_cameras_by_gateway(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Two boxes for the same caregiver each see only their own camera."""
    headers, email, pid = await _setup_patient(client, auth_user)
    a = await _provision(client, email, "gw-token-A", room="Room A", headers=headers, patient_id=pid)
    b = await _provision(client, email, "gw-token-B", room="Room B", headers=headers, patient_id=pid)
    a_ident = a["cameras"][0]["livekit_identity"]
    b_ident = b["cameras"][0]["livekit_identity"]

    roster_a = await _roster_identities(client, "gw-token-A")
    roster_b = await _roster_identities(client, "gw-token-B")

    assert a_ident in roster_a and b_ident not in roster_a
    assert b_ident in roster_b and a_ident not in roster_b


@pytest.mark.asyncio
async def test_roster_includes_unowned_null_cameras(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A gateway camera with no owner (gateway_id NULL) still reaches every
    box — the transitional fallback so un-backfilled rows keep working."""
    headers, email, pid = await _setup_patient(client, auth_user)
    await _provision(client, email, "gw-token-A", room="Owned", headers=headers, patient_id=pid)

    orphan = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.cameras "
            "(id, patient_id, display_id, room, icon_name, online, created_at, "
            " livekit_identity, livekit_room, source_type, ingest_token_hash, gateway_id) "
            "VALUES (:id, :pid, 99, 'Orphan', 'tv', false, now(), "
            " 'cam_orphan', :room, 'gateway', :tok, NULL)"
        ),
        {"id": str(orphan), "pid": pid, "room": f"patient_{pid}", "tok": "x" * 64},
    )
    await session.commit()

    assert "cam_orphan" in await _roster_identities(client, "gw-token-A")


@pytest.mark.asyncio
async def test_roster_assigns_stable_unique_tunnel_ports(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """First roster poll assigns a unique reverse-SSH port (never the
    reserved 22001); later polls return the SAME port; two boxes never
    collide."""
    headers, email, pid = await _setup_patient(client, auth_user)
    await _provision(client, email, "gw-token-A", room="Room A", headers=headers, patient_id=pid)
    await _provision(client, email, "gw-token-B", room="Room B", headers=headers, patient_id=pid)

    async def _port(token: str) -> int:
        r = await client.get(
            "/v1/gateway/roster",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        return r.json()["tunnel_port"]

    port_a = await _port("gw-token-A")
    port_b = await _port("gw-token-B")
    assert port_a >= 22002  # 22001 is reserved for the legacy box-01 unit
    assert port_b >= 22002
    assert port_a != port_b
    assert await _port("gw-token-A") == port_a  # stable across polls


@pytest.mark.asyncio
async def test_roster_reports_backend_view_of_frames_arriving(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The roster hands back `online` — the SUBSCRIBER's view.

    The gateway cannot tell a healthy publish from a zombie one: in the
    2026-07-29 wifi blip its room, publications and connection_state all
    read healthy while zero media left the box. Only this side knows,
    and publisher.py's watchdog restarts on it, so the field must be
    present on every camera row and must track the column.
    """
    headers, email, pid = await _setup_patient(client, auth_user)
    resp = await _provision(client, email, "gw-token-A", room="Living Room", headers=headers, patient_id=pid)
    camera_id = resp["cameras"][0]["camera_id"]

    async def _rows() -> dict[str, bool]:
        r = await client.get(
            "/v1/gateway/roster",
            headers={"Authorization": "Bearer gw-token-A"},
        )
        assert r.status_code == 200, r.text
        return {c["camera_id"]: c["online"] for c in r.json()["cameras"]}

    # A freshly provisioned camera has published nothing yet.
    assert (await _rows())[camera_id] is False

    await session.execute(
        text("UPDATE public.cameras SET online = true WHERE id = :id"),
        {"id": camera_id},
    )
    await session.commit()
    assert (await _rows())[camera_id] is True


@pytest.mark.asyncio
async def test_audio_kill_honours_every_covered_patient(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A shared camera must obey the STRICTEST covered patient.

    A camera covering two patients streams into the primary's room, so
    the roster derived `audio_capture_enabled` from `cameras.patient_id`
    — the primary — and a secondary patient's audio-off was silently
    ignored. That is the same fall-back-to-primary we removed from
    identity, sitting in the facilities audio kill, where the failure is
    a microphone open at a site that does not permit recording.
    """
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    primary = (await client.post(
        "/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    secondary = (await client.post(
        "/v1/patients", headers=headers,
        json={**_PATIENT_BODY, "full_name": "Arthur O'Brien"})).json()

    resp = await _provision(client, email, "gw-token-AUDIO", room="Shared Room")
    camera_id = resp["cameras"][0]["camera_id"]
    cover = await client.post(
        f"/v1/cameras/{camera_id}/coverage", headers=headers,
        json={"primary_patient_id": primary["id"],
              "also_patient_ids": [secondary["id"]]},
    )
    assert cover.status_code == 200, cover.text

    async def _audio() -> bool:
        r = await client.get(
            "/v1/gateway/roster",
            headers={"Authorization": "Bearer gw-token-AUDIO"},
        )
        assert r.status_code == 200, r.text
        rows = {c["camera_id"]: c["audio_capture_enabled"]
                for c in r.json()["cameras"]}
        return rows[camera_id]

    # Both permit it.
    await session.execute(
        text("UPDATE public.patients SET audio_capture_enabled = true "
             "WHERE id IN (:a, :b)"),
        {"a": primary["id"], "b": secondary["id"]},
    )
    await session.commit()
    assert await _audio() is True

    # The SECONDARY says no. The primary still says yes.
    await session.execute(
        text("UPDATE public.patients SET audio_capture_enabled = false "
             "WHERE id = :b"),
        {"b": secondary["id"]},
    )
    await session.commit()
    assert await _audio() is False, (
        "a covered patient's audio-off was ignored because the roster "
        "reads the primary only"
    )
