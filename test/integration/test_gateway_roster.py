"""Integration tests for the gateway roster: per-box camera ownership
+ reverse-SSH tunnel_port assignment (branch + main union, 2026-07-19).

Each gateway camera is owned by a specific box (cameras.gateway_id), so
a caregiver with more than one gateway gets only its own cameras — plus
the transitional NULL-owner fallback. Each box's FIRST roster poll also
lazily assigns it a UNIQUE reverse-SSH relay port (folded into the
liveness UPDATE).

Also covers `POST /v1/gateway/enroll`: how a service's token becomes
known to the backend WITHOUT provisioning a camera, which is what makes
the roster answer it at all.

Requires the local DB to have `scripts/deltas/apply_gateway_delta.sql`,
`scripts/deltas/apply_gateway_ownership_delta.sql` AND
`scripts/deltas/apply_gateway_tunnel_port_delta.sql` applied, same as the
other delta-gated suites.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.fleet_auth import hash_ingest_token
from test.conftest import mint_test_jwt, provision_sensor, seed_camera

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
    """Bind `token` to the account and create a sensor.

    Binding takes a pairing code the caregiver generated (2026-08-23) —
    an email proved nothing, since it is on the invoice. A token already
    bound ignores the code, so a second sensor needs no second code.
    `email` is retained in the signature because these suites read
    naturally with it; the endpoint treats it as a typo check only.
    """
    assert headers is not None, "provisioning needs caregiver headers for the code"
    body = await provision_sensor(client, headers, token=token, room=room)
    if patient_id:
        for cam in body["cameras"]:
            a = await client.post(
                f"/v1/cameras/{cam['camera_id']}/coverage",
                headers=headers, json={"patient_ids": [patient_id]},
            )
            assert a.status_code == 200, a.text
    return body


async def _gateway_rows(session: AsyncSession, token: str) -> list:
    """Every gateways row matching `token`'s hash — plural on purpose, so a
    duplicate row shows up as a length rather than hiding behind a
    scalar_one()."""
    return (await session.execute(
        text("SELECT id, caregiver_id FROM public.gateways WHERE token_hash = :h"),
        {"h": hash_ingest_token(token)},
    )).all()


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
    await seed_camera(
        session, patient_id=pid, camera_id=orphan, display_id=99,
        room="Orphan", online=False, source_type="gateway",
        livekit_identity="cam_orphan", livekit_room=f"patient_{pid}",
        ingest_token_hash="x" * 64, gateway_id=None,
    )

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
async def test_the_audio_kill_belongs_to_the_sensor(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The audio kill is the SENSOR's, and coverage cannot weaken it.

    It used to be derived from the covered patients with a bool_and, to
    stop a shared camera obeying only its primary. That was the right
    patch for the wrong model: the kill is a statement about a physical
    room ("no recording in here"), not about a person, and once a sensor
    could cover nobody the expression returned NULL — which fails closed,
    so every unassigned sensor would have been silently deaf.

    So it is a column on the sensor now, and the properties to hold are
    that the roster reports it verbatim, and that assigning or
    unassigning patients never changes it.

    Open: `patients.audio_capture_enabled` still exists and is still on
    the patient API, but nothing reads it any more. Wiring that control
    to the sensors covering the patient (and deciding what a shared
    sensor does when two patients disagree) is a UI design question, not
    a silent default to pick here.
    """
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    primary = (await client.post(
        "/v1/patients", headers=headers, json=_PATIENT_BODY)).json()
    secondary = (await client.post(
        "/v1/patients", headers=headers,
        json={**_PATIENT_BODY, "full_name": "Arthur O'Brien"})).json()

    resp = await _provision(client, email, "gw-token-AUDIO", room="Shared Room", headers=headers)
    camera_id = resp["cameras"][0]["camera_id"]
    cover = await client.post(
        f"/v1/cameras/{camera_id}/coverage", headers=headers,
        json={"patient_ids": [primary["id"], secondary["id"]]},
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

    # A fresh sensor permits audio, and covering two patients does not
    # change that — under the old bool_and this depended on both patient
    # rows, and for an unassigned sensor it collapsed to NULL.
    assert await _audio() is True

    # Killed at the sensor, which is where a facilities rule belongs.
    await session.execute(
        text("UPDATE public.cameras SET audio_capture_enabled = false "
             "WHERE id = :id"),
        {"id": camera_id},
    )
    await session.commit()
    assert await _audio() is False

    # And coverage cannot lift it. Taking every patient off the sensor
    # must not re-open a microphone at a site that does not permit one —
    # the failure the old NULL-fails-closed rule was guarding against,
    # now structural rather than incidental.
    cleared = await client.delete(
        f"/v1/cameras/{camera_id}/coverage", headers=headers)
    assert cleared.status_code == 200, cleared.text
    assert await _audio() is False


# Enrolment moved to test_fleet_pairing_code.py (2026-08-23).
# `/gateway/enroll` is gone: a token binds only by redeeming a pairing
# code at provision time, so "how a token becomes known WITHOUT a camera"
# is no longer a thing that can happen.
