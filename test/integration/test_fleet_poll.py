"""POST /v1/fleet/poll — telemetry as a precondition of instruction.

`docs/design/fleet-service-contract.md`. The camera bridge polled this
backend for `desired_state`, `privacy_enabled` and `audio_capture_enabled`,
acted on none of them, and reported nothing back — and nothing detected
that for weeks, because reporting was optional and acting was unverified.

The fix is structural rather than procedural: you cannot receive
instructions without saying what you did with the last ones. A service
that stops reporting stops being told what to do and goes visibly stale
within one poll interval.

The second rule here is that health is per ENTITY. One bridge polls for
many cameras, and `GET /v1/gateway/roster` stamps `last_heartbeat_at` for
every camera on the box just because the box is alive — so a camera whose
worker died ten minutes ago reads as healthy. "I am alive" is not a
statement about the things I am responsible for.

Requires the same deltas as the other gateway suites.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt, mint_gateway_code

_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}
TOKEN = "bridge-token-FLEET"


def report(entities: list[dict] | None = None, **service) -> dict:
    body = {
        "contract": "fleet-v1",
        "service": {
            "kind": "camera-bridge",
            "instance_id": "bridge-0ea7166c",
            "version": "2.0.0",
            "capabilities": ["privacy_sever", "audio_kill"],
            **service,
        },
        "entities": entities if entities is not None else [],
    }
    return body


def entity(camera_id: str, *, healthy: bool = True, **kw) -> dict:
    return {
        "id": camera_id,
        "state": "publishing" if healthy else "video_stalled",
        "healthy": healthy,
        "detail": "",
        "last_frame_age_s": 0.4 if healthy else None,
        "frame_count": 900,
        "restarts": 0,
        "applied": {"desired_state": "connected", "room": "patient_x",
                    "privacy_enabled": False, "audio_capture_enabled": False},
        **kw,
    }


async def _setup(client: AsyncClient, auth_user) -> tuple[dict, str, str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    p = (await client.post("/v1/patients", headers=headers,
                           json=_PATIENT_BODY)).json()
    return headers, email, p["id"]


async def _provision(client: AsyncClient, email: str, headers: dict,
                     patient_id: str, room: str = "Living Room") -> str:
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"pairing_code": await mint_gateway_code(client, headers), "room": room},
    )
    assert r.status_code == 201, r.text
    camera_id = r.json()["cameras"][0]["camera_id"]
    a = await client.post(f"/v1/cameras/{camera_id}/coverage", headers=headers,
                          json={"primary_patient_id": patient_id})
    assert a.status_code == 200, a.text
    return camera_id


async def _enroll(client: AsyncClient, headers: dict, token: str = TOKEN) -> None:
    """Bind the bridge's token to this account.

    `/gateway/enroll` is gone (2026-08-23) — binding happens once, at
    provision, by redeeming a code. A service with nothing to provision
    has nothing to be told to run either, so the case it existed for
    stopped existing.
    """
    r = await client.post(
        "/v1/gateway/provision-camera",
        headers={"Authorization": f"Bearer {token}"},
        json={"pairing_code": await mint_gateway_code(client, headers),
              "room": f"Bind {token[-6:]}"})
    assert r.status_code == 201, r.text


async def _poll(client: AsyncClient, body: dict, token: str = TOKEN):
    return await client.post("/v1/fleet/poll",
                             headers={"Authorization": f"Bearer {token}"},
                             json=body)


# ── the instruction comes back ────────────────────────────────────
@pytest.mark.asyncio
async def test_a_reporting_bridge_is_told_what_to_run(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, email, pid = await _setup(client, auth_user)
    camera_id = await _provision(client, email, headers, pid)

    r = await _poll(client, report([entity(camera_id)]))
    assert r.status_code == 200, r.text
    cameras = r.json()["cameras"]
    assert [c["camera_id"] for c in cameras] == [camera_id]
    # Same vocabulary the gateway roster returns, so the publish decision
    # is made from one contract rather than two that can drift.
    for field in ("livekit_identity", "livekit_room", "desired_state",
                  "privacy_enabled", "audio_capture_enabled", "online"):
        assert field in cameras[0], field


@pytest.mark.asyncio
async def test_a_bridge_owning_nothing_can_still_poll(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A freshly enrolled bridge owns no cameras. If an empty report were
    rejected it could never be told about its first one."""
    headers, email, pid = await _setup(client, auth_user)
    await _provision(client, email, headers, pid)
    r = await _poll(client, report([]))
    assert r.status_code == 200, r.text


# ── the report is a PRECONDITION ──────────────────────────────────
@pytest.mark.asyncio
async def test_a_poll_with_no_report_is_refused(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The whole point. Instruction is the response to a report, so a
    service that can ask for orders without saying what it did with the
    last ones is exactly how the bridge ignored disconnect for months."""
    headers, email, _pid = await _setup(client, auth_user)
    await _enroll(client, headers)
    r = await _poll(client, {})
    assert r.status_code == 400
    assert "report" in r.json()["detail"]


@pytest.mark.asyncio
async def test_a_report_from_an_unknown_contract_is_refused(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A consumer that would misread a newer body must say so rather
    than silently misparse it — the same rule the identity contract
    already applies, where a version mismatch means "no verdict"."""
    headers, email, _pid = await _setup(client, auth_user)
    await _enroll(client, headers)
    r = await _poll(client, {**report([]), "contract": "fleet-v99"})
    assert r.status_code == 400
    assert "contract" in r.json()["detail"]


# ── liveness is per CAMERA ────────────────────────────────────────
@pytest.mark.asyncio
async def test_only_the_cameras_reported_healthy_are_stamped_alive(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The structural bug in the old GET roster: it stamped
    last_heartbeat_at for every camera on the box because the box was
    alive. A camera whose worker died read as healthy, which is the one
    thing a fleet endpoint exists to prevent — a blind camera looks
    exactly like a quiet house."""
    headers, email, pid = await _setup(client, auth_user)
    alive = await _provision(client, email, headers, pid, room="Living Room")
    dead = await _provision(client, email, headers, pid, room="Bedroom")

    await session.execute(
        text("UPDATE public.cameras SET last_heartbeat_at = NULL "
             "WHERE id IN (:a, :d)"), {"a": alive, "d": dead})
    await session.commit()

    r = await _poll(client, report([entity(alive, healthy=True),
                                    entity(dead, healthy=False)]))
    assert r.status_code == 200, r.text

    stamps = dict((str(row[0]), row[1]) for row in (await session.execute(
        text("SELECT id, last_heartbeat_at FROM public.cameras "
             "WHERE id IN (:a, :d)"), {"a": alive, "d": dead})).all())
    assert stamps[alive] is not None, "a healthy camera must be stamped alive"
    assert stamps[dead] is None, (
        "a camera the bridge reports as stalled must NOT be stamped alive "
        "just because the bridge itself is polling")


@pytest.mark.asyncio
async def test_a_camera_the_bridge_does_not_mention_is_not_stamped(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Silence about a camera is not a claim that it is fine."""
    headers, email, pid = await _setup(client, auth_user)
    camera_id = await _provision(client, email, headers, pid)
    await session.execute(
        text("UPDATE public.cameras SET last_heartbeat_at = NULL WHERE id = :c"),
        {"c": camera_id})
    await session.commit()

    r = await _poll(client, report([]))
    assert r.status_code == 200, r.text   # else the assert below is vacuous
    stamp = (await session.execute(
        text("SELECT last_heartbeat_at FROM public.cameras WHERE id = :c"),
        {"c": camera_id})).scalar()
    assert stamp is None


@pytest.mark.asyncio
async def test_a_bridge_cannot_stamp_a_camera_it_does_not_own(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The report names camera ids. Trusting them without checking
    ownership would let any enrolled bridge mark any camera in the fleet
    as healthy."""
    headers, email, pid = await _setup(client, auth_user)
    mine = await _provision(client, email, headers, pid)

    other = (await session.execute(
        text("SELECT id FROM public.cameras WHERE id <> :m LIMIT 1"),
        {"m": mine})).scalar()
    if other is None:
        pytest.skip("no second camera in the database to attempt")

    await session.execute(
        text("UPDATE public.cameras SET last_heartbeat_at = NULL WHERE id = :o"),
        {"o": other})
    await session.commit()
    r = await _poll(client, report([entity(str(other))]))
    assert r.status_code == 200, r.text   # else the assert below is vacuous
    stamp = (await session.execute(
        text("SELECT last_heartbeat_at FROM public.cameras WHERE id = :o"),
        {"o": other})).scalar()
    assert stamp is None


# ── auth ──────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_an_unknown_token_cannot_poll(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, email, _pid = await _setup(client, auth_user)
    await _enroll(client, headers)
    r = await _poll(client, report([]), token="not-a-real-bridge-token")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_a_missing_bearer_cannot_poll(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    headers, email, _pid = await _setup(client, auth_user)
    await _enroll(client, headers)
    r = await client.post("/v1/fleet/poll", json=report([]))
    assert r.status_code == 401


# ── the SERVICE's own heartbeat ───────────────────────────────────
#
# Camera liveness above is per entity, deliberately. But it can only ever
# speak for cameras that EXIST, and `gateways.last_seen_at` was never
# stamped on this path at all — so a bridge with no cameras, or one that
# died, was indistinguishable from a healthy one. Live on 2026-08-24: the
# running bridge's row read `last_seen_at = NULL` beside the mini-PC
# gateway's `12:09`, and soak_check's bridge assertion passed on it.
async def _last_seen(session: AsyncSession, token: str = TOKEN):
    import hashlib

    return (await session.execute(
        text("select last_seen_at from public.gateways where token_hash = :h"),
        {"h": hashlib.sha256(token.encode()).hexdigest()},
    )).scalar()


@pytest.mark.asyncio
async def test_a_poll_marks_the_service_as_seen(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """THE regression. Without this there is no way to tell a dead bridge
    from a live one."""
    headers, email, pid = await _setup(client, auth_user)
    camera_id = await _provision(client, email, headers, pid)

    assert await _last_seen(session) is None, "nothing has polled yet"
    assert (await _poll(client, report([entity(camera_id)]))).status_code == 200
    assert await _last_seen(session) is not None


@pytest.mark.asyncio
async def test_a_service_with_no_cameras_is_still_seen(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The case per-camera heartbeats can never cover: a freshly
    installed bridge, or one whose last camera was deleted. It polls
    every few seconds and used to leave no trace anywhere."""
    headers, email, pid = await _setup(client, auth_user)
    await _enroll(client, headers)

    assert (await _poll(client, report([]))).status_code == 200
    assert await _last_seen(session) is not None


@pytest.mark.asyncio
async def test_a_malformed_poll_is_not_liveness(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Reporting is a PRECONDITION of instruction here, so a poll this
    endpoint refuses must not count as the service being healthy —
    otherwise a bridge stuck on an old contract looks alive forever."""
    headers, email, pid = await _setup(client, auth_user)
    await _enroll(client, headers)

    r = await _poll(client, {**report([]), "contract": "fleet-v0"})
    assert r.status_code == 400
    assert await _last_seen(session) is None
