"""Integration tests for `role='gateway'` on POST /v1/livekit/token
(2026-08-03).

The gateway box used to sign its OWN LiveKit tokens with the raw API
key + secret held in `cameras.json`. That is how the production LiveKit
credential came to exist in three places at once — the box, a committed
ops doc, and a leaked `.env`. Rotating it meant a coordinated fleet
update, so in practice it did not get rotated.

Now the box authenticates with the ONE site token it already uses for
`/v1/gateway/roster` and asks the backend to mint a per-camera token.
The LiveKit secret lives only in the backend's env.

That makes this endpoint a privilege boundary, so the tests below are
mostly about what it must REFUSE. Three checks have to hold together:
the bearer is a real gateway, the identity names a camera owned by THAT
gateway, and the room is that camera's own room.

Requires the same deltas as test_gateway_roster.py.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.integration.test_gateway_roster import _provision, _setup_patient


async def _mint(client: AsyncClient, token: str, *, room: str, identity: str):
    return await client.post(
        "/v1/livekit/token",
        headers={"Authorization": f"Bearer {token}"},
        json={"room": room, "identity": identity, "role": "gateway"},
    )


@pytest.mark.asyncio
async def test_gateway_mints_token_for_its_own_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The happy path: a box gets a JWT for a camera it provisioned,
    with no LiveKit secret ever leaving the backend."""
    _, email, _ = await _setup_patient(client, auth_user)
    cam = (await _provision(client, email, "gw-token-A", room="Living Room"))["cameras"][0]

    r = await _mint(
        client, "gw-token-A",
        room=cam["livekit_room"], identity=cam["livekit_identity"],
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["token"]
    # A JWT, not an opaque string — three dot-separated segments.
    assert body["token"].count(".") == 2
    assert body["identity"] == cam["livekit_identity"]


@pytest.mark.asyncio
async def test_gateway_cannot_mint_for_another_boxs_camera(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """THE escalation case. Two boxes under one caregiver: A must not be
    able to publish as B's camera. Before per-box ownership existed, any
    holder of any site token could have signed a token for any camera."""
    _, email, _ = await _setup_patient(client, auth_user)
    await _provision(client, email, "gw-token-A", room="Room A")
    b_cam = (await _provision(client, email, "gw-token-B", room="Room B"))["cameras"][0]

    r = await _mint(
        client, "gw-token-A",
        room=b_cam["livekit_room"], identity=b_cam["livekit_identity"],
    )
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_gateway_cannot_mint_for_another_room(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Right camera, wrong room. The room must be the camera's own, or a
    box could join a room it has no business in."""
    _, email, _ = await _setup_patient(client, auth_user)
    cam = (await _provision(client, email, "gw-token-A", room="Living Room"))["cameras"][0]

    r = await _mint(
        client, "gw-token-A",
        room="patient_00000000-0000-0000-0000-000000000000",
        identity=cam["livekit_identity"],
    )
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_unknown_gateway_token_is_rejected(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """A bearer that hashes to no gateway row gets nothing."""
    _, email, _ = await _setup_patient(client, auth_user)
    cam = (await _provision(client, email, "gw-token-A", room="Living Room"))["cameras"][0]

    r = await _mint(
        client, "gw-token-NOT-REAL",
        room=cam["livekit_room"], identity=cam["livekit_identity"],
    )
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_unowned_camera_is_not_mintable(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """`gateway_id IS NULL` cameras are visible on every box's roster as a
    transitional fallback. Minting is stricter on purpose: an unowned
    camera has no proven box, so it must NOT be publishable. Read access
    failing open is tolerable; a publish credential is not."""
    _, email, pid = await _setup_patient(client, auth_user)
    await _provision(client, email, "gw-token-A", room="Owned")

    from uuid import uuid4
    orphan = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.cameras "
            "(id, patient_id, display_id, room, icon_name, online, created_at, "
            " livekit_identity, livekit_room, source_type, ingest_token_hash, gateway_id) "
            "VALUES (:id, :pid, 98, 'Orphan', 'tv', false, now(), "
            " 'cam_orphan_mint', :room, 'gateway', :tok, NULL)"
        ),
        {"id": str(orphan), "pid": pid, "room": f"patient_{pid}", "tok": "y" * 64},
    )
    await session.commit()

    r = await _mint(
        client, "gw-token-A",
        room=f"patient_{pid}", identity="cam_orphan_mint",
    )
    assert r.status_code == 401, r.text
