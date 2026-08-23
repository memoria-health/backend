"""The caregiver's clip comes from OUR origin, not from storage directly.

`storage.signed_url` builds an absolute URL from `settings.supabase_url`,
which is whatever address the BACKEND uses to reach Supabase. In Docker
that is `host.docker.internal`; natively it is loopback or a LAN IP.
None of those resolve on a caregiver's phone, so the app was handed a
URL it could never fetch and showed a player that simply would not
start. Silent, and indistinguishable from a corrupt clip.

The operator verdict page already learned this and proxies its bytes
(`/act/{token}/clip`, 2026-08-xx). The caregiver app never did. Same
bug, same fix, and the same secondary benefit the operator path cites:
no separately-shareable storage URL for a household's footage exists at
any point.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from test.conftest import mint_test_jwt, seed_camera

_PATIENT = {"full_name": "Margaret O'Brien", "preferred_name": "Maggie",
            "relation": "parent"}


async def _event_with_clip(client, session, auth_user, *, key: str | None):
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "T"})
    pid = (await client.post("/v1/patients", headers=headers,
                             json=_PATIENT)).json()["id"]
    cam = await seed_camera(session, caregiver_id=uid, patient_id=pid,
                            room="Living Room")
    row = (await session.execute(text(
        "INSERT INTO public.events "
        "  (patient_id, camera_id, category, source, icon_name, title, "
        "   location, occurred_at, has_clip, clip_storage_key) "
        "VALUES (:p, :c, 'safety', 'preset', 'figure.fall', 'Fall', "
        "        'Living Room', now(), :h, :k) "
        "RETURNING id::text"),
        {"p": pid, "c": str(cam), "h": key is not None, "k": key})).scalar_one()
    await session.commit()
    return headers, row


@pytest.mark.asyncio
async def test_the_url_points_at_our_own_api_not_storage(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """The regression. A storage host is not addressable from a phone."""
    headers, event_id = await _event_with_clip(
        client, session, auth_user, key="clips/x.mp4")

    r = await client.get(f"/v1/events/{event_id}/clip-url", headers=headers)
    assert r.status_code == 200, r.text
    url = r.json()["url"]
    assert "/storage/v1/object/sign" not in url, (
        "handing the app a storage-signed URL is the bug: it is built from "
        "the address the BACKEND uses for Supabase"
    )
    assert url.endswith(f"/v1/events/{event_id}/clip")


@pytest.mark.asyncio
async def test_the_clip_itself_needs_the_caregivers_own_token(
    client: AsyncClient, auth_user, auth_user_factory, session: AsyncSession,
) -> None:
    """Proxying puts the bytes behind OUR auth, so it has to actually be
    checked. A neighbour's token must not fetch this household."""
    _headers, event_id = await _event_with_clip(
        client, session, auth_user, key="clips/x.mp4")
    other = await auth_user_factory()
    stranger = {"Authorization": f"Bearer {mint_test_jwt(*other)}"}

    r = await client.get(f"/v1/events/{event_id}/clip", headers=stranger)
    assert r.status_code == 404, r.text


@pytest.mark.asyncio
async def test_an_unauthenticated_fetch_is_refused(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    _headers, event_id = await _event_with_clip(
        client, session, auth_user, key="clips/x.mp4")
    r = await client.get(f"/v1/events/{event_id}/clip")
    assert r.status_code == 401, r.text


@pytest.mark.asyncio
async def test_a_clip_that_is_not_ready_still_409s(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Unchanged: the app distinguishes 'still processing' from 'gone'."""
    headers, event_id = await _event_with_clip(client, session, auth_user, key=None)
    r = await client.get(f"/v1/events/{event_id}/clip-url", headers=headers)
    assert r.status_code == 409, r.text
