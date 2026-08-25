"""Integration tests for `GET /v1/events/{id}/clip-url`.

Asserts:
  - 200 + signed URL when has_clip=true and caregiver owns the event.
  - 409 when has_clip=false (clip still processing or worker crashed).
  - 404 when the event doesn't exist OR isn't owned by the caregiver
    (we return 404 not 403 so we don't leak existence).
  - 401 without auth.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.event import Event
from src.services import storage as storage_module
from test.conftest import mint_test_jwt


# Fake that ONLY intercepts the /storage/v1/object/sign call against
# the configured Supabase URL. Anything else (e.g. conftest's auth
# admin POSTs) passes through to a real httpx.AsyncClient so seeding
# stays functional.
_real_async_client = httpx.AsyncClient


class _FakeAsyncClient:
    state: dict[str, Any] = {"responses": [], "captured": []}

    def __init__(self, *, timeout: float | None = None, **_kw: Any) -> None:
        self._timeout = timeout
        self._passthrough: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._passthrough is not None:
            await self._passthrough.aclose()
        return None

    def _is_intercepted(self, url: str) -> bool:
        return "/storage/v1/" in url

    async def _ensure_passthrough(self) -> httpx.AsyncClient:
        if self._passthrough is None:
            self._passthrough = _real_async_client(timeout=self._timeout)
        return self._passthrough

    async def post(self, url: str, **kw: Any) -> httpx.Response:
        if not self._is_intercepted(url):
            return await (await self._ensure_passthrough()).post(url, **kw)
        return self._record("POST", url, kw)

    async def get(self, url: str, **kw: Any) -> httpx.Response:
        if not self._is_intercepted(url):
            return await (await self._ensure_passthrough()).get(url, **kw)
        return self._record("GET", url, kw)

    async def delete(self, url: str, **kw: Any) -> httpx.Response:
        if not self._is_intercepted(url):
            return await (await self._ensure_passthrough()).delete(url, **kw)
        return self._record("DELETE", url, kw)

    @classmethod
    def _record(cls, method: str, url: str, kw: dict[str, Any]) -> httpx.Response:
        cls.state["captured"].append({"method": method, "url": url, "kw": kw})
        if not cls.state["responses"]:
            raise RuntimeError(f"no canned response for {method} {url}")
        return cls.state["responses"].pop(0)


@pytest.fixture
def fake_httpx(monkeypatch):
    state: dict[str, Any] = {"responses": [], "captured": []}
    _FakeAsyncClient.state = state
    monkeypatch.setattr(storage_module.httpx, "AsyncClient", _FakeAsyncClient)
    return state


async def _provision(client: httpx.AsyncClient, auth_user) -> dict[str, str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Clip Watcher"})
    return headers


async def _create_patient_and_event(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    session: AsyncSession,
    *,
    has_clip: bool,
) -> str:
    r = await client.post(
        "/v1/patients", headers=headers,
        json={
            "full_name": "Margaret O'Brien",
            "preferred_name": "Maggie",
            "date_of_birth": "1944-05-12",
            "gender": "Female",
            "relation": "parent",
            "home_address": "12 Cedar Ln",
        },
    )
    assert r.status_code == 201
    pid = r.json()["id"]

    now = datetime.now(timezone.utc)
    event_id = uuid4()
    session.add(
        Event(
            id=event_id, patient_id=pid,
            detector_id=None, camera_id=None,
            category="safety", source="preset", icon_name="fall",
            title="Fall", location="Bedroom", occurred_at=now,
            has_clip=has_clip, narrative="Maggie fell.",
            opened=False, reviewed=False,
            review_choice=None, review_notes=None, reviewed_at=None,
            clip_storage_key=f"events/{event_id}.mp4" if has_clip else None,
            created_at=now,
        )
    )
    await session.commit()
    return str(event_id)


@pytest.mark.asyncio
async def test_clip_url_points_at_our_own_proxy_when_ready(
    client: httpx.AsyncClient, auth_user, session, fake_httpx,
):
    """Was: asserts a storage-signed absolute URL.

    That URL was built from `settings.supabase_url`, the address the
    BACKEND uses to reach Supabase — unroutable from a phone, so the app
    got a player that silently never started (2026-08-23). The clip is
    served from our own origin now, and this asserts the new contract
    rather than the shape that could not work.

    No `fake_httpx` response is queued any more: minting this URL no
    longer talks to storage at all. The bytes are fetched, under the
    caregiver's own token, only when the player actually opens.
    """
    headers = await _provision(client, auth_user)
    event_id = await _create_patient_and_event(client, headers, session, has_clip=True)

    r = await client.get(f"/v1/events/{event_id}/clip-url", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["url"] == f"/v1/events/{event_id}/clip"
    assert body["expires_in_seconds"] == 600


@pytest.mark.asyncio
async def test_clip_url_returns_409_when_not_ready(
    client: httpx.AsyncClient, auth_user, session, fake_httpx,
):
    headers = await _provision(client, auth_user)
    event_id = await _create_patient_and_event(client, headers, session, has_clip=False)

    r = await client.get(f"/v1/events/{event_id}/clip-url", headers=headers)
    assert r.status_code == 409
    assert r.json() == {"detail": "clip_not_ready"}
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_clip_url_returns_404_when_event_missing(
    client: httpx.AsyncClient, auth_user, fake_httpx,
):
    headers = await _provision(client, auth_user)
    r = await client.get(f"/v1/events/{uuid4()}/clip-url", headers=headers)
    assert r.status_code == 404
    assert r.json() == {"detail": "event_not_found"}
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_clip_url_returns_404_cross_caregiver(
    client: httpx.AsyncClient, auth_user_factory, session, fake_httpx,
):
    """Caregiver A creates an event; caregiver B asks for its clip
    URL. We return 404 (not 403) so we don't leak existence."""
    cg_a = await auth_user_factory()
    cg_b = await auth_user_factory()
    h_a = await _provision(client, cg_a)
    h_b = await _provision(client, cg_b)

    event_id = await _create_patient_and_event(client, h_a, session, has_clip=True)

    r = await client.get(f"/v1/events/{event_id}/clip-url", headers=h_b)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_clip_url_requires_auth(client: httpx.AsyncClient) -> None:
    r = await client.get(f"/v1/events/{uuid4()}/clip-url")
    assert r.status_code == 401
