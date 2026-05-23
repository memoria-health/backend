"""Integration tests for patient avatar upload + signed URL + delete.

Same fake-httpx-with-passthrough pattern as test_clip_url.py: the
fake only intercepts Supabase Storage URLs, so the conftest's
auth-user seeding (which hits real Supabase admin) stays functional.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest
from httpx import AsyncClient

from src.services import storage as storage_module
from test.conftest import mint_test_jwt


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

    def _intercepted(self, url: str) -> bool:
        return "/storage/v1/" in url

    async def _ensure_pt(self) -> httpx.AsyncClient:
        if self._passthrough is None:
            self._passthrough = _real_async_client(timeout=self._timeout)
        return self._passthrough

    async def post(self, url: str, **kw: Any) -> httpx.Response:
        if not self._intercepted(url):
            return await (await self._ensure_pt()).post(url, **kw)
        return self._record("POST", url, kw)

    async def get(self, url: str, **kw: Any) -> httpx.Response:
        if not self._intercepted(url):
            return await (await self._ensure_pt()).get(url, **kw)
        return self._record("GET", url, kw)

    async def delete(self, url: str, **kw: Any) -> httpx.Response:
        if not self._intercepted(url):
            return await (await self._ensure_pt()).delete(url, **kw)
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


async def _provision_with_patient(client: AsyncClient, auth_user) -> tuple[dict[str, str], str]:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Tester"})
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
    return headers, r.json()["id"]


_TINY_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 1024


@pytest.mark.asyncio
async def test_upload_photo_updates_storage_key(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    fake_httpx["responses"].append(httpx.Response(200, text="ok"))

    files = {"photo": ("avatar.jpg", _TINY_JPEG, "image/jpeg")}
    r = await client.post(f"/v1/patients/{pid}/photo", headers=headers, files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["photo_storage_key"] == f"{pid}.jpg"
    # Storage POST hit Supabase Storage with the right URL + body.
    assert len(fake_httpx["captured"]) == 1
    call = fake_httpx["captured"][0]
    assert call["method"] == "POST"
    assert call["url"].endswith(f"/patient-photos/{pid}.jpg")
    assert call["kw"]["content"] == _TINY_JPEG
    assert call["kw"]["headers"]["Content-Type"] == "image/jpeg"


@pytest.mark.asyncio
async def test_upload_photo_rejects_oversized(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    # 2 MB > 1 MB default cap.
    big = b"\xff\xd8" + b"\x00" * (2 * 1024 * 1024)
    files = {"photo": ("avatar.jpg", big, "image/jpeg")}
    r = await client.post(f"/v1/patients/{pid}/photo", headers=headers, files=files)
    assert r.status_code == 413
    assert "too large" in r.json()["detail"]
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_photo_rejects_unsupported_content_type(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    files = {"photo": ("avatar.gif", b"GIF89a" + b"\x00" * 100, "image/gif")}
    r = await client.post(f"/v1/patients/{pid}/photo", headers=headers, files=files)
    assert r.status_code == 400
    assert "unsupported content type" in r.json()["detail"]
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_photo_cross_caregiver_returns_404(
    client: AsyncClient, auth_user_factory, fake_httpx,
):
    headers_a, pid = await _provision_with_patient(client, await auth_user_factory())
    cg_b = await auth_user_factory()
    headers_b = {"Authorization": f"Bearer {mint_test_jwt(cg_b[0], cg_b[1])}"}
    await client.post("/v1/me", headers=headers_b, json={"display_name": "B"})

    files = {"photo": ("avatar.jpg", _TINY_JPEG, "image/jpeg")}
    r = await client.post(f"/v1/patients/{pid}/photo", headers=headers_b, files=files)
    assert r.status_code == 404
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_photo_url_returns_signed_url(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    # First upload so photo_storage_key is set.
    fake_httpx["responses"].append(httpx.Response(200))
    files = {"photo": ("avatar.jpg", _TINY_JPEG, "image/jpeg")}
    await client.post(f"/v1/patients/{pid}/photo", headers=headers, files=files)

    # Then ask for a URL.
    fake_httpx["responses"].append(
        httpx.Response(
            200,
            json={"signedURL": f"/object/sign/patient-photos/{pid}.jpg?token=t"},
        )
    )
    r = await client.get(f"/v1/patients/{pid}/photo-url", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["url"].endswith(f"/patient-photos/{pid}.jpg?token=t")
    assert body["expires_in_seconds"] == 600


@pytest.mark.asyncio
async def test_photo_url_returns_404_when_no_photo(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    r = await client.get(f"/v1/patients/{pid}/photo-url", headers=headers)
    assert r.status_code == 404
    assert r.json() == {"detail": "photo_not_found"}
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_delete_photo_clears_key_and_calls_storage(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    # Upload first.
    fake_httpx["responses"].append(httpx.Response(200))
    files = {"photo": ("avatar.jpg", _TINY_JPEG, "image/jpeg")}
    await client.post(f"/v1/patients/{pid}/photo", headers=headers, files=files)

    # Delete should clear the key + DELETE the storage object.
    fake_httpx["responses"].append(httpx.Response(200))
    r = await client.delete(f"/v1/patients/{pid}/photo", headers=headers)
    assert r.status_code == 204

    # Two captured calls: POST upload, DELETE removal.
    methods = [c["method"] for c in fake_httpx["captured"]]
    assert methods == ["POST", "DELETE"]
    delete_call = fake_httpx["captured"][1]
    assert delete_call["url"].endswith(f"/patient-photos/{pid}.jpg")

    # The patient row now has photo_storage_key = null.
    g = await client.get(f"/v1/patients/{pid}", headers=headers)
    assert g.status_code == 200
    assert g.json()["photo_storage_key"] is None


@pytest.mark.asyncio
async def test_delete_photo_idempotent_when_none_set(
    client: AsyncClient, auth_user, fake_httpx,
):
    headers, pid = await _provision_with_patient(client, auth_user)
    r = await client.delete(f"/v1/patients/{pid}/photo", headers=headers)
    assert r.status_code == 204
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_requires_auth(client: AsyncClient) -> None:
    files = {"photo": ("avatar.jpg", _TINY_JPEG, "image/jpeg")}
    r = await client.post(f"/v1/patients/{uuid4()}/photo", files=files)
    assert r.status_code == 401
