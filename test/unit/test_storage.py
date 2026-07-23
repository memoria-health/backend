"""Unit tests for `src.services.storage`.

The Supabase Storage REST surface is mocked through a fake
`httpx.AsyncClient` injected via monkeypatch. We assert exactly what
the storage layer sends on the wire (URL, headers, body bytes, JSON
shape) — that contract is what production failures would surface in,
and the layer above (api/services) trusts it.

Validation tests (empty / oversized / wrong content type) exercise the
pre-flight checks without ever needing the fake client.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx
import pytest

from src.errors import StorageError
from src.services import storage as storage_module
from src.services.storage import (
    delete_object,
    download_object,
    signed_url,
    upload_clip,
    upload_patient_photo,
)
from src.settings import Settings


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "db_url": "postgresql://x",
        "supabase_url": "https://example.supabase.co",
        "supabase_publishable_key": "anon",
        "supabase_jwt_secret": "jwtsecret",
        "supabase_service_role_key": "service-role-key",
    }
    base.update(overrides)
    return Settings(**base)


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in. One instance per `async
    with` block; calls record into the shared `state` dict so tests
    can assert.
    """

    state: dict[str, Any] = {"responses": [], "captured": [], "timeouts": []}

    def __init__(self, *, timeout: float | None = None, **_kw: Any) -> None:
        self._timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        type(self).state["timeouts"].append(self._timeout)
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None

    async def post(self, url: str, **kw: Any) -> httpx.Response:
        return self._record_and_return("POST", url, kw)

    async def get(self, url: str, **kw: Any) -> httpx.Response:
        return self._record_and_return("GET", url, kw)

    async def delete(self, url: str, **kw: Any) -> httpx.Response:
        return self._record_and_return("DELETE", url, kw)

    @classmethod
    def _record_and_return(cls, method: str, url: str, kw: dict[str, Any]) -> httpx.Response:
        cls.state["captured"].append({"method": method, "url": url, "kw": kw})
        if not cls.state["responses"]:
            raise RuntimeError(
                f"no canned response queued for {method} {url}; the test "
                "needs to push one into state['responses']"
            )
        return cls.state["responses"].pop(0)


@pytest.fixture
def fake_httpx(monkeypatch):
    state: dict[str, Any] = {"responses": [], "captured": [], "timeouts": []}
    _FakeAsyncClient.state = state
    monkeypatch.setattr(storage_module.httpx, "AsyncClient", _FakeAsyncClient)
    return state


# ── upload_patient_photo ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_patient_photo_sends_expected_request(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200, text="ok"))
    patient_id = uuid4()
    settings = _settings()
    body = b"\xff\xd8\xff\xe0" + b"\x00" * 1024  # fake JPEG

    key = await upload_patient_photo(
        patient_id=patient_id,
        image_bytes=body,
        content_type="image/jpeg",
        settings=settings,
    )

    assert key == f"{patient_id}.jpg"
    assert len(fake_httpx["captured"]) == 1
    call = fake_httpx["captured"][0]
    assert call["method"] == "POST"
    assert call["url"] == (
        f"https://example.supabase.co/storage/v1/object/patient-photos/{patient_id}.jpg"
    )
    assert call["kw"]["content"] == body
    headers = call["kw"]["headers"]
    assert headers["Authorization"] == "Bearer service-role-key"
    assert headers["Content-Type"] == "image/jpeg"
    assert headers["x-upsert"] == "true"
    assert fake_httpx["timeouts"] == [settings.storage_upload_timeout_seconds]


@pytest.mark.asyncio
async def test_upload_patient_photo_png_extension(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200))
    patient_id = uuid4()
    key = await upload_patient_photo(
        patient_id=patient_id,
        image_bytes=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,
        content_type="image/png",
        settings=_settings(),
    )
    assert key == f"{patient_id}.png"
    assert "patient-photos" in fake_httpx["captured"][0]["url"]


@pytest.mark.asyncio
async def test_upload_patient_photo_in_region_is_prefixed(fake_httpx) -> None:
    # Data-residency: an India patient's avatar key gets an `IN/` prefix so
    # the object is isolated + liftable. The stored key (returned here) is
    # reused verbatim for read/sign/delete, so the prefix must be in it.
    fake_httpx["responses"].append(httpx.Response(200, text="ok"))
    patient_id = uuid4()
    key = await upload_patient_photo(
        patient_id=patient_id,
        image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 1024,
        content_type="image/jpeg",
        settings=_settings(),
        region="IN",
    )
    assert key == f"IN/{patient_id}.jpg"
    assert (
        fake_httpx["captured"][0]["url"]
        == f"https://example.supabase.co/storage/v1/object/patient-photos/IN/{patient_id}.jpg"
    )


@pytest.mark.asyncio
async def test_upload_clip_in_region_is_prefixed(fake_httpx) -> None:
    # Same for incident clips: IN → `IN/events/...`; US stays unprefixed.
    fake_httpx["responses"].append(httpx.Response(200))
    event_id = uuid4()
    key = await storage_module.upload_clip(
        event_id=event_id,
        mp4_bytes=b"\x00" * 256,
        settings=_settings(),
        region="IN",
    )
    assert key == f"IN/events/{event_id}.mp4"
    assert f"/incident-clips/IN/events/{event_id}.mp4" in fake_httpx["captured"][0]["url"]


@pytest.mark.asyncio
async def test_upload_patient_photo_us_region_is_unprefixed(fake_httpx) -> None:
    # US path must be unchanged (no prefix, no migration).
    fake_httpx["responses"].append(httpx.Response(200))
    patient_id = uuid4()
    key = await upload_patient_photo(
        patient_id=patient_id,
        image_bytes=b"\xff\xd8\xff\xe0" + b"\x00" * 1024,
        content_type="image/jpeg",
        settings=_settings(),
        region="US",
    )
    assert key == f"{patient_id}.jpg"


@pytest.mark.asyncio
async def test_upload_patient_photo_rejects_unsupported_content_type(fake_httpx) -> None:
    with pytest.raises(StorageError, match="unsupported content type 'image/gif'"):
        await upload_patient_photo(
            patient_id=uuid4(),
            image_bytes=b"GIF89a" + b"\x00" * 10,
            content_type="image/gif",
            settings=_settings(),
        )
    # No HTTP call made.
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_patient_photo_rejects_empty_bytes(fake_httpx) -> None:
    with pytest.raises(StorageError, match="image_bytes is empty"):
        await upload_patient_photo(
            patient_id=uuid4(),
            image_bytes=b"",
            content_type="image/jpeg",
            settings=_settings(),
        )
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_patient_photo_rejects_oversized(fake_httpx) -> None:
    settings = _settings(storage_max_photo_bytes=2048)
    with pytest.raises(StorageError, match=r"too large: 4000 bytes \(max 2048\)"):
        await upload_patient_photo(
            patient_id=uuid4(),
            image_bytes=b"\xff\xd8" + b"\x00" * 3998,
            content_type="image/jpeg",
            settings=settings,
        )
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_patient_photo_surfaces_provider_error(fake_httpx) -> None:
    fake_httpx["responses"].append(
        httpx.Response(503, text="Service Unavailable", request=httpx.Request("POST", "https://example.supabase.co/x"))
    )
    with pytest.raises(StorageError, match="upload failed: 503"):
        await upload_patient_photo(
            patient_id=uuid4(),
            image_bytes=b"\xff\xd8" + b"\x00" * 10,
            content_type="image/jpeg",
            settings=_settings(),
        )


# ── upload_clip ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_clip_default_key_and_long_timeout(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200))
    settings = _settings()
    event_id = uuid4()
    mp4 = b"\x00\x00\x00\x18ftyp" + b"\x00" * 10_000

    key = await upload_clip(event_id=event_id, mp4_bytes=mp4, settings=settings)

    assert key == f"events/{event_id}.mp4"
    assert len(fake_httpx["captured"]) == 1
    call = fake_httpx["captured"][0]
    assert call["method"] == "POST"
    assert call["url"] == (
        f"https://example.supabase.co/storage/v1/object/incident-clips/{key}"
    )
    assert call["kw"]["content"] == mp4
    assert call["kw"]["headers"]["Content-Type"] == "video/mp4"
    assert call["kw"]["headers"]["x-upsert"] == "true"
    # Clips must use the longer clip-timeout, not the photo timeout.
    assert fake_httpx["timeouts"] == [settings.storage_clip_upload_timeout_seconds]


@pytest.mark.asyncio
async def test_upload_clip_sub_key_namespaces_intermediates(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200))
    event_id = uuid4()
    key = await upload_clip(
        event_id=event_id,
        mp4_bytes=b"x" * 100,
        settings=_settings(),
        sub_key="pre.mp4",
    )
    assert key == f"events/{event_id}/pre.mp4"
    assert fake_httpx["captured"][0]["url"].endswith(key)


@pytest.mark.asyncio
async def test_upload_clip_rejects_empty(fake_httpx) -> None:
    with pytest.raises(StorageError, match="mp4_bytes is empty"):
        await upload_clip(event_id=uuid4(), mp4_bytes=b"", settings=_settings())
    assert fake_httpx["captured"] == []


@pytest.mark.asyncio
async def test_upload_clip_rejects_oversized(fake_httpx) -> None:
    settings = _settings(storage_max_clip_bytes=10_000)
    with pytest.raises(StorageError, match=r"too large: 20000 bytes \(max 10000\)"):
        await upload_clip(
            event_id=uuid4(),
            mp4_bytes=b"x" * 20_000,
            settings=settings,
        )
    assert fake_httpx["captured"] == []


# ── signed_url ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_signed_url_builds_absolute_url(fake_httpx) -> None:
    fake_httpx["responses"].append(
        httpx.Response(200, json={"signedURL": "/object/sign/incident-clips/events/x.mp4?token=abc"}),
    )
    settings = _settings()
    url = await signed_url(
        bucket="incident-clips",
        key="events/x.mp4",
        settings=settings,
    )
    assert url == (
        "https://example.supabase.co/storage/v1/object/sign/incident-clips/events/x.mp4?token=abc"
    )
    call = fake_httpx["captured"][0]
    assert call["method"] == "POST"
    assert call["url"] == (
        "https://example.supabase.co/storage/v1/object/sign/incident-clips/events/x.mp4"
    )
    assert call["kw"]["json"] == {
        "expiresIn": settings.storage_signed_url_default_ttl_seconds,
    }


@pytest.mark.asyncio
async def test_signed_url_honors_explicit_ttl(fake_httpx) -> None:
    fake_httpx["responses"].append(
        httpx.Response(200, json={"signedURL": "/object/sign/patient-photos/x.jpg?token=t"}),
    )
    await signed_url(
        bucket="patient-photos",
        key="x.jpg",
        settings=_settings(),
        ttl_seconds=120,
    )
    assert fake_httpx["captured"][0]["kw"]["json"] == {"expiresIn": 120}


@pytest.mark.asyncio
async def test_signed_url_raises_on_missing_signedurl(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200, json={}))
    with pytest.raises(StorageError, match="missing signedURL"):
        await signed_url(bucket="b", key="k", settings=_settings())


# ── delete_object ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_object_sends_expected_request(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200))
    await delete_object(bucket="patient-photos", key="abc.jpg", settings=_settings())
    call = fake_httpx["captured"][0]
    assert call["method"] == "DELETE"
    assert call["url"] == "https://example.supabase.co/storage/v1/object/patient-photos/abc.jpg"
    assert call["kw"]["headers"]["Authorization"] == "Bearer service-role-key"


@pytest.mark.asyncio
async def test_delete_object_treats_404_as_success(fake_httpx) -> None:
    """The caller's intent is "make sure this object is not there".
    A 404 satisfies that — surfacing it as an error would force every
    deletion callsite to catch-and-ignore."""
    fake_httpx["responses"].append(
        httpx.Response(404, text="Not Found", request=httpx.Request("DELETE", "https://x")),
    )
    # No exception; returns None.
    result = await delete_object(bucket="b", key="k", settings=_settings())
    assert result is None


@pytest.mark.asyncio
async def test_delete_object_surfaces_non_404_error(fake_httpx) -> None:
    fake_httpx["responses"].append(
        httpx.Response(500, text="boom", request=httpx.Request("DELETE", "https://x")),
    )
    with pytest.raises(StorageError, match="delete failed: 500"):
        await delete_object(bucket="b", key="k", settings=_settings())


# ── download_object ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_download_object_returns_response_bytes(fake_httpx) -> None:
    fake_httpx["responses"].append(httpx.Response(200, content=b"hello-mp4"))
    blob = await download_object(bucket="b", key="k", settings=_settings())
    assert blob == b"hello-mp4"
    call = fake_httpx["captured"][0]
    assert call["method"] == "GET"
    assert call["url"] == "https://example.supabase.co/storage/v1/object/b/k"
    assert call["kw"]["headers"]["Authorization"] == "Bearer service-role-key"
