"""Task 4 — B2 uploader wiring, with a stubbed S3 client (no network).

We don't exercise real Backblaze here — we assert the uploader passes
the right bucket/key/credentials and stringifies object metadata (S3
metadata values must be strings). A fake aioboto3-style session is
injected so nothing touches the network or needs real creds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.services.data_collection import b2


def _settings(**over: object) -> SimpleNamespace:
    base: dict[str, object] = dict(
        b2_key_id="kid",
        b2_app_key="akey",
        b2_bucket="memoria-data",
        b2_endpoint_url="https://s3.us-west-004.backblazeb2.com",
        b2_region="us-west-004",
    )
    base.update(over)
    return SimpleNamespace(**base)


class _FakeS3:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def upload_file(
        self, *, Filename: str, Bucket: str, Key: str, ExtraArgs: dict
    ) -> None:
        self.calls.append(
            dict(Filename=Filename, Bucket=Bucket, Key=Key, ExtraArgs=ExtraArgs)
        )


class _FakeClientCM:
    def __init__(self, s3: _FakeS3) -> None:
        self._s3 = s3

    async def __aenter__(self) -> _FakeS3:
        return self._s3

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    def __init__(self, s3: _FakeS3) -> None:
        self._s3 = s3
        self.service: str | None = None
        self.client_kwargs: dict | None = None

    def client(self, service: str, **kwargs: object) -> _FakeClientCM:
        self.service = service
        self.client_kwargs = kwargs
        return _FakeClientCM(self._s3)


@pytest.mark.asyncio
async def test_upload_passes_bucket_key_creds_and_string_metadata(tmp_path) -> None:
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"\x00\x01\x02")
    s3 = _FakeS3()
    session = _FakeSession(s3)
    key = "data/v1/c/p/2026-06-24/090503Z__den.mp4"

    returned = await b2.upload_file_to_b2(
        f,
        key,
        metadata={"patient_id": "p1", "confidence": 0.83, "gate": "poi"},
        settings=_settings(),
        session=session,
    )

    assert returned == key
    assert session.service == "s3"
    assert session.client_kwargs["endpoint_url"] == (
        "https://s3.us-west-004.backblazeb2.com"
    )
    assert session.client_kwargs["region_name"] == "us-west-004"
    assert session.client_kwargs["aws_access_key_id"] == "kid"
    assert session.client_kwargs["aws_secret_access_key"] == "akey"

    assert len(s3.calls) == 1
    call = s3.calls[0]
    assert call["Filename"] == str(f)
    assert call["Bucket"] == "memoria-data"
    assert call["Key"] == key
    assert call["ExtraArgs"]["ContentType"] == "video/mp4"
    # S3 object metadata values MUST be strings — confidence float → str.
    assert call["ExtraArgs"]["Metadata"] == {
        "patient_id": "p1",
        "confidence": "0.83",
        "gate": "poi",
    }


def test_b2_configured_true_when_creds_present() -> None:
    assert b2.b2_configured(_settings()) is True


def test_b2_configured_false_when_key_id_missing() -> None:
    assert b2.b2_configured(_settings(b2_key_id="")) is False


def test_b2_configured_false_when_app_key_missing() -> None:
    assert b2.b2_configured(_settings(b2_app_key="")) is False
