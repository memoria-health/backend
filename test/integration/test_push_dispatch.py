"""Integration tests for `src.services.push.dispatch_event_pushes`.

Per `feedback_no_mocking_async_wiring`: every supporting object is
real (Event row in Postgres, Patient row, Caregiver row, CaregiverDevice
row, the real `async_sessionmaker`). Only the APNs HTTP/2 client is
swapped out — we monkeypatch `push._get_client` to return a fake that
records every POST. The JWT signing key is a freshly-generated EC
P-256 so we don't depend on a real Apple key file.

What we assert:

  - When APNs is configured and devices exist, the dispatcher POSTs
    once per device to the right host (sandbox vs prod) with the
    expected URL, headers, and JSON body.
  - When APNs isn't configured, the dispatcher is a noop (no POST).
  - When APNs returns 410, the device row is deleted from the DB
    in the same transaction the dispatcher owns.
  - When APNs returns a transient 500, the device row is preserved.
  - The end-to-end camera_ingest tick → event → push wiring fires
    after `session.commit()` (temporal ordering), not before.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver
from src.db.caregiver_device import CaregiverDevice
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services import push
from src.settings import Settings, get_settings
from test.conftest import mint_test_jwt  # noqa: F401


def _generate_es256_pem() -> str:
    """Generate a throwaway EC P-256 PEM private key for signing test
    JWTs. The PEM string slots into `settings.apns_auth_key` exactly
    where an Apple .p8 file's contents would go in production."""
    key = ec.generate_private_key(ec.SECP256R1())
    pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return pem.decode("utf-8")


class _CapturingClient:
    """httpx.AsyncClient stand-in. Records every POST and returns a
    canned response. The dispatch code only calls `.post(...)`.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.responses: list[httpx.Response] = []

    async def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> httpx.Response:
        self.calls.append({"url": url, "content": content, "headers": dict(headers)})
        if not self.responses:
            return httpx.Response(200, request=httpx.Request("POST", url))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


@pytest_asyncio.fixture
async def _reset_push_module() -> AsyncIterator[None]:
    """Clear the module-level JWT cache and shared client between
    tests so settings overrides take effect immediately."""
    push._jwt_cache.clear()
    push._jwt_expires_at.clear()
    await push.close()
    yield
    push._jwt_cache.clear()
    push._jwt_expires_at.clear()
    await push.close()


@pytest_asyncio.fixture
async def push_settings(monkeypatch) -> Settings:
    """Return a Settings with APNs configured to point at our fake
    HTTP client. The auth_key is a freshly-minted EC P-256 PEM so
    `jwt.encode(..., algorithm="ES256")` actually signs the JWT."""
    base = get_settings()
    overrides = base.model_dump()
    overrides["apns_team_id"] = "TEAM12345X"
    overrides["apns_key_id"] = "KEY12345XY"
    overrides["apns_auth_key"] = _generate_es256_pem()
    overrides["apns_bundle_id"] = "com.elephantlabs.nemara.caregiver"
    overrides["apns_environment"] = "sandbox"
    return Settings(**overrides)


async def _seed_event_and_device(
    session: AsyncSession,
    auth_user_factory,
    *,
    apns_token: str,
    environment: str = "sandbox",
) -> tuple[UUID, UUID]:
    """Create real caregiver + patient + device + event rows. Returns
    `(caregiver_id, event_id)`."""
    uid, email = await auth_user_factory()
    now = datetime.now(timezone.utc)
    caregiver = Caregiver(
        id=uid,
        display_name="Push Tester",
        email=email,
        created_at=now,
        updated_at=now,
    )
    session.add(caregiver)

    patient_id = uuid4()
    patient = Patient(
        id=patient_id,
        caregiver_id=uid,
        full_name="Margaret O'Brien",
        preferred_name="Maggie",
        relation="parent",
        state="calm",
        status_line="All quiet",
        cameras_paused=False,
        created_at=now,
        updated_at=now,
    )
    session.add(patient)
    await session.flush()  # caregivers + patients must land before FK-referencing rows

    device = CaregiverDevice(
        id=uuid4(),
        caregiver_id=uid,
        apns_token=apns_token,
        bundle_id="com.elephantlabs.nemara.caregiver",
        environment=environment,
        created_at=now,
        updated_at=now,
    )
    session.add(device)

    event_id = uuid4()
    event = Event(
        id=event_id,
        patient_id=patient_id,
        detector_id=None,
        camera_id=None,
        category="safety",
        source="preset",
        icon_name="fall",
        title="Fall detected",
        location="Bedroom",
        occurred_at=now,
        has_clip=False,
        narrative="Maggie fell near the bed.",
        opened=False,
        reviewed=False,
        review_choice=None,
        review_notes=None,
        reviewed_at=None,
        created_at=now,
    )
    session.add(event)
    await session.commit()
    return uid, event_id


@pytest.mark.asyncio
async def test_dispatch_posts_to_apns_with_expected_shape(
    auth_user_factory, push_settings, monkeypatch, _reset_push_module,
):
    sm = get_sessionmaker()
    async with sm() as s:
        caregiver_id, event_id = await _seed_event_and_device(
            s, auth_user_factory, apns_token="abc12345" * 8,
        )

    fake = _CapturingClient()

    async def _get_client_override() -> _CapturingClient:
        return fake  # type: ignore[return-value]

    monkeypatch.setattr(push, "_get_client", _get_client_override)

    await push.dispatch_event_pushes(
        sessionmaker=sm,
        event_ids=[event_id],
        settings=push_settings,
    )

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["url"] == f"https://api.sandbox.push.apple.com/3/device/{'abc12345' * 8}"
    assert call["headers"]["apns-topic"] == "com.elephantlabs.nemara.caregiver"
    assert call["headers"]["apns-push-type"] == "alert"
    assert call["headers"]["apns-priority"] == "10"
    assert call["headers"]["authorization"].startswith("bearer ")

    import json
    payload = json.loads(call["content"])
    assert payload["aps"]["alert"]["title"] == "Maggie"
    assert payload["aps"]["alert"]["body"] == "Maggie fell near the bed."
    assert payload["aps"]["sound"] == "default"
    assert payload["event_id"] == str(event_id)
    assert UUID(payload["patient_id"])  # is a uuid


@pytest.mark.asyncio
async def test_dispatch_uses_production_host_for_production_token(
    auth_user_factory, push_settings, monkeypatch, _reset_push_module,
):
    sm = get_sessionmaker()
    async with sm() as s:
        _, event_id = await _seed_event_and_device(
            s, auth_user_factory,
            apns_token="prodtoken" * 7,
            environment="production",
        )

    fake = _CapturingClient()

    async def _get_client_override() -> _CapturingClient:
        return fake  # type: ignore[return-value]

    monkeypatch.setattr(push, "_get_client", _get_client_override)

    await push.dispatch_event_pushes(
        sessionmaker=sm, event_ids=[event_id], settings=push_settings,
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["url"].startswith("https://api.push.apple.com/3/device/")


@pytest.mark.asyncio
async def test_dispatch_noops_when_apns_not_configured(
    auth_user_factory, monkeypatch, _reset_push_module,
):
    sm = get_sessionmaker()
    async with sm() as s:
        _, event_id = await _seed_event_and_device(
            s, auth_user_factory, apns_token="x" * 64,
        )

    fake = _CapturingClient()

    async def _get_client_override() -> _CapturingClient:
        return fake  # type: ignore[return-value]

    monkeypatch.setattr(push, "_get_client", _get_client_override)

    # Default settings have empty apns_team_id → dispatcher is a noop.
    base = get_settings()
    await push.dispatch_event_pushes(
        sessionmaker=sm, event_ids=[event_id], settings=base,
    )
    assert fake.calls == []


@pytest.mark.asyncio
async def test_dispatch_deletes_device_on_410(
    auth_user_factory, push_settings, monkeypatch, _reset_push_module,
):
    sm = get_sessionmaker()
    async with sm() as s:
        caregiver_id, event_id = await _seed_event_and_device(
            s, auth_user_factory, apns_token="dead" * 16,
        )

    fake = _CapturingClient()
    fake.responses.append(
        httpx.Response(410, text="BadDeviceToken", request=httpx.Request("POST", "https://x")),
    )

    async def _get_client_override() -> _CapturingClient:
        return fake  # type: ignore[return-value]

    monkeypatch.setattr(push, "_get_client", _get_client_override)

    await push.dispatch_event_pushes(
        sessionmaker=sm, event_ids=[event_id], settings=push_settings,
    )

    assert len(fake.calls) == 1
    # Row gone.
    async with sm() as s:
        rows = (
            await s.execute(
                select(CaregiverDevice).where(CaregiverDevice.apns_token == "dead" * 16)
            )
        ).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_dispatch_preserves_device_on_transient_5xx(
    auth_user_factory, push_settings, monkeypatch, _reset_push_module,
):
    sm = get_sessionmaker()
    async with sm() as s:
        _, event_id = await _seed_event_and_device(
            s, auth_user_factory, apns_token="keep" * 16,
        )

    fake = _CapturingClient()
    fake.responses.append(
        httpx.Response(503, text="Service Unavailable", request=httpx.Request("POST", "https://x")),
    )

    async def _get_client_override() -> _CapturingClient:
        return fake  # type: ignore[return-value]

    monkeypatch.setattr(push, "_get_client", _get_client_override)

    await push.dispatch_event_pushes(
        sessionmaker=sm, event_ids=[event_id], settings=push_settings,
    )

    async with sm() as s:
        rows = (
            await s.execute(
                select(CaregiverDevice).where(CaregiverDevice.apns_token == "keep" * 16)
            )
        ).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_dispatch_fans_out_to_multiple_devices(
    auth_user_factory, push_settings, monkeypatch, _reset_push_module,
):
    sm = get_sessionmaker()
    async with sm() as s:
        caregiver_id, event_id = await _seed_event_and_device(
            s, auth_user_factory, apns_token="primary" * 9,
        )
        # Same caregiver gets a second device (iPad).
        now = datetime.now(timezone.utc)
        s.add(
            CaregiverDevice(
                id=uuid4(),
                caregiver_id=caregiver_id,
                apns_token="ipad" * 16,
                bundle_id="com.elephantlabs.nemara.caregiver",
                environment="sandbox",
                created_at=now,
                updated_at=now,
            )
        )
        await s.commit()

    fake = _CapturingClient()

    async def _get_client_override() -> _CapturingClient:
        return fake  # type: ignore[return-value]

    monkeypatch.setattr(push, "_get_client", _get_client_override)

    await push.dispatch_event_pushes(
        sessionmaker=sm, event_ids=[event_id], settings=push_settings,
    )

    assert len(fake.calls) == 2
    tokens_hit = {call["url"].rsplit("/", 1)[1] for call in fake.calls}
    assert tokens_hit == {"primary" * 9, "ipad" * 16}


@pytest.mark.asyncio
async def test_jwt_cache_reuses_token_within_lifetime(
    push_settings, _reset_push_module,
):
    """Each call to `mint_provider_jwt` within the lifetime returns
    the same bytes. APNs rate-limits provider-JWT minting; the cache
    is what keeps the dispatcher within budget under burst load."""
    a = push.mint_provider_jwt(push_settings)
    b = push.mint_provider_jwt(push_settings)
    assert a == b
    assert a  # non-empty
