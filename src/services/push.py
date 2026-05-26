"""APNs push dispatch for detector events.

Spawns one fire-and-forget HTTP/2 POST per registered device when an
event fires. Called from `CameraIngest._run_tick` after the events
row is committed (so a rolled-back tick doesn't leave a phantom push
trail).

Two layers:

  - `mint_provider_jwt(settings)` builds the APNs JWT — ES256 over
    `{iss: team_id, iat: now}` with the `kid` header set to the key
    ID. Apple rejects tokens older than ~60 min, so we cache + refresh
    every `apns_jwt_lifetime_seconds` (default 55 min).
  - `dispatch_event_pushes(sessionmaker, event_ids, settings)` resolves
    each event's caregiver via the patient row, fans out to that
    caregiver's `caregiver_devices`, and POSTs to APNs with HTTP/2.

The dispatcher noops when APNs isn't configured — lets local dev and
the test fixture skip APNs without touching the call site.

Connection multiplexing: a process-wide `httpx.AsyncClient(http2=True)`
is held in `_get_client()`. Disposed at lifespan shutdown via
`close()`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

import httpx
import jwt as pyjwt
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver_device import CaregiverDevice
from src.db.event import Event
from src.db.patient import Patient
from src.repos import caregiver_device_repo
from src.settings import Settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

log = logging.getLogger(__name__)

# APNs operates two distinct hosts. A token from one is rejected by
# the other; the caregiver app sends its environment in
# `RegisterDeviceRequest` and we route accordingly.
_APNS_PROD_HOST = "https://api.push.apple.com"
_APNS_SANDBOX_HOST = "https://api.sandbox.push.apple.com"

# Module-level JWT cache. APNs rejects JWTs older than 60 min; we
# refresh proactively at `settings.apns_jwt_lifetime_seconds`.
_jwt_cache: dict[str, str] = {}
_jwt_expires_at: dict[str, float] = {}

# Shared HTTP/2 client. Lazy on first dispatch so import doesn't open
# sockets.
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the process-wide HTTP/2 client used for all APNs POSTs."""
    global _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(http2=True, timeout=10.0)
        return _client


async def close() -> None:
    """Close the shared client. Called from FastAPI lifespan shutdown."""
    global _client
    async with _client_lock:
        if _client is not None:
            await _client.aclose()
            _client = None


def mint_provider_jwt(settings: Settings) -> str:
    """Return a cached + valid ES256 JWT for APNs provider auth.

    Cache key is `(team_id, key_id)` so re-keying via env change
    invalidates implicitly. Returns the empty string if APNs isn't
    configured — callers check this and skip the POST.
    """
    if not settings.apns_team_id or not settings.apns_key_id or not settings.apns_auth_key:
        return ""
    cache_key = f"{settings.apns_team_id}:{settings.apns_key_id}"
    now = time.time()
    expires_at = _jwt_expires_at.get(cache_key, 0.0)
    if cache_key in _jwt_cache and now < expires_at:
        return _jwt_cache[cache_key]

    issued_at = int(now)
    token = pyjwt.encode(
        {"iss": settings.apns_team_id, "iat": issued_at},
        settings.apns_auth_key,
        algorithm="ES256",
        headers={"kid": settings.apns_key_id, "typ": "JWT"},
    )
    _jwt_cache[cache_key] = token
    _jwt_expires_at[cache_key] = now + settings.apns_jwt_lifetime_seconds
    return token


def build_event_payload(*, title: str, body: str, event_id: str, patient_id: str) -> dict:
    """Construct the JSON body APNs hands to the caregiver app.

    `aps.alert` becomes the lock-screen alert; `aps.sound = "default"`
    plays the standard ringtone; `aps.thread-id` groups multiple
    pushes per patient. Custom keys at the root level are surfaced to
    SwiftUI via `UNNotificationContent.userInfo` for deep-linking.
    """
    return {
        "aps": {
            "alert": {"title": title, "body": body},
            "sound": "default",
            "thread-id": patient_id,
        },
        "event_id": event_id,
        "patient_id": patient_id,
    }


async def dispatch_event_pushes(
    *,
    sessionmaker: "async_sessionmaker",
    event_ids: Sequence[UUID],
    settings: Settings,
) -> None:
    """Resolve each event's caregiver + devices and fan out APNs POSTs.

    `event_ids` are UUIDs; we reload the rows in our own session so
    the caller's session can be closed safely before this task runs.

    Errors are logged but never raised — push delivery is best-effort
    by design; the events row already landed in the DB regardless.
    """
    if not event_ids:
        return
    jwt_token = mint_provider_jwt(settings)
    if not jwt_token:
        log.info("push.skip reason=apns_not_configured events=%d", len(event_ids))
        return

    client = await _get_client()
    async with sessionmaker() as session:
        for event_id in event_ids:
            try:
                await _dispatch_one(
                    session=session,
                    event_id=event_id,
                    jwt_token=jwt_token,
                    client=client,
                    settings=settings,
                )
            except Exception:  # noqa: BLE001
                log.exception("push.dispatch_failed event_id=%s", event_id)


async def _dispatch_one(
    *,
    session: AsyncSession,
    event_id: UUID,
    jwt_token: str,
    client: httpx.AsyncClient,
    settings: Settings,
) -> None:
    event = await session.get(Event, event_id)
    if event is None:
        log.warning("push.dispatch_skip reason=event_not_found id=%s", event_id)
        return
    patient = await session.get(Patient, event.patient_id)
    if patient is None:
        log.warning("push.dispatch_skip reason=patient_not_found event=%s", event.id)
        return
    devices = await caregiver_device_repo.find_by_caregiver(session, patient.caregiver_id)
    if not devices:
        log.info(
            "push.dispatch_skip reason=no_devices event=%s caregiver=%s",
            event.id, patient.caregiver_id,
        )
        return

    payload = build_event_payload(
        title=event.title,
        body=f"{patient.preferred_name} — {event.narrative}",
        event_id=str(event.id),
        patient_id=str(patient.id),
    )
    body = json.dumps(payload).encode("utf-8")
    for device in devices:
        await _send_to_device(
            device=device,
            session=session,
            body=body,
            jwt_token=jwt_token,
            client=client,
            settings=settings,
        )


async def _send_to_device(
    *,
    device: CaregiverDevice,
    session: AsyncSession,
    body: bytes,
    jwt_token: str,
    client: httpx.AsyncClient,
    settings: Settings,
) -> None:
    host = _APNS_PROD_HOST if device.environment == "production" else _APNS_SANDBOX_HOST
    url = f"{host}/3/device/{device.apns_token}"
    headers = {
        "authorization": f"bearer {jwt_token}",
        "apns-topic": settings.apns_bundle_id,
        "apns-push-type": "alert",
        "apns-priority": "10",
    }
    try:
        response = await client.post(url, content=body, headers=headers)
    except httpx.RequestError as exc:
        log.warning("push.transport_error device=%s err=%s", device.apns_token[:8], exc)
        return

    if response.status_code == 410:
        # APNs's signal that this token is dead. Clean up the row
        # so we don't keep paying transport for an unreachable device.
        await session.delete(device)
        await session.commit()
        log.info("push.token_gone deleted device id=%s", device.id)
        return
    if response.status_code >= 400:
        log.warning(
            "push.apns_error device=%s status=%d body=%s",
            device.apns_token[:8], response.status_code, response.text,
        )
        return
    log.info(
        "push.delivered event=ok device=%s status=%d",
        device.apns_token[:8], response.status_code,
    )
