"""Email alert dispatch via Resend.

Companion to `src/services/push.py`. Where push fans APNs out to a
caregiver's iOS devices, this fans an email out to the caregiver's
account address + any additional recipients they've configured on the
web Settings page. Called from `CameraIngest._run_tick` right after
the events row commits, same fire-and-forget pattern as push.

No-ops cleanly when `settings.resend_api_key` is empty (local dev /
tests / anyone who hasn't wired Resend) — logs a single info line and
returns. Delivery is best-effort: errors are logged, never raised, so
a flaky email provider can't wedge the detector tick or lose the
events row (which is already committed).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING
from uuid import UUID

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver
from src.db.event import Event
from src.db.patient import Patient
from src.settings import Settings

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

log = logging.getLogger(__name__)

_RESEND_ENDPOINT = "https://api.resend.com/emails"

# Shared client, lazy on first send so import opens no sockets.
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    global _client
    async with _client_lock:
        if _client is None:
            _client = httpx.AsyncClient(timeout=10.0)
        return _client


async def close() -> None:
    """Close the shared client. Called from FastAPI lifespan shutdown."""
    global _client
    async with _client_lock:
        if _client is not None:
            await _client.aclose()
            _client = None


def _recipients_for(patient: Patient, caregiver: Caregiver) -> list[str]:
    """Owning caregiver's account email + this patient's configured
    extras, de-duped, order-stable.

    The account email always leads so the caregiver themselves is
    never accidentally dropped by an empty/edited extras list. Extras
    are per-patient (different family/aides per patient).
    """
    seen: set[str] = set()
    out: list[str] = []
    for addr in [caregiver.email, *(patient.email_alert_recipients or [])]:
        norm = (addr or "").strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            out.append(addr)
    return out


async def _send(
    *,
    to: Sequence[str],
    subject: str,
    html: str,
    settings: Settings,
) -> None:
    """One Resend POST. Raises on transport error; caller logs."""
    client = await _get_client()
    resp = await client.post(
        _RESEND_ENDPOINT,
        headers={"Authorization": f"Bearer {settings.resend_api_key}"},
        json={
            "from": settings.email_from,
            "to": list(to),
            "subject": subject,
            "html": html,
        },
    )
    if resp.status_code >= 400:
        # Surface Resend's error body so domain-not-verified etc. is
        # obvious in the logs.
        log.warning(
            "email.resend_rejected status=%d body=%s", resp.status_code, resp.text[:500],
        )


def _event_html(*, patient_name: str, title: str, narrative: str, location: str) -> str:
    return (
        f"<div style=\"font-family:-apple-system,Segoe UI,sans-serif;color:#1c1a17\">"
        f"<h2 style=\"margin:0 0 4px\">{title}</h2>"
        f"<p style=\"margin:0 0 12px;color:#6b6357;font-size:13px\">"
        f"{patient_name} · {location}</p>"
        f"<p style=\"margin:0 0 16px;font-size:15px;line-height:1.5\">{narrative}</p>"
        f"<p style=\"margin:0;color:#6b6357;font-size:12px\">"
        f"Sent by Nemara because you enabled email alerts. "
        f"Manage alerts in your dashboard settings.</p>"
        f"</div>"
    )


async def dispatch_event_emails(
    *,
    sessionmaker: "async_sessionmaker",
    event_ids: Sequence[UUID],
    settings: Settings,
) -> None:
    """Resolve each event's caregiver + send an alert email if they've
    opted in. Reloads rows in its own session so the caller's session
    can close before this background task runs.
    """
    if not event_ids:
        return
    if not settings.resend_api_key:
        log.info("email.skip reason=resend_not_configured events=%d", len(event_ids))
        return

    async with sessionmaker() as session:
        for event_id in event_ids:
            try:
                await _dispatch_one(session=session, event_id=event_id, settings=settings)
            except Exception:  # noqa: BLE001
                log.exception("email.dispatch_failed event_id=%s", event_id)


async def _dispatch_one(
    *,
    session: AsyncSession,
    event_id: UUID,
    settings: Settings,
) -> None:
    event = await session.get(Event, event_id)
    if event is None:
        log.warning("email.dispatch_skip reason=event_not_found id=%s", event_id)
        return
    patient = await session.get(Patient, event.patient_id)
    if patient is None:
        log.warning("email.dispatch_skip reason=patient_not_found event=%s", event.id)
        return
    # Per-patient gate: this patient must have email alerts on.
    if not patient.email_alerts_enabled:
        return
    caregiver = await session.get(Caregiver, patient.caregiver_id)
    if caregiver is None:
        log.warning("email.dispatch_skip reason=caregiver_not_found event=%s", event.id)
        return

    recipients = _recipients_for(patient, caregiver)
    if not recipients:
        return

    patient_name = patient.preferred_name or patient.full_name
    await _send(
        to=recipients,
        subject=f"{event.title} — {patient_name}",
        html=_event_html(
            patient_name=patient_name,
            title=event.title,
            narrative=event.narrative,
            location=event.location,
        ),
        settings=settings,
    )
    log.info(
        "email.sent event=%s caregiver=%s recipients=%d",
        event.id, caregiver.id, len(recipients),
    )


async def send_test_email(
    *, patient: Patient, caregiver: Caregiver, settings: Settings,
) -> str:
    """Send a one-off test email to this patient's configured
    recipients (owning caregiver's account email + extras). Returns a
    human-readable status for the API to echo. Used by
    POST /v1/patients/{id}/test-email from the per-patient settings tab.
    """
    if not settings.resend_api_key:
        return "Email isn't configured on the server yet (no Resend key)."
    recipients = _recipients_for(patient, caregiver)
    if not recipients:
        return "No recipients to send to."
    patient_name = patient.preferred_name or patient.full_name
    await _send(
        to=recipients,
        subject=f"Nemara test alert — {patient_name}",
        html=_event_html(
            patient_name=patient_name,
            title="Test alert",
            narrative=(
                "This is a test email from Nemara. If you received it, email "
                f"alerts for {patient_name} are set up correctly."
            ),
            location="Living room",
        ),
        settings=settings,
    )
    return f"Test email sent to {', '.join(recipients)}."
