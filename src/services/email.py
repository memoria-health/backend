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
import base64
import binascii
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
    reply_to: str | None = None,
    from_email: str | None = None,
    attachments: Sequence[dict[str, str]] | None = None,
) -> None:
    """One Resend POST. Raises on transport error; caller logs.

    `from_email` overrides the default `settings.email_from` sender —
    used by the feedback path so feedback comes from its own address.

    `attachments` is Resend's attachment array — each item
    `{"filename": ..., "content": <base64>}`. Used by the feedback path
    to forward caregiver screenshots.
    """
    client = await _get_client()
    payload: dict[str, object] = {
        "from": from_email or settings.email_from,
        "to": list(to),
        "subject": subject,
        "html": html,
    }
    if reply_to:
        payload["reply_to"] = reply_to
    if attachments:
        payload["attachments"] = list(attachments)
    resp = await client.post(
        _RESEND_ENDPOINT,
        headers={"Authorization": f"Bearer {settings.resend_api_key}"},
        json=payload,
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
        f"Sent by Memoria because you enabled email alerts. "
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


async def send_support_email(
    *,
    caregiver: Caregiver,
    subject: str,
    message: str,
    settings: Settings,
) -> str:
    """Forward a Contact Support message from the iOS app to the team.

    Recipients come from `settings.support_email_recipients`; replyTo
    is the caregiver's account email so hitting reply in any mail
    client routes directly back to them. Returns a human-readable
    status the API echoes to the client.
    """
    if not settings.resend_api_key:
        return "Email isn't configured on the server yet (no Resend key)."
    recipients = list(settings.support_email_recipients or [])
    if not recipients:
        return "No support recipients are configured."
    full_subject = f"[Memoria support] {subject.strip() or '(no subject)'}"
    body_html = _support_html(
        caregiver_name=caregiver.display_name,
        caregiver_email=caregiver.email,
        caregiver_phone=caregiver.phone,
        message=message,
    )
    await _send(
        to=recipients,
        subject=full_subject,
        html=body_html,
        settings=settings,
        reply_to=caregiver.email,
    )
    log.info(
        "email.support_sent caregiver=%s recipients=%d", caregiver.id, len(recipients),
    )
    return "Sent. We'll reply to your account email."


def _support_html(
    *,
    caregiver_name: str,
    caregiver_email: str,
    caregiver_phone: str | None,
    message: str,
) -> str:
    safe = message.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    # Tap-to-text line: on iOS Mail the sms: link opens Messages addressed to
    # the caregiver, so a founder can reply to their phone in one tap.
    phone_line = ""
    if caregiver_phone:
        phone_line = (
            "<p style=\"margin:0 0 12px;font-size:15px\">"
            f"\U0001F4F1 Text back: <a href=\"sms:{caregiver_phone}\">{caregiver_phone}</a>"
            "</p>"
        )
    return (
        "<div style=\"font-family:-apple-system,Segoe UI,sans-serif;color:#1c1a17\">"
        f"<p style=\"margin:0 0 12px;color:#6b6357;font-size:13px\">"
        f"From {caregiver_name} &lt;{caregiver_email}&gt; via the Memoria iOS app."
        "</p>"
        f"{phone_line}"
        f"<div style=\"margin:0;font-size:15px;line-height:1.5\">{safe}</div>"
        "</div>"
    )


async def send_feedback_email(
    *,
    caregiver: Caregiver,
    rating: int | None,
    whats_working: str | None,
    whats_frustrating: str | None,
    category: str | None,
    settings: Settings,
    attachments: Sequence[str] | None = None,
) -> str:
    """Forward caregiver product feedback to the founders.

    Recipients come from `settings.feedback_email_recipients`; the
    sender is `settings.feedback_email_from`; replyTo is the caregiver's
    account email so hitting reply lands directly back with them.

    `attachments` are base64-encoded JPEG screenshots from the feedback
    form — forwarded as email attachments, never persisted. A screenshot
    that fails to decode is skipped (logged) so one bad image can't drop
    the whole notification.

    This is best-effort from the caller's perspective — the
    /v1/feedback handler has already committed the feedback row before
    calling here and swallows any exception this raises. Like
    send_support_email it no-ops (returns a status string) when Resend
    isn't configured. Returns a human-readable status.
    """
    if not settings.resend_api_key:
        return "Email isn't configured on the server yet (no Resend key)."
    recipients = list(settings.feedback_email_recipients or [])
    if not recipients:
        return "No feedback recipients are configured."
    rating_part = f"rating {rating}/5" if rating is not None else "no rating"
    category_part = (category or "").strip() or "uncategorized"
    full_subject = f"[Memoria feedback] {rating_part} — {category_part}"
    body_html = _feedback_html(
        caregiver_name=caregiver.display_name,
        caregiver_email=caregiver.email,
        rating=rating,
        whats_working=whats_working,
        whats_frustrating=whats_frustrating,
        category=category,
    )
    resend_attachments = _build_screenshot_attachments(attachments)
    await _send(
        to=recipients,
        subject=full_subject,
        html=body_html,
        settings=settings,
        reply_to=caregiver.email,
        from_email=settings.feedback_email_from,
        attachments=resend_attachments or None,
    )
    log.info(
        "email.feedback_sent caregiver=%s recipients=%d screenshots=%d",
        caregiver.id, len(recipients), len(resend_attachments),
    )
    return "Sent. We'll reply to your account email."


def _build_screenshot_attachments(
    attachments: Sequence[str] | None,
) -> list[dict[str, str]]:
    """Turn base64 JPEG strings into Resend attachment dicts, skipping any
    that don't decode as valid base64 (a malformed image must not drop the
    whole feedback email). A leading `data:` URI prefix is stripped if the
    client sent one."""
    if not attachments:
        return []
    out: list[dict[str, str]] = []
    for idx, raw in enumerate(attachments, start=1):
        b64 = raw.strip()
        if b64.startswith("data:"):
            # data:image/jpeg;base64,XXXX → keep only the payload.
            _, _, b64 = b64.partition(",")
        try:
            base64.b64decode(b64, validate=True)
        except (ValueError, binascii.Error):
            log.warning("email.feedback_attachment_invalid index=%d — skipped", idx)
            continue
        out.append({"filename": f"screenshot-{idx}.jpg", "content": b64})
    return out


def _feedback_html(
    *,
    caregiver_name: str,
    caregiver_email: str,
    rating: int | None,
    whats_working: str | None,
    whats_frustrating: str | None,
    category: str | None,
) -> str:
    def esc(text: str | None) -> str:
        return (
            (text or "")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )

    rating_str = f"{rating}/5" if rating is not None else "(none)"
    rows = (
        f"<p style=\"margin:0 0 8px;font-size:15px\"><b>Rating:</b> {rating_str}</p>"
        f"<p style=\"margin:0 0 8px;font-size:15px\"><b>Category:</b> "
        f"{esc(category) or '(none)'}</p>"
        f"<p style=\"margin:0 0 4px;font-size:15px\"><b>What's working:</b></p>"
        f"<div style=\"margin:0 0 12px;font-size:15px;line-height:1.5\">"
        f"{esc(whats_working) or '(none)'}</div>"
        f"<p style=\"margin:0 0 4px;font-size:15px\"><b>What's frustrating:</b></p>"
        f"<div style=\"margin:0 0 12px;font-size:15px;line-height:1.5\">"
        f"{esc(whats_frustrating) or '(none)'}</div>"
    )
    return (
        "<div style=\"font-family:-apple-system,Segoe UI,sans-serif;color:#1c1a17\">"
        f"<p style=\"margin:0 0 12px;color:#6b6357;font-size:13px\">"
        f"Feedback from {esc(caregiver_name)} &lt;{esc(caregiver_email)}&gt; "
        f"via the Memoria iOS app."
        "</p>"
        f"{rows}"
        "</div>"
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
        subject=f"Memoria test alert — {patient_name}",
        html=_event_html(
            patient_name=patient_name,
            title="Test alert",
            narrative=(
                "This is a test email from Memoria. If you received it, email "
                f"alerts for {patient_name} are set up correctly."
            ),
            location="Living room",
        ),
        settings=settings,
    )
    return f"Test email sent to {', '.join(recipients)}."
