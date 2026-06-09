"""Outbound SMS via Twilio for caregiver Contact Support messages.

The iOS app's floating "Text us" button posts to /v1/support; alongside the
durable Contact Support email, we text the founder cell(s) so they can reply
to the caregiver directly from their own Messages app.

No-ops cleanly when Twilio isn't configured (local dev, tests, or prod before
the integration is set up) so the support flow never fails just because SMS is
unavailable — the email is the durable path. Uses a plain httpx POST to the
Twilio REST API (HTTP Basic auth); no `twilio` SDK dependency.
"""

from __future__ import annotations

import logging

import httpx

from src.db.caregiver import Caregiver
from src.settings import Settings

log = logging.getLogger(__name__)

_TWILIO_MESSAGES_URL = "https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"


def _configured(settings: Settings) -> bool:
    """All four pieces are required to send: account, token, the Twilio-owned
    sender number, and at least one recipient cell."""
    return bool(
        settings.twilio_account_sid
        and settings.twilio_auth_token
        and settings.twilio_from_number
        and settings.support_sms_recipients
    )


async def send_support_sms(
    *,
    caregiver: Caregiver,
    message: str,
    settings: Settings,
) -> None:
    """Text the configured founder cell(s) the caregiver's help message plus
    their name and phone number (so the founder can tap it and reply directly).

    Best-effort: every failure is logged, never raised — the caller has already
    handled the durable path (email). No-ops when Twilio is unconfigured.
    """
    if not _configured(settings):
        log.info("sms.support_skipped reason=not_configured caregiver=%s", caregiver.id)
        return

    phone = (caregiver.phone or "").strip() or "no number on file"
    body = f"Memoria help — {caregiver.display_name} ({phone}):\n{message.strip()}"
    url = _TWILIO_MESSAGES_URL.format(sid=settings.twilio_account_sid)
    auth = (settings.twilio_account_sid, settings.twilio_auth_token)

    async with httpx.AsyncClient(timeout=10.0) as client:
        for to in settings.support_sms_recipients:
            try:
                resp = await client.post(
                    url,
                    auth=auth,
                    data={
                        "From": settings.twilio_from_number,
                        "To": to,
                        "Body": body,
                    },
                )
                if resp.status_code >= 400:
                    # Surface Twilio's error so misconfig (unverified trial
                    # recipient, bad number, etc.) is obvious in the logs.
                    log.warning(
                        "sms.twilio_rejected to=%s status=%d body=%s",
                        to, resp.status_code, resp.text[:500],
                    )
                else:
                    log.info("sms.support_sent caregiver=%s to=%s", caregiver.id, to)
            except httpx.HTTPError as exc:
                log.warning("sms.transport_error to=%s err=%s", to, exc)
