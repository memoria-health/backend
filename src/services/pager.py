"""Operator paging via the PagerDuty Events API v2.

WHY NOT OUR OWN PUSH
--------------------
The event gate holds an alert and pages a human. That page has to land
on a phone that may be face-down, silenced, and inside a Focus mode,
and it has to keep trying if the first one is slept through. Apple only
lets a notification bypass the mute switch and Do Not Disturb if the
app holds the `com.apple.developer.usernotifications.critical-alerts`
entitlement, which is granted by application and tied to one bundle id
— and no notification of any kind repeats itself.

PagerDuty's mobile app already holds that entitlement and already does
repeat-until-acknowledged and escalation. Routing through it gets both
properties today, with no Apple review and no iOS work, and it is the
same integration the ops dashboard will want later. The trade is a
third party in the alerting path, which is acceptable because the
auto-release in `event_gate` means a page that never arrives degrades
to "the caregiver is told anyway", not "nobody is told".

SHAPE
-----
One PagerDuty incident per held event, keyed `dedup_key=<event_id>`.
Two households firing in the same second produce two independent
incidents rather than one collapsed alert — a collapse would be a
safety bug, because settling one would hide the other.

  `trigger()`  — the event has been held; page the operator.
  `resolve()`  — the event reached a terminal state (approved,
                 rejected, or auto-released). Clears the incident so
                 the phone stops paging about something already
                 settled, including when the verdict came from the
                 60s timer rather than from a human.

EVERYTHING HERE IS BEST-EFFORT
------------------------------
No function raises. A paging failure must never propagate into the
detector tick and must never hold an alert — the auto-release is the
backstop. Failures are logged loudly because a silent paging outage
looks identical to a quiet night.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx

if TYPE_CHECKING:
    from src.settings import Settings

log = logging.getLogger(__name__)

# PagerDuty rejects anything outside this set with a 400.
_VALID_SEVERITIES = {"critical", "error", "warning", "info"}

# Shared client, lazy so import doesn't open sockets. Mirrors push.py.
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


def is_configured(settings: "Settings") -> bool:
    """True when paging can actually leave the box.

    Callers use this to log the difference between "paged" and "held
    with nobody told", which are very different operational states that
    otherwise produce identical logs.
    """
    return bool(settings.pagerduty_routing_key)


async def trigger_keyed(
    *,
    settings: "Settings",
    dedup_key: str,
    source: str,
    summary: str,
    link_url: str | None = None,
    details: dict[str, Any] | None = None,
    severity: str = "critical",
    link_text: str = "Review and decide",
) -> bool:
    """Open (or update) one PagerDuty incident, keyed by an arbitrary string.

    The event gate keys on an event id, but it is not the only thing worth
    paging about: a GPU node that stops heartbeating leaves homes with no
    identity, and until this existed nothing told a human. Rather than a
    second copy of the payload and retry handling, both go through here.

    The dedup_key namespace is shared, so callers prefix it (`node:<id>`,
    or a bare event uuid for the gate). Two different subsystems colliding
    on one key would silently collapse two incidents into one, and settling
    either would hide the other.
    """
    if not is_configured(settings):
        log.warning(
            "pager.skip reason=not_configured key=%s — NOBODY is being told",
            dedup_key,
        )
        return False
    if severity not in _VALID_SEVERITIES:
        severity = "critical"

    payload: dict[str, Any] = {
        "routing_key": settings.pagerduty_routing_key,
        "event_action": "trigger",
        "dedup_key": dedup_key,
        "payload": {
            "summary": summary[:1024],
            "source": source,
            "severity": severity,
            # Distinct class per alert kind: PD's alert grouping keys on
            # these; without them it merged fall alerts into an open
            # record-only incident (live 2026-08-18, three silent falls).
            "class": summary.split(":")[-1].strip()[:64],
            "component": dedup_key.split(":")[0][:64],
            "custom_details": details or {},
        },
    }
    if link_url:
        payload["links"] = [{"href": link_url, "text": link_text}]
    return await _post(payload, settings=settings, action="trigger", key=dedup_key)


async def resolve_keyed(*, settings: "Settings", dedup_key: str) -> bool:
    """Close an incident by key. A resolve for an unknown key is a no-op at
    PagerDuty, so callers need not track whether the trigger ever landed."""
    if not is_configured(settings):
        return False
    return await _post(
        {
            "routing_key": settings.pagerduty_routing_key,
            "event_action": "resolve",
            "dedup_key": dedup_key,
        },
        settings=settings, action="resolve", key=dedup_key,
    )


async def trigger(
    *,
    settings: "Settings",
    event_id: UUID,
    summary: str,
    link_url: str | None = None,
    details: dict[str, Any] | None = None,
    severity: str = "critical",
) -> bool:
    """Open a PagerDuty incident for one held event. Returns success.

    `summary` is what the operator reads on a locked screen, so it
    carries the patient's name — unlike the caregiver push, which
    deliberately omits the narrative (see push.build_event_payload).
    That asymmetry is intentional: the operator's whole job is to judge
    whether the alert is real, which is impossible without the detail.
    The page reaches one operator device over TLS, not a household.
    """
    if not is_configured(settings):
        log.warning(
            "pager.skip reason=not_configured event=%s — event is held "
            "with NOBODY notified; only the auto-release will fire",
            event_id,
        )
        return False
    return await trigger_keyed(
        settings=settings,
        dedup_key=str(event_id),
        source="memoria-event-gate",
        summary=summary,
        link_url=link_url,
        details=details,
        severity=severity,
    )


async def resolve(*, settings: "Settings", event_id: UUID) -> bool:
    """Close the incident for one event. Safe to call for an event that
    was never paged — PagerDuty accepts a resolve for an unknown
    dedup_key as a no-op, so callers don't need to track whether the
    trigger landed.
    """
    return await resolve_keyed(settings=settings, dedup_key=str(event_id))


async def _post(
    payload: dict[str, Any],
    *,
    settings: "Settings",
    action: str,
    key: str,
) -> bool:
    try:
        client = await _get_client()
        response = await client.post(settings.pagerduty_events_url, json=payload)
    except Exception:  # noqa: BLE001 — paging must never break the caller
        log.exception("pager.transport_error action=%s key=%s", action, key)
        return False

    # 202 is the documented success. Anything else means the operator
    # is not being told, which matters enough to log at warning.
    if response.status_code != 202:
        log.warning(
            "pager.rejected action=%s key=%s status=%d body=%s",
            action, key, response.status_code, response.text[:300],
        )
        return False
    log.info("pager.%s key=%s", action, key)
    return True
