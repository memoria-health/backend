"""Pre-dispatch operator gate for fired alerts.

NOT THE CAREGIVER REVIEW FEATURE
--------------------------------
There are two unrelated things called "review" in this codebase and
conflating them is the easiest mistake to make here, because they sit
one column apart on the same table:

  `events.reviewed` / `reviewed_at` / `review_choice` / `review_notes`
      The CAREGIVER feature. A family member looking back over alerts
      they have already received, through `PATCH /v1/events/{id}`.
      Post-dispatch, scoped to one caregiver's own patients. Shipping.
      Nothing in this module touches it.

  `events.gate_state` (this module)
      An OPERATOR — us, running the product — deciding whether an alert
      is a false positive BEFORE any caregiver is told. Pre-dispatch,
      cross-tenant, and invisible to customers. The operator is not a
      caregiver and does not authenticate as one; see
      `deps.operator_auth`.

WHY IT EXISTS
-------------
Over-firing is the product risk that loses customers. This gate lets a
human delete a false alarm in the seconds before it reaches a family.

THE SAFETY PROPERTY
-------------------
Human review may only ever REMOVE a false alarm, never delay a real one
indefinitely. Every held event carries an unconditional auto-release
after AUTO_RELEASE_S: if no verdict lands, the alert goes out anyway.

That guarantee is only as durable as the thing enforcing it. An
`asyncio.sleep` timer dies with the process, so a deploy landing inside
the window would strand an event at 'pending' forever — silently, since
a held event sends nothing. So there are two enforcers, and they
are safe to run together because both settle the row with the same
conditional UPDATE and only one can win it:

  * the in-process timer  — exact 60s, the normal path
  * `GateSweeper`         — DB-backed, catches anything the timer lost
                            to a restart, within one sweep interval

Do not add anything to this module that can hold an alert past
AUTO_RELEASE_S.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import text

from src.services import email, pager, push

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)

AUTO_RELEASE_S = 60


# ── verdict links ────────────────────────────────────────────────────
# The page PagerDuty delivers carries one link. Opening it renders the
# clip and the two buttons; there is no session and no login, because
# the whole point is to decide in seconds from a locked phone. The link
# itself is therefore the credential, so it is:
#
#   * HMAC-signed with a dedicated secret (not the Supabase JWT secret)
#   * bound to exactly one event id
#   * expiring, via a timestamp inside the signed payload
#   * single-use in effect — acting moves `gate_state` off 'pending',
#     and the page refuses to act on anything that is not 'pending', so
#     a replayed link renders "already settled" and can change nothing
#
# Single-use falls out of the state machine rather than a token table.
# That is deliberate: a stored token is another row to leak and another
# thing to sweep, and it would buy nothing the state check doesn't.


def mint_verdict_token(*, event_id: UUID, settings: "Settings") -> str | None:
    """Return a signed one-event verdict token, or None if unconfigured.

    Fails closed: with no secret we return None and the page ships
    without an action link, rather than shipping an unsigned one.
    """
    secret = settings.event_gate_link_secret
    if not secret:
        log.warning(
            "event_gate.link_unconfigured event=%s — paging without a "
            "verdict link; only the auto-release can settle this",
            event_id,
        )
        return None
    expires_at = int(time.time()) + settings.event_gate_link_ttl_seconds
    body = f"{event_id}.{expires_at}"
    return f"{body}.{_sign(body, secret)}"


def verify_verdict_token(token: str, *, settings: "Settings") -> UUID | None:
    """Return the event id a token authorises, or None if it is invalid.

    None covers every failure — malformed, expired, wrong signature,
    unconfigured secret — because the caller has nothing useful to say
    to whoever holds a bad link, and distinguishing them leaks whether
    a given event exists.
    """
    secret = settings.event_gate_link_secret
    if not secret or not token:
        return None
    parts = token.split(".")
    if len(parts) != 3:
        return None
    raw_id, raw_exp, signature = parts
    if not hmac.compare_digest(_sign(f"{raw_id}.{raw_exp}", secret), signature):
        return None
    try:
        if int(raw_exp) < int(time.time()):
            return None
        return UUID(raw_id)
    except ValueError:
        return None


def _sign(body: str, secret: str) -> str:
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()


# Record-only tick verdict tokens: rule on the COUNT, not an event.
# Same signer as event verdicts; `rt` prefix keeps the kinds separate.

TICK_TOKEN_TTL_S = 24 * 3600   # a count correction can wait overnight


def mint_tick_token(
    *, camera_id, captured_at, ident: str, settings: "Settings",
) -> str | None:
    secret = settings.event_gate_link_secret
    if not secret:
        return None
    exp = int(time.time()) + TICK_TOKEN_TTL_S
    body = f"rt.{camera_id}.{captured_at.timestamp()}.{ident}.{exp}"
    return f"{body}.{_sign(body, secret)}"


def verify_tick_token(token: str, *, settings: "Settings") -> tuple | None:
    """(camera_id, captured_at, ident) or None for any failure."""
    from datetime import datetime, timezone as _tz

    secret = settings.event_gate_link_secret
    if not secret or not token:
        return None
    body, _, signature = token.rpartition(".")
    if not body or not hmac.compare_digest(_sign(body, secret), signature):
        return None
    parts = body.split(".")
    # rt . camera_id . epoch-int . epoch-frac . ident . exp
    if len(parts) != 6 or parts[0] != "rt":
        return None
    camera_id, epoch = parts[1], f"{parts[2]}.{parts[3]}"
    ident, exp = parts[4], parts[5]
    try:
        if int(exp) < int(time.time()):
            return None
        captured_at = datetime.fromtimestamp(float(epoch), tz=_tz.utc)
    except (ValueError, OSError):
        return None
    return camera_id, captured_at, ident


def tick_verdict_url(
    *, camera_id, captured_at, ident: str, settings: "Settings",
) -> str | None:
    token = mint_tick_token(
        camera_id=camera_id, captured_at=captured_at, ident=ident,
        settings=settings,
    )
    base = (settings.event_gate_public_base_url or "").rstrip("/")
    if token is None or not base:
        return None
    return f"{base}/v1/event-gate/tick/{token}"


def _verdict_url(*, event_id: UUID, settings: "Settings") -> str | None:
    token = mint_verdict_token(event_id=event_id, settings=settings)
    if token is None:
        return None
    base = settings.event_gate_public_base_url.rstrip("/")
    if not base:
        log.warning(
            "event_gate.base_url_unconfigured event=%s — no verdict link",
            event_id,
        )
        return None
    return f"{base}/v1/event-gate/act/{token}"


# ── routing ──────────────────────────────────────────────────────────


async def hold_gated(
    *,
    sessionmaker: "async_sessionmaker",
    event_ids: list[UUID],
    settings: "Settings",  # noqa: ARG001 — kept for a uniform call shape
) -> tuple[list[UUID], list[UUID]]:
    """Stamp the hold and return `(gated, free)`. Fast: one SELECT, one UPDATE.

    Split out of `route_events` and made awaitable by the caller because
    of a race that cost us the operator's clip. `clip_archival`'s fast
    path decides whether to upload the pre-roll by asking the database
    whether the event is `gate_state='pending'`. Both that task and the
    routing task were spawned in the same tick, so the archival query
    routinely ran BEFORE the hold was committed, saw nothing pending,
    and skipped the upload. The operator then got a verdict page with no
    clip, which is the one thing the page exists to show.

    Awaiting this before spawning archival makes the ordering explicit
    rather than a coin flip. It costs the tick two round trips against
    the local database, and it is the only part of the gate that has to
    be synchronous.
    """
    if not event_ids:
        return [], []
    async with sessionmaker() as session:
        rows = (await session.execute(text(
            """select e.id, p.event_gate_enabled from events e
               join patients p on p.id = e.patient_id
               where e.id = any(:ids)"""),
            {"ids": [str(i) for i in event_ids]})).all()
        gated = [r[0] for r in rows if r[1]]
        free = [r[0] for r in rows if not r[1]]
        if gated:
            # `gate_held_at` is what the sweeper and the countdown both
            # measure from. It has to be stamped in the same statement
            # that holds the event, or a crash between the two would
            # leave a held event with no release deadline.
            # `gate_state is null` makes this idempotent. The caller awaits
            # it and `route_events` calls it again, so without the guard the
            # second call would push `gate_held_at` forward and quietly
            # extend the deadline — or worse, re-hold an event a reviewer
            # had already settled in between.
            await session.execute(text(
                """update events set gate_state = 'pending', gate_held_at = now()
                   where id = any(:ids) and gate_state is null"""),
                {"ids": [str(i) for i in gated]})
            await session.commit()
    return gated, free


async def route_events(
    *,
    sessionmaker: "async_sessionmaker",
    gated: list[UUID],
    free: list[UUID],
    settings: "Settings",
) -> None:
    """Dispatch the free events; page + time out the held ones.

    With `patients.event_gate_enabled` off (the default, and true for
    every patient today) every event lands in `free` and this is the
    old direct fan-out with an extra SELECT.

    `gated` and `free` come from the caller's `hold_gated()` — this used
    to recompute them, which meant an operator toggling the patient's gate
    off in the milliseconds between the two calls reclassified an
    already-held event as free: it dispatched immediately, stayed
    'pending' because nothing settled it, and was then dispatched a
    SECOND time by the sweeper 60s later. One classification, taken once,
    at the moment of the hold.
    """
    if not gated and not free:
        return

    if free:
        await _dispatch(sessionmaker=sessionmaker, event_ids=free, settings=settings)

    # One page and one timer PER held event. Batching them would mean a
    # tick that fires two detectors pages about one and silently drops
    # the other, and settling the paged one would hide the unpaged one.
    #
    # Safety-only urgency: only safety events auto-release at 60s;
    # everything else waits for a human verdict (visible in /queue).
    cats = {}
    if gated:
        async with sessionmaker() as session:
            cats = dict((await session.execute(text(
                "select id, category::text from events where id = any(:ids)"),
                {"ids": [str(e) for e in gated]})).all())
    for event_id in gated:
        urgent = cats.get(event_id) == "safety"
        log.info(
            "event_gate.held event=%s category=%s auto_release=%s",
            event_id, cats.get(event_id),
            f"{AUTO_RELEASE_S}s" if urgent else "none (awaits verdict)",
        )
        asyncio.create_task(
            _page_for(sessionmaker=sessionmaker, event_id=event_id, settings=settings),
            name=f"event-gate-page-{event_id}",
        )
        if urgent:
            asyncio.create_task(
                _auto_release(
                    sessionmaker=sessionmaker, event_id=event_id, settings=settings
                ),
                name=f"event-gate-auto-{event_id}",
            )


async def _page_for(
    *, sessionmaker: "async_sessionmaker", event_id: UUID, settings: "Settings"
) -> None:
    """Build and send the operator page for one held event."""
    try:
        async with sessionmaker() as session:
            row = (await session.execute(text(
                """select e.title, coalesce(e.narrative, ''),
                          coalesce(nullif(p.preferred_name, ''), p.full_name),
                          coalesce(e.location, '')
                   from events e join patients p on p.id = e.patient_id
                   where e.id = :id"""), {"id": str(event_id)})).first()
        if row is None:
            log.warning("event_gate.page_skip reason=event_vanished id=%s", event_id)
            return
        title, narrative, patient_name, room = row
        summary = f"{patient_name}: {title}" + (f" ({room})" if room else "")
        await pager.trigger(
            settings=settings,
            event_id=event_id,
            summary=summary,
            link_url=_verdict_url(event_id=event_id, settings=settings),
            details={
                "patient": patient_name,
                "room": room or "unknown",
                "narrative": narrative,
                "auto_release_seconds": AUTO_RELEASE_S,
            },
        )
    except Exception:  # noqa: BLE001 — a paging failure must not strand the event
        log.exception("event_gate.page_failed event=%s", event_id)


async def _dispatch(
    *, sessionmaker: "async_sessionmaker", event_ids: list[UUID], settings: "Settings"
) -> None:
    """The real fan-out to the caregiver: push + email, unchanged."""
    await push.dispatch_event_pushes(
        sessionmaker=sessionmaker, event_ids=event_ids, settings=settings)
    await email.dispatch_event_emails(
        sessionmaker=sessionmaker, event_ids=event_ids, settings=settings)


async def _settle(
    *,
    session: "AsyncSession",
    event_id: UUID,
    new_state: str,
) -> bool:
    """Move one event from 'pending' to a terminal state. True if we won.

    The `and gate_state = 'pending'` is the whole concurrency story: the
    in-process timer, the sweeper, and an operator tapping a button all
    race for the same row, and exactly one of them gets a row back.
    """
    row = (await session.execute(text(
        """update events set gate_state = :new
           where id = :id and gate_state = 'pending' returning id"""),
        {"new": new_state, "id": str(event_id)})).first()
    await session.commit()
    return row is not None


async def _auto_release(
    *, sessionmaker: "async_sessionmaker", event_id: UUID, settings: "Settings"
) -> None:
    """In-process release timer — the exact-60s path.

    `GateSweeper` covers the case where this task dies with the process.
    """
    await asyncio.sleep(AUTO_RELEASE_S)
    async with sessionmaker() as session:
        won = await _settle(session=session, event_id=event_id, new_state="auto")
    if won:
        log.info("event_gate.auto_released event=%s", event_id)
        await pager.resolve(settings=settings, event_id=event_id)
        await _dispatch(
            sessionmaker=sessionmaker, event_ids=[event_id], settings=settings)


async def act(
    *,
    sessionmaker: "async_sessionmaker",
    event_id: UUID,
    approve: bool,
    settings: "Settings",
) -> str:
    """Record an operator verdict. Idempotent.

    Returns 'approved', 'rejected', or 'already_settled' — the last
    meaning the auto-release (or another tap) got there first. Callers
    render that state rather than treating it as an error: arriving 200ms
    after the timer is normal, not a fault.
    """
    new_state = "approved" if approve else "rejected"
    async with sessionmaker() as session:
        won = await _settle(session=session, event_id=event_id, new_state=new_state)
    if not won:
        return "already_settled"

    log.info("event_gate.verdict event=%s verdict=%s", event_id, new_state)
    # Clear the incident either way — a rejected alert is settled just as
    # much as an approved one, and a phone that keeps paging about a
    # decision already made trains the operator to ignore pages.
    await pager.resolve(settings=settings, event_id=event_id)
    if approve:
        await _dispatch(
            sessionmaker=sessionmaker, event_ids=[event_id], settings=settings)
    return new_state


# ── durable auto-release ─────────────────────────────────────────────


class GateSweeper:
    """Backstop that releases held events the in-process timer lost.

    Exists because the timer is an `asyncio.sleep` in this process: a
    deploy, crash, or OOM inside the 60s window would otherwise leave an
    event 'pending' with nothing scheduled to release it and no consumer
    to notice — the alert would simply never reach the caregiver. That
    is the exact failure the safety property forbids, so it needs an
    enforcer that survives the process.

    Runs unconditionally. There is no disable flag on purpose: a knob
    that switches off the safety net is a knob that will be found in the
    off position after an incident.
    """

    def __init__(
        self, *, settings: "Settings", sessionmaker: "async_sessionmaker"
    ) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        # Sweep once before looping so events stranded by the restart we
        # just completed go out now, not one interval from now.
        await self._sweep()
        self._task = asyncio.create_task(self._run(), name="event-gate-sweeper")
        log.info(
            "event_gate.sweeper_started interval=%ds",
            self._settings.event_gate_sweep_seconds,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._settings.event_gate_sweep_seconds)
            try:
                await self._sweep()
            except Exception:  # noqa: BLE001 — the sweeper must never die
                log.exception("event_gate.sweep_failed")

    async def _sweep(self) -> None:
        """Release every pending event past its deadline, in one claim.

        The UPDATE ... RETURNING claims rows atomically, so two replicas
        cannot both dispatch the same event. `gate_held_at is null` is
        included as a repair case: an event that somehow reached
        'pending' without a stamp has no deadline and would otherwise be
        held forever, which is worse than releasing it immediately.
        """
        async with self._sessionmaker() as session:
            rows = (await session.execute(text(
                """update events set gate_state = 'auto'
                   where gate_state = 'pending'
                     and category = 'safety'
                     and (gate_held_at is null
                          or gate_held_at < now() - make_interval(secs => :s))
                   returning id"""), {"s": AUTO_RELEASE_S})).all()
            await session.commit()
        if not rows:
            return
        event_ids = [r[0] for r in rows]
        log.warning(
            "event_gate.swept count=%d events=%s — these outlived their "
            "in-process timer (restart during the hold window)",
            len(event_ids), [str(e) for e in event_ids],
        )
        for event_id in event_ids:
            await pager.resolve(settings=self._settings, event_id=event_id)
        await _dispatch(
            sessionmaker=self._sessionmaker,
            event_ids=event_ids,
            settings=self._settings,
        )
