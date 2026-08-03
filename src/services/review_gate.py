"""Human-in-the-loop alert gate (default OFF — fully backward compatible).

When `patients.require_review` is true, fired events are HELD (review_gate=
'pending') instead of dispatching push/email. A reviewer approves (dispatch)
or rejects (silent drop). Safety valve: auto-release after AUTO_RELEASE_S —
human review may only ever REMOVE false alarms, never delay a real one
indefinitely.
"""

import asyncio
import logging
from uuid import UUID

from sqlalchemy import text

from src.services import email, push

log = logging.getLogger(__name__)

AUTO_RELEASE_S = 60


async def route_events(*, sessionmaker, event_ids: list[UUID], settings) -> None:
    """Split fired events: gated → hold + timer; rest → dispatch now."""
    async with sessionmaker() as session:
        rows = (await session.execute(text(
            """select e.id, p.require_review from events e
               join patients p on p.id = e.patient_id
               where e.id = any(:ids)"""), {"ids": [str(i) for i in event_ids]})).all()
        gated = [r[0] for r in rows if r[1]]
        free = [r[0] for r in rows if not r[1]]
        if gated:
            await session.execute(text(
                "update events set review_gate='pending' where id = any(:ids)"),
                {"ids": [str(i) for i in gated]})
            await session.commit()
    if free:
        await _dispatch(sessionmaker=sessionmaker, event_ids=free, settings=settings)
    if gated:
        async with sessionmaker() as session:
            row = (await session.execute(text(
                """select e.title, coalesce(nullif(p.preferred_name,''), p.full_name)
                   from events e join patients p on p.id = e.patient_id
                   where e.id = :id"""), {"id": str(gated[0])})).first()
        asyncio.create_task(push.dispatch_review_pushes(
            sessionmaker=sessionmaker,
            title="Review needed",
            body_text=f"{row[1]}: {row[0]}" if row else "New event awaiting review",
            event_id=str(gated[0]),
            settings=settings), name=f"review-push-{gated[0]}")
    for eid in gated:
        log.info("review_gate HELD event=%s (auto-release in %ss)", eid, AUTO_RELEASE_S)
        asyncio.create_task(_auto_release(
            sessionmaker=sessionmaker, event_id=eid, settings=settings),
            name=f"review-auto-{eid}")


async def _dispatch(*, sessionmaker, event_ids, settings) -> None:
    await push.dispatch_event_pushes(
        sessionmaker=sessionmaker, event_ids=event_ids, settings=settings)
    await email.dispatch_event_emails(
        sessionmaker=sessionmaker, event_ids=event_ids, settings=settings)


async def _auto_release(*, sessionmaker, event_id, settings) -> None:
    await asyncio.sleep(AUTO_RELEASE_S)
    async with sessionmaker() as session:
        row = (await session.execute(text(
            "update events set review_gate='auto' where id=:id and review_gate='pending' returning id"),
            {"id": str(event_id)})).first()
        await session.commit()
    if row:
        log.info("review_gate AUTO-RELEASED event=%s", event_id)
        await _dispatch(sessionmaker=sessionmaker, event_ids=[event_id], settings=settings)


async def act(*, sessionmaker, event_id, approve: bool, settings) -> str:
    """Reviewer verdict. Idempotent: only a 'pending' event changes state."""
    new = "approved" if approve else "rejected"
    async with sessionmaker() as session:
        row = (await session.execute(text(
            "update events set review_gate=:new where id=:id and review_gate='pending' returning id"),
            {"new": new, "id": str(event_id)})).first()
        await session.commit()
    if row and approve:
        await _dispatch(sessionmaker=sessionmaker, event_ids=[event_id], settings=settings)
    return new if row else "already_settled"
