"""Repository for the event aggregate.

Reads are scoped through `patients.caregiver_id`; the application
layer enforces ownership and RLS catches anything that slips past.

Cursor pagination uses a compound key (occurred_at desc, id) so a
client that's halfway through a list sees stable continuation even if
new events land while they're paging.
"""

import base64
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.event import Event
from src.db.patient import Patient
from src.repos.access import accessible_patient_ids


def encode_cursor(occurred_at: datetime, event_id: UUID) -> str:
    """Encode the (occurred_at, id) pair into an opaque base64 string."""
    payload = json.dumps({"o": occurred_at.isoformat(), "i": str(event_id)})
    return base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, UUID] | None:
    """Decode a cursor produced by `encode_cursor`. Returns None on
    any decode failure — caller treats that as "start from the top"."""
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
        data = json.loads(raw)
        return datetime.fromisoformat(data["o"]), UUID(data["i"])
    except Exception:
        return None


async def list_for_patient(
    session: AsyncSession,
    patient_id: UUID,
    *,
    caregiver_id: UUID,
    only_unreviewed: bool | None = None,
    cursor: str | None = None,
    limit: int = 30,
) -> tuple[Sequence[Event], str | None]:
    """Return the next page of events for a caregiver-owned patient,
    plus the cursor for the page after this one (or None if this is
    the last page).

    `only_unreviewed`:
        - None  → no filter (return everything)
        - True  → only reviewed = false
        - False → only reviewed = true
    """
    stmt = (
        select(Event)
        .join(Patient, Patient.id == Event.patient_id)
        .where(
            Event.patient_id == patient_id,
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
        )
        .order_by(Event.occurred_at.desc(), Event.id.desc())
        .limit(limit + 1)   # +1 sentinel: tells us whether another page exists
    )

    if only_unreviewed is True:
        stmt = stmt.where(Event.reviewed.is_(False))
    elif only_unreviewed is False:
        stmt = stmt.where(Event.reviewed.is_(True))

    # Event gate: never page a caregiver an alert that is still being
    # checked for a false positive, or one an operator suppressed. The
    # RLS policy in apply_event_gate_delta.sql enforces the same rule for
    # Supabase Realtime and for direct client reads, but the backend
    # connects as the service role, which bypasses RLS — so the filter
    # has to be repeated here or this endpoint becomes the hole.
    # `gate_state` is null for every patient with the gate off.
    stmt = stmt.where(
        or_(
            Event.gate_state.is_(None),
            Event.gate_state.notin_(("pending", "rejected")),
        )
    )

    if cursor:
        decoded = decode_cursor(cursor)
        if decoded is not None:
            cur_ts, cur_id = decoded
            # Compound (occurred_at desc, id desc) cursor: a row is
            # "after" the cursor if its occurred_at is earlier, OR
            # ties on occurred_at but id is lower.
            stmt = stmt.where(
                or_(
                    Event.occurred_at < cur_ts,
                    and_(Event.occurred_at == cur_ts, Event.id < cur_id),
                )
            )

    result = await session.execute(stmt)
    rows = list(result.scalars().all())
    if len(rows) > limit:
        last_visible = rows[limit - 1]
        return rows[:limit], encode_cursor(last_visible.occurred_at, last_visible.id)
    return rows, None


async def find_by_id(
    session: AsyncSession, event_id: UUID, *, caregiver_id: UUID
) -> Event | None:
    """Return the event if it exists AND the parent patient is owned
    by `caregiver_id`. Returns None otherwise (API maps to 404).
    """
    result = await session.execute(
        select(Event)
        .join(Patient, Patient.id == Event.patient_id)
        .where(
            Event.id == event_id,
            Patient.id.in_(accessible_patient_ids(caregiver_id)),
        )
    )
    return result.scalar_one_or_none()


async def update_partial(
    session: AsyncSession,
    event_id: UUID,
    *,
    caregiver_id: UUID,
    fields: Mapping[str, Any],
) -> Event | None:
    """Apply a partial update to a caregiver-owned event. Returns the
    updated row, or None if not found / not owned.
    """
    existing = await find_by_id(session, event_id, caregiver_id=caregiver_id)
    if existing is None:
        return None
    if fields:
        await session.execute(
            update(Event).where(Event.id == event_id).values(**fields)
        )
        await session.flush()
        return await find_by_id(session, event_id, caregiver_id=caregiver_id)
    return existing
