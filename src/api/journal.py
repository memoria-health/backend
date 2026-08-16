"""The Daily Journal — GET /v1/patients/{id}/journal?date=YYYY-MM-DD.

One page per local day: every active detector contributes a line built
by src/services/journal/day.py (pure), grouped into category sections.
Day boundary is the CAREGIVER's timezone. No summary sentence, no LLM.
"""

from __future__ import annotations

from datetime import date as date_t, datetime, timedelta
from typing import Any
from uuid import UUID
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select, text

from src.api.deps import CaregiverAuth, SessionDep
from src.db.event import Event
from src.repos import detector_repo, patient_repo
from src.services.journal import day as journal_day

router = APIRouter(prefix="/v1", tags=["journal"])


async def _caregiver_tz(session, caregiver_id) -> ZoneInfo:
    row = (await session.execute(
        text("select timezone from caregivers where id = :c"),
        {"c": caregiver_id})).first()
    name = (row[0] if row else None) or "UTC"
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


@router.get("/patients/{patient_id}/journal")
async def get_journal(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"),
) -> dict[str, Any]:
    patient = await patient_repo.find_by_id(
        session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found")
    try:
        day = date_t.fromisoformat(date)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bad_date")

    tz = await _caregiver_tz(session, auth.id)
    start = datetime(day.year, day.month, day.day, tzinfo=tz)
    end = start + timedelta(days=1)
    today_local = datetime.now(tz).date()

    # ALL displayed detectors, on or off — build_entries keeps history
    # for off detectors that have events, and skips their quiet lines.
    detectors = list(await detector_repo.list_for_patient(
        session, patient_id, caregiver_id=auth.id))

    # The day's events, grouped per detector.
    events = (await session.execute(
        select(Event)
        .where(Event.patient_id == patient_id,
               Event.occurred_at >= start, Event.occurred_at < end)
        .order_by(Event.occurred_at)
    )).scalars().all()
    events_by_detector: dict[str, list] = {}
    for ev in events:
        if ev.detector_id is not None:
            events_by_detector.setdefault(str(ev.detector_id), []).append(ev)

    # Analysis ticks: activity states + per-identifier detected times for
    # record streams, in one pass. Modest row counts at a day's scale.
    record_idents = {
        d.identifier for d in detectors
        if d.alert_trigger == "record_only"
        and d.identifier not in (None, "sleep", "activity")
    }
    ticks = (await session.execute(text("""
        select captured_at, detector_fields
        from analyses
        where patient_id = :p
          and captured_at >= :s and captured_at < :e
        order by captured_at
        """), {"p": patient_id, "s": start, "e": end})).mappings().all()
    activity_ticks = []
    detected_times: dict[str, list] = {i: [] for i in record_idents}
    for r in ticks:
        fields = r["detector_fields"] or {}
        state = fields.get("activity_state")
        if state:
            activity_ticks.append((r["captured_at"], state))
        for ident in record_idents:
            if fields.get(f"{ident}_detected") is True:
                detected_times[ident].append(r["captured_at"])

    # Coverage: hours of the day with any analysis at all.
    cov = (await session.execute(text("""
        select count(distinct date_trunc('hour', captured_at))::int as n
        from analyses
        where patient_id = :p and captured_at >= :s and captured_at < :e
        """), {"p": patient_id, "s": start, "e": end})).mappings().one()

    # Days-since streaks for quiet point detectors.
    last_rows = (await session.execute(text("""
        select detector_id, max(occurred_at) as last
        from events
        where patient_id = :p and detector_id is not null
          and occurred_at < :e
        group by detector_id
        """), {"p": patient_id, "e": end})).mappings().all()
    streaks: dict[str, int] = {}
    for r in last_rows:
        streaks[str(r["detector_id"])] = max(
            0, (day - r["last"].astimezone(tz).date()).days)

    # Last night's sleep = the night that ENDED on this day's morning
    # (sleep_nights.night_date is the evening's calendar date).
    sleep_row = (await session.execute(text("""
        select bedtime, wake_time, duration_minutes, disturbances,
               out_of_bed_events
        from sleep_nights
        where patient_id = :p and night_date = :n
        """), {"p": patient_id, "n": day - timedelta(days=1)})).mappings().first()

    entries = journal_day.build_entries(
        detectors=detectors, events_by_detector=events_by_detector, tz=tz,
        sleep_row=dict(sleep_row) if sleep_row else None,
        activity_ticks=activity_ticks,
        detected_times_by_identifier=detected_times,
        streak_days_by_detector=streaks,
        coverage_hours=int(cov["n"]),
    )
    return {
        "date": date,
        "partial": day == today_local,
        "timezone": str(tz),
        "coverage_hours": int(cov["n"]),
        "sections": journal_day.assemble_sections(entries),
    }
