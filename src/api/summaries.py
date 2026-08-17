"""Read endpoints for the nightly summary tables.

`sleep_nights` / `activity_days` are written by
tools/run_daily_summaries.py (fed by the record_only sleep/activity
detectors) and, until now, served to nothing. The Home tab's
sleep_bands and trend_band widgets are their first reader.

Rows come back ascending by date — chart order — and the window is
clamped to 90 days like the rest of /v1/insights.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import text

from src.api.deps import CaregiverAuth, SessionDep
from src.repos import patient_repo

router = APIRouter(prefix="/v1", tags=["summaries"])


async def _own_patient(session, patient_id: UUID, caregiver_id) -> None:
    patient = await patient_repo.find_by_id(
        session, patient_id, caregiver_id=caregiver_id,
    )
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )


@router.get("/patients/{patient_id}/sleep-nights")
async def sleep_nights(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    days: int = Query(14, ge=1, le=90),
) -> dict[str, Any]:
    await _own_patient(session, patient_id, auth.id)
    rows = (await session.execute(text("""
        select to_char(night_date, 'YYYY-MM-DD') as night,
               bedtime, sleep_onset, wake_time, duration_minutes,
               disturbances, out_of_bed_events, flags
        from sleep_nights
        where patient_id = :pid
          and night_date >= (now() - (:days * interval '1 day'))::date
        order by night_date
        """), {"pid": patient_id, "days": days})).mappings().all()
    return {"nights": [dict(r) for r in rows]}


@router.get("/patients/{patient_id}/activity-days")
async def activity_days(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    days: int = Query(30, ge=1, le=90),
) -> dict[str, Any]:
    await _own_patient(session, patient_id, auth.id)
    rows = (await session.execute(text("""
        select to_char(day_date, 'YYYY-MM-DD') as day,
               active_fraction, state_minutes, labeled_minutes,
               room_transitions
        from activity_days
        where patient_id = :pid
          and day_date >= (now() - (:days * interval '1 day'))::date
        order by day_date
        """), {"pid": patient_id, "days": days})).mappings().all()
    return {"days": [dict(r) for r in rows]}
