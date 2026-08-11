"""Insights read API — the caregiver dashboard's data layer.

Ports the web app's `lib/insight/db.ts` queries behind caregiver auth
(2026-08-11). Until now the web app connected straight to prod Postgres
as `nemara_readonly` — a SELECT-only role WITH BYPASSRLS — and tenant
isolation was a hand-written `p.id = any($tenant)` clause in every
query, with the tenant array assembled client-side from /v1/patients.
One forgotten clause, or one bug in the array plumbing, and a caregiver
reads someone else's home. The role's whole reason to exist was that
the web app had no authenticated read path; this router is that path.

Scoping contract: the patient set is derived HERE, per request, from
`patient_repo.list_for_caregiver` — the same authority /v1/patients
gives the mobile app. An explicit patient_id outside that set is a 404
(not 403 — existence is not disclosed). Nothing in this module trusts
a client-supplied tenant list, because there isn't one.

The SQL bodies are ported verbatim where possible so the dashboard's
numbers cannot drift in the move; response shapes match db.ts exactly
(the UI can't tell). `days` is clamped server-side — the old layer let
the client pick any interval and the readonly role would scan it.

After the web app flips to this router, `nemara_readonly`'s BYPASSRLS
must be REVOKED in the same prod deploy batch (runbook: DB delta rides
the batch) — the role stays for nemara-cost, without the superpower.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from src.api.deps import CaregiverAuth, SessionDep
from src.repos import patient_repo

router = APIRouter(prefix="/v1/insights", tags=["insights"])

_MAX_DAYS = 90


async def _scoped_ids(
    session, auth, patient_id: str | None,
) -> list[UUID]:
    """The caregiver's patient ids — optionally narrowed to one.

    Derived per-request from the same repo call /v1/patients serves, so
    membership sharing (patient_members) is honoured automatically. A
    requested patient outside the set 404s: fail closed, disclose
    nothing.
    """
    rows = await patient_repo.list_for_caregiver(session, auth.id)
    ids = [p.id for p in rows]
    if patient_id is None:
        return ids
    try:
        pid = UUID(patient_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="patient_not_found")
    if pid not in ids:
        raise HTTPException(status_code=404, detail="patient_not_found")
    return [pid]


def _days(days: int) -> int:
    return max(1, min(int(days), _MAX_DAYS))


@router.get("/patients")
async def insight_patients(
    auth: CaregiverAuth, session: SessionDep,
) -> list[dict[str, Any]]:
    """Patient list with camera/room summary — `listPatients` shape."""
    ids = await _scoped_ids(session, auth, None)
    if not ids:
        return []
    rows = (await session.execute(text(
        """
        select p.id, p.full_name as name, p.preferred_name as preferred,
               coalesce(json_agg(json_build_object('room', c.room, 'online', c.online)
                        ) filter (where c.id is not null), '[]') as cams
        from patients p
        left join cameras c on c.patient_id = p.id and not c.disabled
        where p.id = any(:ids)
        group by p.id order by p.full_name
        """), {"ids": ids})).mappings().all()
    return [
        {
            "id": str(r["id"]),
            "name": r["name"],
            "preferred": r["preferred"],
            "cams": r["cams"],
        }
        for r in rows
    ]


@router.get("/activity-trend")
async def activity_trend(
    auth: CaregiverAuth, session: SessionDep,
    patient_id: str, days: int = Query(default=7),
) -> list[dict[str, Any]]:
    ids = await _scoped_ids(session, auth, patient_id)
    rows = (await session.execute(text(
        """
        select to_char(captured_at::date, 'YYYY-MM-DD') as date,
               round(100.0 * avg((motion_in_clip and people_present)::int)) as activity,
               count(*) as windows
        from analyses
        where patient_id = any(:ids)
          and captured_at >= now() - (:days * interval '1 day')
        group by 1 order by 1
        """), {"ids": ids, "days": _days(days)})).mappings().all()
    return [
        {"date": r["date"], "activity": float(r["activity"] or 0),
         "windows": int(r["windows"])}
        for r in rows
    ]


@router.get("/detection-counts")
async def detection_counts(
    auth: CaregiverAuth, session: SessionDep,
    patient_id: str, days: int = Query(default=7),
    type: str | None = None,
) -> dict[str, Any]:
    ids = await _scoped_ids(session, auth, patient_id)
    rows = (await session.execute(text(
        """
        select to_char(occurred_at::date, 'YYYY-MM-DD') as date,
               category, count(*) as n
        from events
        where patient_id = any(:ids)
          and occurred_at >= now() - (:days * interval '1 day')
          and (cast(:type as text) is null or category::text = :type)
        group by 1, 2 order by 1
        """), {"ids": ids, "days": _days(days), "type": type})).mappings().all()
    types = sorted({str(r["category"]) for r in rows})
    by_date: dict[str, dict[str, Any]] = {}
    for r in rows:
        row = by_date.setdefault(r["date"], {"date": r["date"]})
        row[str(r["category"])] = int(r["n"])
    return {"types": types, "rows": list(by_date.values())}


@router.get("/uptime")
async def uptime(
    auth: CaregiverAuth, session: SessionDep,
    patient_id: str, days: int = Query(default=7),
) -> list[dict[str, Any]]:
    """Per-camera coverage hours — gap math stays in the web layer,
    which already owns it; this returns the raw hour buckets."""
    ids = await _scoped_ids(session, auth, patient_id)
    rows = (await session.execute(text(
        """
        select c.id,
               coalesce(c.room, 'Camera ' || left(c.id::text, 4)) as camera,
               c.online,
               (select array_agg(distinct date_trunc('hour', a.captured_at)
                                 order by date_trunc('hour', a.captured_at))
                from analyses a
                where a.camera_id = c.id
                  and a.captured_at >= now() - (:days * interval '1 day')) as hours
        from cameras c where c.patient_id = any(:ids) and not c.disabled
        """), {"ids": ids, "days": _days(days)})).mappings().all()
    return [
        {
            "camera": r["camera"],
            "online": bool(r["online"]),
            "hours": [h.isoformat() for h in (r["hours"] or [])],
        }
        for r in rows
    ]


@router.get("/event-leaderboard")
async def event_leaderboard(
    auth: CaregiverAuth, session: SessionDep,
    days: int = Query(default=7), patient_id: str | None = None,
) -> list[dict[str, Any]]:
    ids = await _scoped_ids(session, auth, patient_id)
    if not ids:
        return []
    rows = (await session.execute(text(
        """
        select p.id, p.full_name as patient, count(e.id)::int as events,
               coalesce(json_object_agg(e.category, cnt.n)
                        filter (where e.category is not null), '{}') as by_category
        from patients p
        left join events e on e.patient_id = p.id
          and e.occurred_at >= now() - (:days * interval '1 day')
        left join lateral (
          select count(*)::int n from events e2
          where e2.patient_id = p.id and e2.category = e.category
            and e2.occurred_at >= now() - (:days * interval '1 day')
        ) cnt on true
        where p.id = any(:ids)
        group by p.id order by events desc
        limit 12
        """), {"ids": ids, "days": _days(days)})).mappings().all()
    return [
        {"id": str(r["id"]), "patient": r["patient"],
         "events": int(r["events"]), "byCategory": r["by_category"]}
        for r in rows
    ]


@router.get("/event-timeline")
async def event_timeline(
    auth: CaregiverAuth, session: SessionDep,
    patient_id: str, days: int = Query(default=7),
    filter: str | None = None,
) -> list[dict[str, Any]]:
    ids = await _scoped_ids(session, auth, patient_id)
    rows = (await session.execute(text(
        """
        select to_char(occurred_at, 'YYYY-MM-DD HH24:MI') as ts,
               to_char(occurred_at::date, 'YYYY-MM-DD') as date,
               category as type,
               title,
               nullif(narrative, '') as detail,
               case when has_clip then clip_storage_key end as clip_key
        from events
        where patient_id = any(:ids)
          and occurred_at >= now() - (:days * interval '1 day')
          and (cast(:f as text) is null or title ilike '%'||:f||'%'
               or narrative ilike '%'||:f||'%' or category::text ilike '%'||:f||'%')
        order by occurred_at desc limit 40
        """), {"ids": ids, "days": _days(days), "f": filter})).mappings().all()
    return [
        {"ts": r["ts"], "date": r["date"], "type": str(r["type"]),
         "title": r["title"], "detail": r["detail"], "clipKey": r["clip_key"]}
        for r in rows
    ]


@router.get("/trend-facts")
async def trend_facts(
    auth: CaregiverAuth, session: SessionDep,
    patient_id: str | None = None,
) -> dict[str, Any]:
    ids = await _scoped_ids(session, auth, patient_id)
    if not ids:
        return {
            "events": {"cur": 0, "prev": 0},
            "nightEvents": {"cur": 0, "prev": 0},
            "activity": {"cur": 0, "prev": 0},
            "offlineCameras": 0,
        }
    events = (await session.execute(text(
        """
        select count(*) filter (where occurred_at >= now() - interval '7 days')::int as cur,
               count(*) filter (where occurred_at >= now() - interval '14 days'
                            and occurred_at <  now() - interval '7 days')::int as prev
        from events where patient_id = any(:ids)
        """), {"ids": ids})).mappings().one()
    nights = (await session.execute(text(
        """
        select count(*) filter (where occurred_at >= now() - interval '7 days')::int as cur,
               count(*) filter (where occurred_at >= now() - interval '14 days'
                            and occurred_at <  now() - interval '7 days')::int as prev
        from events
        where patient_id = any(:ids)
          and (extract(hour from occurred_at) >= 22 or extract(hour from occurred_at) < 6)
        """), {"ids": ids})).mappings().one()
    activity = (await session.execute(text(
        """
        select round(100.0 * avg((motion_in_clip and people_present)::int)
                 filter (where captured_at >= now() - interval '7 days')) as cur,
               round(100.0 * avg((motion_in_clip and people_present)::int)
                 filter (where captured_at >= now() - interval '14 days'
                     and captured_at <  now() - interval '7 days')) as prev
        from analyses where patient_id = any(:ids)
        """), {"ids": ids})).mappings().one()
    offline = (await session.execute(text(
        """
        select count(*)::int as n from cameras
        where not disabled and not online and patient_id = any(:ids)
        """), {"ids": ids})).mappings().one()
    return {
        "events": {"cur": events["cur"], "prev": events["prev"]},
        "nightEvents": {"cur": nights["cur"], "prev": nights["prev"]},
        "activity": {"cur": float(activity["cur"] or 0),
                     "prev": float(activity["prev"] or 0)},
        "offlineCameras": offline["n"],
    }


@router.get("/busiest-patient")
async def busiest_patient(
    auth: CaregiverAuth, session: SessionDep,
) -> dict[str, Any] | None:
    ids = await _scoped_ids(session, auth, None)
    if not ids:
        return None
    row = (await session.execute(text(
        """
        select p.id, p.full_name, p.preferred_name, count(e.id) as n
        from patients p join events e on e.patient_id = p.id
        where e.occurred_at >= now() - interval '30 days'
          and p.id = any(:ids)
        group by p.id order by n desc limit 1
        """), {"ids": ids})).mappings().first()
    if row is None:
        return None
    return {"id": str(row["id"]), "full_name": row["full_name"],
            "preferred_name": row["preferred_name"], "n": int(row["n"])}


@router.get("/offline-cameras")
async def offline_cameras(
    auth: CaregiverAuth, session: SessionDep,
    patient_id: str | None = None,
) -> list[dict[str, Any]]:
    ids = await _scoped_ids(session, auth, patient_id)
    if not ids:
        return []
    rows = (await session.execute(text(
        """
        select coalesce(c.room, 'Camera') as room, p.full_name as patient,
               to_char(coalesce(c.last_seen_at, c.created_at), 'Mon DD HH24:MI') as last_seen
        from cameras c join patients p on p.id = c.patient_id
        where not c.disabled and not c.online and c.patient_id = any(:ids)
        order by c.last_seen_at desc nulls last limit 40
        """), {"ids": ids})).mappings().all()
    return [dict(r) for r in rows]
