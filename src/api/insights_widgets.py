"""Widget query engine — port of the web app's widget-data route.

The dashboard's widgets are driven by a JSON "source" (metric + flags,
never SQL) that the web route used to satisfy by querying prod Postgres
directly as nemara_readonly/BYPASSRLS. This is that engine, verbatim
where possible, behind CaregiverAuth — the last and largest consumer of
the master-key role, whose removal is what finally lets BYPASSRLS be
revoked.

Parity rules of the port (2026-08-11):
  - SQL bodies match the TS route line-for-line except: tenant ids are
    SERVER-DERIVED (never client-supplied), intervals bind as ints, and
    the search-term filter interpolates only after stripping everything
    but word chars/space/hyphen — the exact sanitiser the TS used.
  - Post-processing (bucket fill, rolling mean, baseline band) is
    ported whole, because the response must be the FINAL widget JSON:
    splitting raw rows from shaping across two services would smear one
    engine across two repos.
  - Response keys are the TS names (statValue, rowsM, clipKey…) — the
    UI renders these directly and must not be able to tell.
  - `compareWith` gets the same scope check as the primary patient; a
    foreign id is a 404 (the TS threw 403 — changed deliberately:
    existence is not disclosed, matching the rest of /v1/insights).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from src.api.deps import CaregiverAuth, SessionDep
from src.repos import patient_repo

router = APIRouter(prefix="/v1/insights", tags=["insights"])

NIGHT = "(extract(hour from occurred_at) >= 22 or extract(hour from occurred_at) < 6)"
KIND = ("initcap(trim(replace(replace(title, 'Possible ', ''), "
        "'possible ', '')))")
DOWS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DAY_MS = 86_400_000


class WidgetRequest(BaseModel):
    source: dict[str, Any] | None = None


def _terms_filter(terms: list[str], alias: str = "",
                  detector: str | None = None) -> str:
    """Sanitised interpolation — the TS route's exact approach. Only
    [\\w\\s-] survives, so the interpolated string cannot escape its
    quotes; parity beats parameter purity here because the term list
    is variable-length inside an OR chain.

    `detector` (Home-tab widgets, 2026-08-15) narrows every events query
    to one detector's rows. It is interpolated only after widget_data
    has round-tripped it through UUID() AND scope-checked it, so the
    string is guaranteed hex-and-hyphens."""
    a = f"{alias}." if alias else ""
    base = "true"
    if terms:
        parts = []
        for t in terms:
            esc = re.sub(r"[^\w\s-]", "", str(t))
            parts.append(
                f"({a}title ilike '%{esc}%' or {a}narrative ilike '%{esc}%' "
                f"or {a}category::text ilike '%{esc}%')"
            )
        base = "(" + " or ".join(parts) + ")"
    # Gate visibility rides EVERY events query through this filter
    # (2026-08-18, soak day 1): pending/operator-rejected events are
    # hidden from caregivers everywhere else, and widget counts that
    # included them showed events whose tap-through finds nothing.
    base = (
        f"({base} and coalesce({a}gate_state,'none') "
        "not in ('pending','rejected'))"
    )
    if detector is None:
        return base
    return f"({base} and {a}detector_id = '{detector}')"


async def _scoped(session, auth) -> list[UUID]:
    rows = await patient_repo.list_for_caregiver(session, auth.id)
    return [p.id for p in rows]


def _narrow(ids: list[UUID], explicit: Any) -> list[UUID]:
    if explicit is None:
        return ids
    try:
        pid = UUID(str(explicit))
    except ValueError:
        raise HTTPException(status_code=404, detail="patient_not_found")
    if pid not in ids:
        raise HTTPException(status_code=404, detail="patient_not_found")
    return [pid]


# ── post-processing (ported whole from the TS route) ─────────────────

def _trunc(dt: datetime, gran: str) -> datetime:
    x = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc)
    if gran == "week":
        x -= timedelta(days=(x.weekday()) % 7)  # Monday-start, = PG date_trunc
    return x


def fill_buckets(
    rows: list[dict], terms: list[str], days: int, gran: str,
    mode: str, offset_days: int = 0,
) -> list[dict]:
    """Every bucket in the window exists: counts get 0 (no events IS
    zero); measurements get None (no monitoring is a gap, not a zero).
    All-time windows fill from the first data point."""
    step = timedelta(days=7 if gran == "week" else 1)
    have = {str(r["bucket"]): r for r in rows}
    now = datetime.now(timezone.utc)
    if days >= 365 and offset_days == 0:
        if not rows:
            return rows
        start = datetime.strptime(
            str(rows[0]["bucket"]), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    else:
        start = _trunc(now - timedelta(days=offset_days + days - 1), gran)
    end = _trunc(now - timedelta(days=offset_days), gran)
    out: list[dict] = []
    t = start
    while t <= end:
        k = t.strftime("%Y-%m-%d")
        out.append(have.get(k) or {
            "bucket": k,
            **{tm: (0 if mode == "zero" else None) for tm in terms},
        })
        t += step
    return out


def add_rolling(rows: list[dict], key: str, win: int) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        window = rows[max(0, i - win + 1): i + 1]
        avg = sum(float(x.get(key) or 0) for x in window) / len(window)
        out.append({**r, "trend": round(avg * 10) / 10})
    return out


def band_of(vals: list[float]) -> dict | None:
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    sd = sqrt(sum((v - avg) ** 2 for v in vals) / len(vals))
    return {
        "avg": round(avg * 10) / 10,
        "lo": max(0.0, round((avg - sd) * 10) / 10),
        "hi": round((avg + sd) * 10) / 10,
    }


def _series_rows(rows: list[dict]) -> list[dict]:
    return [{"bucket": r["t"], "events": r["n"]} for r in rows]


# ── query helpers ────────────────────────────────────────────────────

async def _term_series(session, patient, term, days, gran, night, offset=0,
                       detector=None):
    tf = _terms_filter([term] if term else [], detector=detector)
    rows = (await session.execute(text(f"""
        select to_char(date_trunc(:gran, occurred_at)::date, 'YYYY-MM-DD') as bucket,
               count(*)::int as n
        from events
        where patient_id = any(:ids)
          and occurred_at >= now() - (((:days)::int + (:off)::int) * interval '1 day')
          and occurred_at <  now() - (:off * interval '1 day')
          and {tf}
          {f'and {NIGHT}' if night else ''}
        group by 1 order by 1
        """), {"ids": patient, "days": days, "gran": gran, "off": offset},
    )).mappings().all()
    return [dict(r) for r in rows]


async def _ratio_stats(session, patient, terms, days, ratio, detector=None):
    """nightsWith/daysWith caption stats — shared by the bare ratio_tile
    response and the merged chart+ratio responses."""
    is_night = ratio == "nightsWith"
    tf = _terms_filter(terms, detector=detector)

    async def qn(offset: int) -> int:
        row = (await session.execute(text(f"""
            select count(distinct occurred_at::date)::int as n from events
            where patient_id = any(:ids)
              and occurred_at >= now() - (((:days)::int + (:off)::int) * interval '1 day')
              and occurred_at <  now() - (:off * interval '1 day')
              and {tf} {f'and {NIGHT}' if is_night else ''}
            """), {"ids": patient, "days": days, "off": offset},
        )).mappings().one()
        return int(row["n"])
    cur_n, prev_n = await qn(0), await qn(days)

    def pct(n: int) -> int:
        return round(100 * n / days)
    return {
        "statValue": f"{pct(cur_n)}%", "prevValue": pct(prev_n),
        "subtitle": (
            f"{'nights' if is_night else 'days'} with a matching event "
            f"({cur_n} of {days})"
        ),
    }


async def _event_total(session, patient, terms, days, night, detector=None,
                       offset=0):
    tf = _terms_filter(terms, detector=detector)
    row = (await session.execute(text(f"""
        select count(*)::int as n from events
        where patient_id = any(:ids)
          and occurred_at >= now() - (((:days)::int + (:off)::int) * interval '1 day')
          and occurred_at <  now() - (:off * interval '1 day')
          and {tf} {f'and {NIGHT}' if night else ''}
        """), {"ids": patient, "days": days, "off": offset},
    )).mappings().one()
    return int(row["n"])


async def _categorical(session, patient, terms, days, night, bucket_expr,
                       join_cameras=False, detector=None):
    tf = _terms_filter(terms, "e", detector=detector)
    join = "left join cameras c on c.id = e.camera_id" if join_cameras else ""
    rows = (await session.execute(text(f"""
        select {bucket_expr} as t, count(*)::int as n
        from events e {join}
        where e.patient_id = any(:ids)
          and occurred_at >= now() - (:days * interval '1 day')
          and {tf}
          {f'and {NIGHT}' if night else ''}
        group by 1 order by n desc limit 24
        """), {"ids": patient, "days": days})).mappings().all()
    return [dict(r) for r in rows]


# ── the engine ───────────────────────────────────────────────────────

@router.post("/widget-data")
async def widget_data(
    auth: CaregiverAuth, session: SessionDep, body: WidgetRequest,
) -> dict[str, Any]:
    source = body.source or {}
    allowed = await _scoped(session, auth)
    if not allowed:
        return {"kind": "series", "rows": [], "statValue": None}

    metric = source.get("metric")
    raw_terms = source.get("terms") or []
    terms: list[str] = [t for t in raw_terms if t][:8]
    gran = "week" if source.get("granularity") == "week" else "day"
    try:
        d = min(3650, max(1, int(source.get("days") or 14)))
    except (TypeError, ValueError):
        d = 14
    night = bool(source.get("night"))
    patient = _narrow(allowed, source.get("patient"))

    # Per-detector narrowing (Home-tab widgets). Same disclosure rule as
    # _narrow: unknown and foreign ids are both 404. The patient set
    # collapses to the detector's own patient so a spec can't mix one
    # detector's events with another patient's scope.
    det: str | None = None
    if source.get("detector") is not None:
        try:
            det_uuid = UUID(str(source["detector"]))
        except ValueError:
            raise HTTPException(status_code=404, detail="detector_not_found")
        drow = (await session.execute(
            text("select patient_id from detectors where id = :i"),
            {"i": det_uuid},
        )).first()
        if drow is None or drow[0] not in patient:
            raise HTTPException(status_code=404, detail="detector_not_found")
        patient = [drow[0]]
        det = str(det_uuid)

    # ── summary-table metrics are NOT served here ───────────────
    # sleep_bands / summary widgets read the dedicated endpoints in
    # src/api/summaries.py. Dead-end explicitly: falling through would
    # return an unfiltered events series silently mislabeled as sleep.
    if metric in ("sleepNights", "activityDays"):
        return {
            "kind": "series", "rows": [], "statValue": None,
            "unavailable": (
                "served by /v1/patients/{id}/sleep-nights and "
                "/activity-days, not widget-data"
            ),
        }

    # ── detectedTicks: record_only customs ──────────────────────
    # A record_only detector never promotes events (records, not
    # alarms) — its truth lives in analyses.detector_fields. This is
    # the per-day instance count for counting-style customs.
    if metric == "detectedTicks":
        ident = None
        if det:
            row = (await session.execute(text(
                "select identifier from detectors where id = :i"),
                {"i": UUID(det)})).first()
            ident = row[0] if row else None
        if not ident:
            return {"kind": "series", "rows": [], "statValue": None,
                    "unavailable": "detector is still compiling"}
        rows = (await session.execute(text("""
            select to_char(date_trunc(:gran, captured_at)::date, 'YYYY-MM-DD') as bucket,
                   count(*) filter (where (detector_fields->>:f) = 'true')::int as n
            from analyses
            where patient_id = any(:ids)
              and captured_at >= now() - (:days * interval '1 day')
            group by 1 order by 1
            """), {"ids": patient, "days": d, "gran": gran,
                   "f": f"{ident}_detected"})).mappings().all()
        filled = fill_buckets(
            [{"bucket": r["bucket"], "events": int(r["n"])} for r in rows],
            ["events"], d, gran, "zero")
        return {"kind": "series", "terms": ["events"], "rows": filled,
                "statValue": str(sum(int(r["n"]) for r in rows))}

    # ── activity ────────────────────────────────────────────────
    if metric == "activity":
        if source.get("byHour"):
            rows = (await session.execute(text("""
                select extract(hour from captured_at)::int as h,
                       round(100.0 * avg((motion_in_clip and people_present)::int))::int as n
                from analyses
                where patient_id = any(:ids)
                  and captured_at >= now() - (:days * interval '1 day')
                group by 1 order by 1
                """), {"ids": patient, "days": d})).mappings().all()
            by = {int(r["h"]): int(r["n"]) for r in rows}
            return {
                "kind": "series", "categorical": True, "unit": "%",
                "terms": ["activity"],
                "rows": [{"bucket": f"{h:02d}:00", "activity": by.get(h, 0)}
                         for h in range(24)],
            }

        async def q(offset: int):
            return [dict(r) for r in (await session.execute(text("""
                select to_char(date_trunc(:gran, captured_at)::date, 'YYYY-MM-DD') as bucket,
                       round(100.0 * avg((motion_in_clip and people_present)::int))::int as n
                from analyses
                where patient_id = any(:ids)
                  and captured_at >= now() - (((:days)::int + (:off)::int) * interval '1 day')
                  and captured_at <  now() - (:off * interval '1 day')
                group by 1 order by 1
                """), {"ids": patient, "days": d, "gran": gran, "off": offset},
            )).mappings().all()]
        cur, prev = await q(0), await q(d)

        def avg(rs):
            return round(sum(r["n"] for r in rs) / len(rs)) if rs else 0
        rows = fill_buckets(
            [{"bucket": r["bucket"], "activity": r["n"]} for r in cur],
            ["activity"], d, gran, "null")
        rolling = source.get("rolling")
        if rolling and int(rolling) > 1:
            rows = add_rolling(
                [x for x in rows if x.get("activity") is not None],
                "activity", int(rolling))
        out: dict[str, Any] = {
            "kind": "series", "unit": "%", "terms": ["activity"],
            "rows": rows,
            "statValue": f"{avg(cur)}%", "prevValue": avg(prev),
        }
        if source.get("baseline"):
            out["baseline"] = band_of([float(r["n"]) for r in prev])
        if source.get("threshold") is not None:
            out["threshold"] = float(source["threshold"])
        return out

    if metric == "coverage":
        rows = (await session.execute(text("""
            select coalesce(c.room, 'Camera') as camera, c.online,
                   (select count(distinct date_trunc('hour', a.captured_at))::int
                    from analyses a
                    where a.camera_id = c.id
                      and a.provider <> 'skipped'
                      and a.captured_at >= now() - (:days * interval '1 day')) as hours
            from cameras c where not c.disabled and c.patient_id = any(:ids)
            order by hours desc limit 12
            """), {"ids": patient, "days": d})).mappings().all()
        total = d * 24
        cams = [
            {"camera": r["camera"] + ("" if r["online"] else " (offline)"),
             "uptimePct": round(1000 * int(r["hours"] or 0) / total) / 10}
            for r in rows
        ]
        avg_pct = (round(sum(c["uptimePct"] for c in cams) / len(cams) * 10) / 10
                   if cams else None)
        return {
            "kind": "cameras", "cameras": cams,
            **({"statValue": f"{avg_pct}%"} if avg_pct is not None else {}),
            **({"subtitle": f"average coverage across {len(cams)} cameras"}
               if avg_pct is not None else {}),
        }

    if metric == "offline":
        rows = (await session.execute(text("""
            select coalesce(c.room, 'Camera') as room, p.full_name as patient,
                   to_char(coalesce(c.last_seen_at, c.created_at), 'Mon DD HH24\:MI') as last_seen
            from cameras c join patients p on p.id = c.patient_id
            where not c.disabled and not c.online and c.patient_id = any(:ids)
            order by c.last_seen_at desc nulls last limit 40
            """), {"ids": patient})).mappings().all()
        return {"kind": "offline", "statValue": str(len(rows)),
                "cameras": [dict(r) for r in rows]}

    if metric == "timeline":
        tf = _terms_filter(terms, "e", detector=det)
        rows = (await session.execute(text(f"""
            select to_char(occurred_at, 'YYYY-MM-DD HH24\:MI') as ts,
                   to_char(occurred_at::date, 'YYYY-MM-DD') as date,
                   category::text as type, title, nullif(narrative,'') as detail,
                   case when has_clip then clip_storage_key end as "clipKey",
                   p.preferred_name as who
            from events e join patients p on p.id = e.patient_id
            where e.patient_id = any(:ids)
              and occurred_at >= now() - (:days * interval '1 day')
              and {tf}
              {f'and {NIGHT}' if night else ''}
            order by occurred_at desc limit 40
            """), {"ids": patient, "days": d})).mappings().all()
        return {"kind": "events", "events": [dict(r) for r in rows]}

    if metric == "leaderboard":
        rows = (await session.execute(text(f"""
            select p.id, p.full_name as patient, count(e.id)::int as events
            from patients p left join events e on e.patient_id = p.id
              and e.occurred_at >= now() - (:days * interval '1 day')
              and coalesce(e.gate_state,'none') not in ('pending','rejected')
              {f'and {NIGHT}' if night else ''}
              {f"and e.detector_id = '{det}'" if det else ''}
            where p.id = any(:ids)
            group by p.id order by events desc limit 10
            """), {"ids": patient, "days": d})).mappings().all()
        ranking = [
            {"id": str(r["id"]), "patient": r["patient"],
             "events": int(r["events"]), "byCategory": {}}
            for r in rows if int(r["events"]) > 0
        ]
        return {"kind": "leaderboard", "ranking": ranking,
                "statValue": str(sum(r["events"] for r in ranking))}

    if metric == "daysSince":
        tf = _terms_filter(terms, detector=det)
        rows = (await session.execute(text(f"""
            select occurred_at, {KIND} as t from events
            where patient_id = any(:ids) and {tf}
              {f'and {NIGHT}' if night else ''}
            order by occurred_at desc limit 500
            """), {"ids": patient})).mappings().all()
        if not rows:
            return {"kind": "series", "rows": [], "statValue": None,
                    "unavailable": "no matching events have ever been recorded"}
        now = datetime.now(timezone.utc)
        last = rows[0]["occurred_at"]
        days_since = int((now - last).total_seconds() // 86_400)
        longest = 0.0
        for i in range(len(rows) - 1):
            gap = (rows[i]["occurred_at"] - rows[i + 1]["occurred_at"]
                   ).total_seconds() / 86_400
            longest = max(longest, gap)
        return {
            "kind": "series", "rows": [], "statValue": str(days_since),
            "subtitle": (
                f"days since the last {rows[0]['t']} "
                f"({last.strftime('%b %-d')}) · longest gap {round(longest)}d"
            ),
        }

    if metric == "presence":
        rows = (await session.execute(text("""
            select to_char(date_trunc(:gran, captured_at)::date, 'YYYY-MM-DD') as bucket,
                   round(100.0 * avg(people_present::int))::int as n
            from analyses
            where patient_id = any(:ids)
              and captured_at >= now() - (:days * interval '1 day')
            group by 1 order by 1
            """), {"ids": patient, "days": d, "gran": gran})).mappings().all()
        stat = (f"{round(sum(int(r['n']) for r in rows) / len(rows))}%"
                if rows else None)
        return {
            "kind": "series", "unit": "%", "terms": ["presence"],
            "rows": fill_buckets(
                [{"bucket": r["bucket"], "presence": int(r["n"])} for r in rows],
                ["presence"], d, gran, "null"),
            "statValue": stat,
            "subtitle": "share of monitored windows with someone visible",
        }

    if metric == "routine":
        rows = (await session.execute(text("""
            select to_char(captured_at::date, 'YYYY-MM-DD') as bucket,
                   min(extract(hour from captured_at) + extract(minute from captured_at)/60.0)
                     filter (where people_present) as wake,
                   max(extract(hour from captured_at) + extract(minute from captured_at)/60.0)
                     filter (where people_present) as bed
            from analyses
            where patient_id = any(:ids)
              and captured_at >= now() - (:days * interval '1 day')
            group by 1 order by 1
            """), {"ids": patient, "days": d})).mappings().all()
        rs = [
            {"bucket": r["bucket"],
             "wake": round(float(r["wake"]) * 10) / 10,
             "bed": round(float(r["bed"]) * 10) / 10}
            for r in rows if r["wake"] is not None
        ]
        return {"kind": "series", "terms": ["wake", "bed"], "unit": "h",
                "rows": rs}

    if metric == "reviews":
        tf = _terms_filter(terms, detector=det)
        rows = (await session.execute(text(f"""
            select case when not reviewed then 'Unreviewed'
                        else initcap(coalesce(nullif(review_choice::text,''), 'reviewed')) end as t,
                   count(*)::int as n
            from events
            where patient_id = any(:ids)
              and occurred_at >= now() - (:days * interval '1 day')
              and {tf}
            group by 1 order by n desc
            """), {"ids": patient, "days": d})).mappings().all()
        rows = [dict(r) for r in rows]
        unrev = next((r["n"] for r in rows if r["t"] == "Unreviewed"), 0)
        return {
            "kind": "series", "categorical": True, "terms": ["events"],
            "rows": _series_rows(rows),
            "statValue": str(unrev), "subtitle": "alerts not yet reviewed",
        }

    if metric == "matrix":
        rows = (await session.execute(text(f"""
            select p.preferred_name as patient, {KIND} as t, count(*)::int as n
            from events e join patients p on p.id = e.patient_id
            where occurred_at >= now() - (:days * interval '1 day')
              and p.id = any(:ids)
              {f"and e.detector_id = '{det}'" if det else ''}
            group by 1, 2
            """), {"ids": patient, "days": d})).mappings().all()
        rows = [dict(r) for r in rows]
        type_totals: dict[str, int] = {}
        pat_totals: dict[str, int] = {}
        for r in rows:
            type_totals[r["t"]] = type_totals.get(r["t"], 0) + r["n"]
            pat_totals[r["patient"]] = pat_totals.get(r["patient"], 0) + r["n"]
        cols = [t for t, _ in sorted(type_totals.items(),
                                     key=lambda kv: -kv[1])[:6]]
        pats = [p for p, _ in sorted(pat_totals.items(),
                                     key=lambda kv: -kv[1])[:8]]
        return {
            "kind": "matrix", "cols": cols,
            "rowsM": [
                {"patient": pt,
                 "cells": [next((r["n"] for r in rows
                                 if r["patient"] == pt and r["t"] == t), 0)
                           for t in cols]}
                for pt in pats
            ],
        }

    if metric == "clips":
        tf = _terms_filter(terms, "e", detector=det)
        rows = (await session.execute(text(f"""
            select title, to_char(occurred_at, 'Mon DD HH24\:MI') as ts,
                   clip_storage_key as "clipKey", p.preferred_name as who
            from events e join patients p on p.id = e.patient_id
            where has_clip and clip_storage_key is not null
              and e.patient_id = any(:ids)
              and occurred_at >= now() - (:days * interval '1 day')
              and {tf}
              {f'and {NIGHT}' if night else ''}
            order by occurred_at desc limit 8
            """), {"ids": patient, "days": d})).mappings().all()
        return {"kind": "clips", "items": [dict(r) for r in rows]}

    # ── events (workhorse) ──────────────────────────────────────
    # A request may carry BOTH a chart shape and a caption stat
    # (hour_clock: byHour+ratio; dot_calendar: daily rows+daysWith;
    # week_heatmap: heatmap+comparePrev). Chart shapes win the response
    # body; the stat merges in on top — first-match-wins here used to
    # blank the chart (Phase A review, 2026-08-15).
    ratio = source.get("ratio")
    chart_intent = any(source.get(k) for k in (
        "breakdown", "byRoom", "byHour", "byDow", "heatmap", "mode",
        "compareWith", "comparePrev",
    ))
    if ratio in ("nightsWith", "daysWith") and not chart_intent:
        stats = await _ratio_stats(session, patient, terms, d, ratio, det)
        return {"kind": "series", "rows": [], **stats}

    if source.get("breakdown"):
        rows = await _categorical(session, patient, terms, d, night, KIND,
                                  detector=det)
        return {"kind": "series", "categorical": True, "terms": ["events"],
                "rows": _series_rows(rows),
                "statValue": str(sum(r["n"] for r in rows))}
    if source.get("byRoom"):
        rows = await _categorical(
            session, patient, terms, d, night,
            "coalesce(c.room, 'Unknown room')", join_cameras=True,
            detector=det)
        return {"kind": "series", "categorical": True, "terms": ["events"],
                "rows": _series_rows(rows),
                "statValue": str(sum(r["n"] for r in rows))}
    if source.get("byHour"):
        rows = await _categorical(
            session, patient, terms, d, night,
            r"""to_char(occurred_at, 'HH24"\:00"')""", detector=det)
        by = {r["t"]: r["n"] for r in rows}
        out = {
            "kind": "series", "categorical": True, "terms": ["events"],
            "rows": [{"bucket": f"{h:02d}:00", "events": by.get(f"{h:02d}:00", 0)}
                     for h in range(24)],
            "statValue": str(sum(r["n"] for r in rows)),
        }
        if ratio in ("nightsWith", "daysWith"):
            out.update(await _ratio_stats(session, patient, terms, d, ratio, det))
        return out
    if source.get("byDow"):
        rows = await _categorical(
            session, patient, terms, d, night,
            "extract(isodow from occurred_at)::int::text", detector=det)
        by = {int(r["t"]): r["n"] for r in rows}
        return {
            "kind": "series", "categorical": True, "terms": ["events"],
            "rows": [{"bucket": lbl, "events": by.get(i + 1, 0)}
                     for i, lbl in enumerate(DOWS)],
            "statValue": str(sum(r["n"] for r in rows)),
        }
    if source.get("heatmap"):
        tf = _terms_filter(terms, detector=det)
        rows = (await session.execute(text(f"""
            select extract(isodow from occurred_at)::int as dow,
                   extract(hour from occurred_at)::int as h, count(*)::int as n
            from events
            where patient_id = any(:ids)
              and occurred_at >= now() - (:days * interval '1 day')
              and {tf}
            group by 1, 2
            """), {"ids": patient, "days": d})).mappings().all()
        cells = [dict(r) for r in rows]
        out = {"kind": "heatmap", "dows": DOWS, "cells": cells,
               "statValue": str(sum(r["n"] for r in cells))}
        if source.get("comparePrev"):
            out["prevValue"] = await _event_total(
                session, patient, terms, d, night, det, offset=d)
        return out

    if source.get("compareWith"):
        other = _narrow(allowed, source["compareWith"])

        async def combined(ids, offset=0):
            tf = _terms_filter(terms, detector=det)
            rows = (await session.execute(text(f"""
                select to_char(date_trunc(:gran, occurred_at)::date, 'YYYY-MM-DD') as bucket,
                       count(*)::int as n
                from events
                where patient_id = any(:ids)
                  and occurred_at >= now() - (((:days)::int + (:off)::int) * interval '1 day')
                  and occurred_at <  now() - (:off * interval '1 day')
                  and {tf}
                  {f'and {NIGHT}' if night else ''}
                group by 1 order by 1
                """), {"ids": ids, "days": d, "gran": gran, "off": offset},
            )).mappings().all()
            return [dict(r) for r in rows]
        a = await combined(patient)
        b2 = await combined(other)
        la = source.get("patientLabel") or "A"
        lb = source.get("compareWithLabel") or "B"
        af = fill_buckets([{"bucket": r["bucket"], la: r["n"]} for r in a],
                          [la], d, gran, "zero")
        bf = {r["bucket"]: r.get(lb, 0) for r in fill_buckets(
            [{"bucket": r["bucket"], lb: r["n"]} for r in b2],
            [lb], d, gran, "zero")}
        return {"kind": "series", "terms": [la, lb],
                "rows": [{**r, lb: bf.get(r["bucket"], 0)} for r in af]}

    if source.get("comparePrev"):
        cur = await _term_series(session, patient, None, d, gran, night, 0) \
            if not terms else None
        # combined over the term set, matching the TS combinedSeries
        tf = _terms_filter(terms, detector=det)

        async def combined(offset):
            rows = (await session.execute(text(f"""
                select to_char(date_trunc(:gran, occurred_at)::date, 'YYYY-MM-DD') as bucket,
                       count(*)::int as n
                from events
                where patient_id = any(:ids)
                  and occurred_at >= now() - (((:days)::int + (:off)::int) * interval '1 day')
                  and occurred_at <  now() - (:off * interval '1 day')
                  and {tf}
                  {f'and {NIGHT}' if night else ''}
                group by 1 order by 1
                """), {"ids": patient, "days": d, "gran": gran, "off": offset},
            )).mappings().all()
            return [dict(r) for r in rows]
        cur, prev = await combined(0), await combined(d)
        cf = fill_buckets([{"bucket": r["bucket"], "current": r["n"]}
                           for r in cur], ["current"], d, gran, "zero")
        pf = fill_buckets([{"bucket": r["bucket"], "previous": r["n"]}
                           for r in prev], ["previous"], d, gran, "zero", d)
        return {
            "kind": "series", "terms": ["current", "previous"],
            "rows": [{**r, "previous": (pf[i]["previous"] if i < len(pf) else 0)}
                     for i, r in enumerate(cf)],
        }

    active_terms: list[str | None] = terms if terms else [None]
    cur_all = [await _term_series(session, patient, t, d, gran, night, 0,
                                  detector=det)
               for t in active_terms]
    prev_all = [await _term_series(session, patient, t, d, gran, night, d,
                                   detector=det)
                for t in active_terms]
    names = terms if terms else ["events"]
    by_bucket: dict[str, dict] = {}
    for name, series in zip(names, cur_all):
        for r in series:
            row = by_bucket.setdefault(r["bucket"], {"bucket": r["bucket"]})
            row[name] = r["n"]
    rows = sorted(by_bucket.values(), key=lambda r: str(r["bucket"]))
    rows = fill_buckets(rows, names, d, gran, "zero")
    rolling = source.get("rolling")
    if rolling and int(rolling) > 1 and len(names) == 1:
        rows = add_rolling(rows, names[0], int(rolling))
    out = {
        "kind": "series", "terms": names, "rows": rows,
        "statValue": str(sum(r["n"] for s in cur_all for r in s)),
        "prevValue": sum(r["n"] for s in prev_all for r in s),
    }
    if source.get("overlay") == "activity":
        act = (await session.execute(text("""
            select to_char(date_trunc(:gran, captured_at)::date, 'YYYY-MM-DD') as bucket,
                   round(100.0 * avg((motion_in_clip and people_present)::int))::int as n
            from analyses
            where patient_id = any(:ids)
              and captured_at >= now() - (:days * interval '1 day')
            group by 1
            """), {"ids": patient, "days": d, "gran": gran})).mappings().all()
        am = {r["bucket"]: int(r["n"]) for r in act}
        out["rows"] = [{**r, "activityPct": am.get(str(r["bucket"]))}
                       for r in out["rows"]]
        out["hasOverlay"] = True
    if source.get("baseline"):
        out["baseline"] = band_of(
            [float(r["n"]) for s in prev_all for r in s])
    if source.get("threshold") is not None:
        out["threshold"] = float(source["threshold"])
    if ratio in ("nightsWith", "daysWith"):
        out.update(await _ratio_stats(session, patient, terms, d, ratio, det))
    return out
