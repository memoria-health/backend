"""Nightly/daily summary runner — sleep + activity (record-only).

Thin I/O shell around the pure summarizers: queries `analyses`, folds via
sleep_summary/activity_summary, upserts into sleep_nights/activity_days.
Run manually or from cron/EOD; safe to re-run (upsert + the summarizer's
incomplete_night flag means an early run self-corrects on the next one).

    uv run python tools/run_daily_summaries.py             # last night + yesterday
    uv run python tools/run_daily_summaries.py --date 2026-07-28
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from src.db.session import get_sessionmaker

# Night window: 18:00 local on night_date -> 12:00 local next day.
_NIGHT_START_H, _NIGHT_END_H = 18, 12
_LOCAL_TZ = datetime.now().astimezone().tzinfo

_TICKS_SQL = text("""
select a.captured_at, substr(a.camera_id::text, 1, 8) as cam,
       c.patient_id::text as patient_id,
       a.people_present, a.motion_in_clip,
       a.detector_fields->>'sleep_detected'      as sd,
       a.detector_fields->>'sleep_activity'      as sa,
       a.detector_fields->>'sleep_confidence'    as sc,
       a.detector_fields->>'activity_state'      as ast,
       a.detector_fields->>'activity_confidence' as ac,
       a.detector_fields->>'identity_patient_in_clip' as pic
from analyses a join cameras c on c.id = a.camera_id
where a.captured_at >= :t0 and a.captured_at < :t1
  and a.provider != 'skipped'
order by a.captured_at
""")

_SLEEP_UPSERT = text("""
insert into sleep_nights (patient_id, night_date, bedtime, sleep_onset,
    wake_time, duration_minutes, disturbances, out_of_bed_events, flags)
values (cast(:pid as uuid), :nd, :bt, :so, :wk, :dm, :db, cast(:ob as jsonb), cast(:fl as jsonb))
on conflict (patient_id, night_date) do update set
    bedtime = excluded.bedtime, sleep_onset = excluded.sleep_onset,
    wake_time = excluded.wake_time, duration_minutes = excluded.duration_minutes,
    disturbances = excluded.disturbances,
    out_of_bed_events = excluded.out_of_bed_events,
    flags = excluded.flags, computed_at = now()
""")

_ACTIVITY_UPSERT = text("""
insert into activity_days (patient_id, day_date, state_minutes,
    active_fraction, room_transitions, labeled_minutes, label_ticks)
values (cast(:pid as uuid), :dd, cast(:sm as jsonb), :af, :rt, :lm, :lt)
on conflict (patient_id, day_date) do update set
    state_minutes = excluded.state_minutes,
    active_fraction = excluded.active_fraction,
    room_transitions = excluded.room_transitions,
    labeled_minutes = excluded.labeled_minutes,
    label_ticks = excluded.label_ticks, computed_at = now()
""")


def _utc_naive(d: date, hour: int) -> datetime:
    """Local wall-clock hour on date d -> naive-UTC (analyses convention)."""
    local = datetime.combine(d, time(hour), tzinfo=_LOCAL_TZ)
    return local.astimezone(timezone.utc).replace(tzinfo=None)


async def _run(night_date: date) -> None:
    from src.services.detection.activity_summary import (
        StateTick, summarize_day,
    )
    from src.services.detection.sleep_summary import SleepTick, summarize_night

    sm = get_sessionmaker()
    async with sm() as s:
        # ── sleep: night window ───────────────────────────────────────
        t0 = _utc_naive(night_date, _NIGHT_START_H)
        t1 = _utc_naive(night_date + timedelta(days=1), _NIGHT_END_H)
        window_end = min(t1, datetime.now(timezone.utc).replace(tzinfo=None))
        rows = (await s.execute(_TICKS_SQL, {"t0": t0, "t1": t1})).all()
        by_patient: dict[str, list] = {}
        for r in rows:
            if r.sd is None:
                continue                     # sleep detector off that tick
            by_patient.setdefault(r.patient_id, []).append(SleepTick(
                at=r.captured_at, in_bed=(r.sd == "true" or r.sd == "True"),
                activity=(r.sa or "none"), confidence=float(r.sc or 0.0),
                patient_confirmed=(r.pic == "true"),
            ))
        for pid, ticks in by_patient.items():
            night = summarize_night(ticks, window_end=window_end)
            await s.execute(_SLEEP_UPSERT, {
                "pid": pid, "nd": night_date,
                "bt": night["bedtime"], "so": night["sleep_onset"],
                "wk": night["wake_time"], "dm": night["duration_minutes"],
                "db": night["disturbances"],
                "ob": json.dumps(night["out_of_bed_events"]),
                "fl": json.dumps(night["flags"]),
            })
            print(f"sleep {pid[:8]} {night_date}: "
                  f"onset={night['sleep_onset']} wake={night['wake_time']} "
                  f"dur={night['duration_minutes']}m "
                  f"disturb={night['disturbances']} flags={night['flags']}")

        # ── activity: the night_date's daytime (00:00 -> 24:00 local) ─
        d0 = _utc_naive(night_date, 0)
        d1 = _utc_naive(night_date + timedelta(days=1), 0)
        day_end = min(d1, datetime.now(timezone.utc).replace(tzinfo=None))
        rows = (await s.execute(_TICKS_SQL, {"t0": d0, "t1": d1})).all()
        act: dict[str, list] = {}
        for r in rows:
            if r.ast is None:
                continue                 # activity detector off that tick
            if r.pic != "true":
                # 100%% patient-gated (Aryaman 2026-07-28): a visitor's
                # movements must NEVER count as the patient's activity.
                # Trade-off accepted: unconfirmed-but-patient ticks drop
                # too — labeled_minutes shows the reduced coverage.
                continue
            act.setdefault(r.patient_id, []).append(StateTick(
                at=r.captured_at, camera=r.cam,
                state=r.ast, confidence=float(r.ac or 0.0),
            ))
        for pid, ticks in act.items():
            day = summarize_day(ticks, window_end=day_end)
            await s.execute(_ACTIVITY_UPSERT, {
                "pid": pid, "dd": night_date,
                "sm": json.dumps(day["state_minutes"]),
                "af": day["active_fraction"],
                "rt": day["room_transitions"],
                "lm": day["labeled_minutes"], "lt": day["label_ticks"],
            })
            print(f"activity {pid[:8]} {night_date}: "
                  f"states={day['state_minutes']} "
                  f"active={day['active_fraction']} "
                  f"labeled={day['labeled_minutes']}m")
        await s.commit()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="night/day date YYYY-MM-DD (default: yesterday)")
    a = ap.parse_args()
    nd = (date.fromisoformat(a.date) if a.date
          else date.today() - timedelta(days=1))
    asyncio.run(_run(nd))


if __name__ == "__main__":
    main()
