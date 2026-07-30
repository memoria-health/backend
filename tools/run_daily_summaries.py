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
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from src.db.session import get_sessionmaker

# Night window: 18:00 local on night_date -> 12:00 local next day.
_NIGHT_START_H, _NIGHT_END_H = 18, 12
# Fallback only. The real window is computed per patient from the
# CAREGIVER's IANA timezone (see _tz_for): this tool loops over every
# patient, so using the runner host's zone put an India pilot's
# "night" at 23:30-17:30 local, and a fixed offset captured at import
# is also wrong across DST.
_HOST_TZ = datetime.now().astimezone().tzinfo

_TZ_SQL = text("""
select p.id::text as patient_id, coalesce(g.timezone, '') as tz
from patients p join caregivers g on g.id = p.caregiver_id
""")

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
  and c.patient_id = cast(:pid as uuid)
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


def _tz_for(name: str):
    """Caregiver's IANA zone, host zone as a last resort. A real zone
    (not a fixed offset) so DST is handled."""
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001 — bad zone string in the DB
            print(f"  ! unknown timezone {name!r}; using host zone")
    return _HOST_TZ


def _utc_naive(d: date, hour: int, tz) -> datetime:
    """Local wall-clock hour on date d -> naive-UTC (analyses convention)."""
    local = datetime.combine(d, time(hour), tzinfo=tz)
    return local.astimezone(timezone.utc).replace(tzinfo=None)


def _naive(ts: datetime) -> datetime:
    """DB timestamptz (offset-aware) -> naive-UTC, the summarizers'
    working convention. analyses.captured_at comes back aware; mixing
    it with the naive window bounds raised TypeError (field 07-29)."""
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts


async def _run(night_date: date) -> None:
    from src.services.detection.activity_summary import (
        StateTick, summarize_day,
    )
    from src.services.detection.sleep_summary import SleepTick, summarize_night

    sm = get_sessionmaker()
    async with sm() as s:
        now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        patients = (await s.execute(_TZ_SQL)).all()
        for prow in patients:
            pid, tz = prow.patient_id, _tz_for(prow.tz)
            # ── sleep: night window, in THIS patient's local time ─────
            t0 = _utc_naive(night_date, _NIGHT_START_H, tz)
            t1 = _utc_naive(night_date + timedelta(days=1), _NIGHT_END_H, tz)
            window_end = min(t1, now_naive)
            rows = (await s.execute(
                _TICKS_SQL, {"t0": t0, "t1": t1, "pid": pid},
            )).all()
            by_cam: dict[str, list] = {}
            for r in rows:
                if r.sd is None:
                    continue                 # sleep detector off that tick
                if not r.people_present:
                    # No person seen → no sleep evidence either way. A
                    # sleep-watch look at an under-blanket sleeper
                    # returns sleep_detected=false with an EMPTY-looking
                    # room — reading that as "awake" would end real
                    # nights.
                    continue
                by_cam.setdefault(r.cam, []).append(SleepTick(
                    at=_naive(r.captured_at),
                    in_bed=(str(r.sd).lower() == "true"),
                    activity=(r.sa or "none"), confidence=float(r.sc or 0.0),
                    patient_confirmed=(r.pic == "true"),
                ))
            # ONE camera's ticks, not the patient's whole home. The
            # sleep detector can be assigned to any room; a living-room
            # camera contributes "person visible, not in bed" ticks that
            # read as awake evidence and manufacture out-of-bed events
            # against the bedroom's night. Pick the camera that actually
            # saw a bed.
            ticks = max(
                by_cam.values(),
                key=lambda tl: sum(1 for t in tl if t.in_bed),
                default=[],
            )
            if not ticks:
                continue
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

            # ── activity: this patient's local calendar day ──────────
            d0 = _utc_naive(night_date, 0, tz)
            d1 = _utc_naive(night_date + timedelta(days=1), 0, tz)
            day_end = min(d1, now_naive)
            rows = (await s.execute(
                _TICKS_SQL, {"t0": d0, "t1": d1, "pid": pid},
            )).all()
            act: list = []
            for r in rows:
                if r.ast is None:
                    continue             # activity detector off that tick
                if r.pic != "true":
                    # 100%% patient-gated (Aryaman 2026-07-28): a
                    # visitor's movements must NEVER count as the
                    # patient's activity. Trade-off accepted:
                    # unconfirmed-but-patient ticks drop too —
                    # labeled_minutes shows the reduced coverage.
                    continue
                act.append(StateTick(
                    at=_naive(r.captured_at), camera=r.cam,
                    state=r.ast, confidence=float(r.ac or 0.0),
                ))
            if not act:
                continue
            day = summarize_day(act, window_end=day_end)
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
