"""Fold a window of ticks into one night / one day, on a schedule.

Two pieces:

  `summarize_date(session, night_date)` — the fold. Reads `analyses`,
  hands the ticks to the pure summarizers, upserts `sleep_nights` and
  `activity_days`. Idempotent: safe to run every half hour all night,
  and the summarizer's own `incomplete_night` flag means an early run
  self-corrects on the next one.

  `NightlySummaryRunner` — runs that fold repeatedly so a caregiver
  wakes up to a finished night instead of an empty widget.

This logic used to live in `tools/run_daily_summaries.py` and ran only
when somebody typed the command. Between 2026-08-18 and 2026-08-25
nobody did: seven nights collected, none summarised, and the table's
only rows were seed data from a demo patient. Nothing failed. That is
what made it worth moving.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)

# Night window: 18:00 local on night_date -> 12:00 local the next day.
_NIGHT_START_H, _NIGHT_END_H = 18, 12

# Fallback only. The real window is per patient, from the CAREGIVER's
# IANA zone (see `_tz_for`) — this loops over every patient, so using
# the host's zone put an India pilot's "night" at 23:30-17:30 local, and
# a fixed offset captured at import is also wrong across DST.
_HOST_TZ = datetime.now().astimezone().tzinfo

_TZ_SQL = text("""
select p.id::text as patient_id, coalesce(g.timezone, '') as tz
from patients p join caregivers g on g.id = p.caregiver_id
""")

# Which sensors watched this person, NOT which person a sensor's shim
# column happens to name. This read `c.patient_id = :pid`, the
# compatibility shim, which resolves to the earliest-added covered
# patient and nobody else — so on a shared sensor the second resident's
# night was folded from zero ticks and they got no record at all, for
# the same reason their observations were invisible before
# `camera_coverage` landed (2026-08-25).
_TICKS_SQL = text("""
select a.captured_at, substr(a.camera_id::text, 1, 8) as cam,
       a.people_present, a.motion_in_clip,
       a.detector_fields->>'sleep_detected'      as sd,
       a.detector_fields->>'sleep_activity'      as sa,
       a.detector_fields->>'sleep_confidence'    as sc,
       a.detector_fields->>'activity_state'      as ast,
       a.detector_fields->>'activity_confidence' as ac,
       a.detector_fields->>'identity_patient_in_clip' as pic
from analyses a
where a.captured_at >= :t0 and a.captured_at < :t1
  and a.provider != 'skipped'
  and a.camera_id in (
      select camera_id from public.camera_patients
      where patient_id = cast(:pid as uuid))
order by a.captured_at
""")

_SLEEP_UPSERT = text("""
insert into sleep_nights (patient_id, night_date, bedtime, sleep_onset,
    wake_time, duration_minutes, disturbances, out_of_bed_events, flags)
values (cast(:pid as uuid), :nd, :bt, :so, :wk, :dm, :db,
        cast(:ob as jsonb), cast(:fl as jsonb))
on conflict (patient_id, night_date) do update set
    bedtime = excluded.bedtime, sleep_onset = excluded.sleep_onset,
    wake_time = excluded.wake_time,
    duration_minutes = excluded.duration_minutes,
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


# ── time conversions ─────────────────────────────────────────────────
# Three of these, because `analyses.captured_at` and the `sleep_nights`
# columns are both `timestamptz` while the summarizers work in naive
# UTC. Every one of them is here because getting it wrong shipped.

def _tz_for(name: str):
    """Caregiver's IANA zone, host zone as a last resort. A real zone,
    not a fixed offset, so DST is handled."""
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001 — bad zone string in the DB
            log.warning("[summaries] unknown timezone %r; using host zone", name)
    return _HOST_TZ


def _utc_naive(d: date, hour: int, tz) -> datetime:
    """Local wall-clock hour on date `d` -> naive UTC."""
    return (datetime.combine(d, time(hour), tzinfo=tz)
            .astimezone(timezone.utc).replace(tzinfo=None))


def _as_utc(ts: datetime | None) -> datetime | None:
    """Naive UTC -> offset-aware, for WRITING to `sleep_nights`.

    Those columns are `timestamptz`. A naive value makes Postgres stamp
    the SESSION timezone, so a night computed as 12:21 UTC was stored as
    19:21+00 and every reader saw times seven hours out. Only the stored
    row was wrong, which is why it hid (found 2026-08-03).
    """
    if ts is None or ts.tzinfo is not None:
        return ts
    return ts.replace(tzinfo=timezone.utc)


def _bind(ts: datetime) -> datetime:
    """Naive UTC -> offset-aware, for BINDING into SQL.

    A naive bound against a `timestamptz` is read in the client's zone,
    not UTC, so on an IST machine the night window slid 5h30m earlier
    and every night was truncated at 06:30 local. Same night, one
    connection: `< naive 06:30` gave 6659 rows, `< '06:30+00'` gave
    7981. Nothing flagged it, because the truncated last tick sits well
    before `window_end` and reads as an observed wake.
    """
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _naive(ts: datetime) -> datetime:
    """DB timestamptz -> naive UTC, the summarizers' convention. Mixing
    an aware `captured_at` with naive window bounds raised TypeError."""
    return ts.astimezone(timezone.utc).replace(tzinfo=None) if ts.tzinfo else ts


# ── the fold ─────────────────────────────────────────────────────────

async def summarize_date(session: AsyncSession, night_date: date) -> int:
    """Fold every patient's `night_date` into their sleep + activity
    rows. Returns how many patients produced a sleep row.

    Does not commit — the caller owns the transaction.
    """
    from src.services.detection.activity_summary import StateTick, summarize_day
    from src.services.detection.sleep_summary import SleepTick, summarize_night

    now_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    wrote = 0

    for prow in (await session.execute(_TZ_SQL)).all():
        pid, tz = prow.patient_id, _tz_for(prow.tz)

        # ── sleep: the night window, in THIS patient's local time ────
        t0 = _utc_naive(night_date, _NIGHT_START_H, tz)
        t1 = _utc_naive(night_date + timedelta(days=1), _NIGHT_END_H, tz)
        rows = (await session.execute(
            _TICKS_SQL, {"t0": _bind(t0), "t1": _bind(t1), "pid": pid},
        )).all()

        by_cam: dict[str, list] = {}
        for r in rows:
            if r.sd is None:
                continue                     # sleep detector off that tick
            if not r.people_present:
                # No person seen means no sleep evidence either way. A
                # forced look at someone under a blanket returns
                # sleep_detected=false with an empty-looking room, and
                # reading that as "awake" would end real nights.
                continue
            activity = r.sa or "none"
            # Ticks before 2026-08-03 used `settling` for bed entry.
            # Same crossing, renamed so both directions read as events.
            if activity == "settling":
                activity = "entering_bed"
            by_cam.setdefault(r.cam, []).append(SleepTick(
                at=_naive(r.captured_at),
                asleep=(str(r.sd).lower() == "true"),
                activity=activity,
                confidence=float(r.sc or 0.0),
                # NULL (identity never wrote the field) is NOT an
                # explicit "false" (identity ran and said not-patient).
                # Collapsing both to False meant that with the identity
                # service stopped, EVERY night came back blank with
                # `no_confirmed_bed_entry`. An absent signal must not
                # veto; only a positive non-patient verdict may.
                patient_confirmed=(r.pic != "false"),
            ))

        # ONE camera's ticks, not the whole home. The sleep detector can
        # be assigned to any room, and a living-room camera contributes
        # "person visible, not in bed" ticks that read as awake evidence
        # and manufacture out-of-bed events against the bedroom's night.
        # Pick the camera that actually saw a bed.
        ticks = max(
            by_cam.values(),
            key=lambda tl: sum(1 for t in tl if t.asleep),
            default=[],
        )
        if ticks:
            night = summarize_night(ticks, window_end=min(t1, now_naive))
            await session.execute(_SLEEP_UPSERT, {
                "pid": pid, "nd": night_date,
                "bt": _as_utc(night["bedtime"]),
                "so": _as_utc(night["sleep_onset"]),
                "wk": _as_utc(night["wake_time"]),
                "dm": night["duration_minutes"],
                "db": night["disturbances"],
                "ob": json.dumps(night["out_of_bed_events"]),
                "fl": json.dumps(night["flags"]),
            })
            wrote += 1
            log.info(
                "[summaries] sleep %s %s onset=%s wake=%s dur=%sm "
                "disturb=%s flags=%s",
                pid[:8], night_date, night["sleep_onset"], night["wake_time"],
                night["duration_minutes"], night["disturbances"],
                night["flags"],
            )

        # ── activity: this patient's local calendar day ──────────────
        d0 = _utc_naive(night_date, 0, tz)
        d1 = _utc_naive(night_date + timedelta(days=1), 0, tz)
        rows = (await session.execute(
            _TICKS_SQL, {"t0": _bind(d0), "t1": _bind(d1), "pid": pid},
        )).all()
        act: list = []
        for r in rows:
            if r.ast is None:
                continue                  # activity detector off that tick
            if r.pic != "true":
                # Fully patient-gated (Aryaman 2026-07-28): a visitor's
                # movements must never count as the patient's activity.
                # Accepted trade: unconfirmed-but-patient ticks drop too,
                # and `labeled_minutes` shows the reduced coverage.
                continue
            act.append(StateTick(
                at=_naive(r.captured_at), camera=r.cam,
                state=r.ast, confidence=float(r.ac or 0.0),
            ))
        if act:
            day = summarize_day(act, window_end=min(d1, now_naive))
            await session.execute(_ACTIVITY_UPSERT, {
                "pid": pid, "dd": night_date,
                "sm": json.dumps(day["state_minutes"]),
                "af": day["active_fraction"],
                "rt": day["room_transitions"],
                "lm": day["labeled_minutes"], "lt": day["label_ticks"],
            })

    return wrote


async def summarize_recent(session: AsyncSession, today: date) -> int:
    """Re-fold the night in progress AND the one before it.

    Both, every pass, rather than reasoning about which side of 18:00
    local each patient is on. A night_date is the EVENING's date, so
    at 03:00 the night under way belongs to yesterday — and patients in
    different zones cross that boundary at different moments. Folding
    two dates costs one extra query against an index and removes the
    class of bug where a record appears everywhere except the hours it
    is actually being watched.
    """
    return (await summarize_date(session, today - timedelta(days=1))
            + await summarize_date(session, today))


# ── the schedule ─────────────────────────────────────────────────────

class NightlySummaryRunner:
    """Re-folds the current and previous night every `interval_s`.

    Deliberately not a once-a-night job at some fixed hour. A caregiver
    who opens the app at 03:00 should see the night so far, and one who
    opens it at 07:00 should see it finished, without either of them
    landing in a gap. The fold is a couple of indexed queries and pure
    Python — no model calls — so running it every half hour costs
    effectively nothing, and the upsert makes repetition free.
    """

    def __init__(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        interval_s: float = 1800.0,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._interval_s = interval_s
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _loop(self) -> None:
        while True:
            try:
                async with self._sessionmaker() as session:
                    wrote = await summarize_recent(session, date.today())
                    await session.commit()
                if wrote:
                    log.info("[summaries] wrote %d sleep row(s)", wrote)
            except asyncio.CancelledError:
                raise
            except Exception:
                # A summary is a record, never an alert. It must not be
                # able to take the app's startup path down with it, and
                # the next pass is half an hour away regardless.
                log.exception("[summaries] pass failed; retrying next interval")
            await asyncio.sleep(self._interval_s)
