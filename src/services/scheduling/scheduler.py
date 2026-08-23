"""Runtime for scheduled ("absence") detectors.

A scheduled detector runs the same per-tick reactive detection as any
other (its `<identifier>_detected` lands in analyses.detector_fields),
but instead of firing the moment it's seen, it fires if the signal is
NEVER seen within a recurring daily/weekly window — "medication not
taken today".

This module is a standalone background task (started in main.lifespan,
parallel to the DetectorWorker). Once per `scheduler_tick_seconds` it:

  1. loads every enabled scheduled detector,
  2. for each whose deadline has passed today and hasn't been evaluated
     yet today (single-fire guard via `last_scheduled_eval_date`),
  3. queries `analyses` for the watched signal inside the window,
  4. fires an absence Event (→ push + email, reusing the existing
     dispatchers) if the signal was absent,
  5. records `last_scheduled_eval_date` so it runs exactly once per day.

Timezone is the home's (patients.timezone, falling back to the
caregiver's) — see services/patient_tz.py.

Limitations (v1, by design):
  - windows assumed within a single local day (window_start < window_end);
  - if the process is down at the deadline, that day evaluates late on
    next startup (still that day) but past days aren't back-filled.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, time, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.db.caregiver import Caregiver
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.services import email, patient_tz, push
from src.services.detection.scheduling.decision import episode_exit_gap_for
from src.settings import Settings

log = logging.getLogger(__name__)

_FALLBACK_TZ = "America/New_York"

# How far back the falling-edge sweep looks for the episode's last
# sighting (bounded so a long-idle detector doesn't scan all history each
# tick). The gap itself now comes from `decision.episode_exit_gap_for` so
# both edges of an episode use one definition.
_TRAILING_LOOKBACK_SECONDS = 6 * 3600


def _parse_hhmm(value: object) -> time | None:
    if not isinstance(value, str):
        return None
    try:
        hh, mm = value.split(":")
        return time(hour=int(hh), minute=int(mm))
    except (ValueError, TypeError):
        return None


class ScheduledDetectorRunner:
    def __init__(self, *, settings: Settings, sessionmaker) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._loop(), name="scheduled-detector-runner")
        log.info("scheduled_runner.start tick=%ds", self._settings.scheduler_tick_seconds)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        log.info("scheduled_runner.stop done")

    async def _loop(self) -> None:
        while True:
            try:
                await self._tick()
            except Exception:  # noqa: BLE001 — one bad tick must not kill the loop
                log.exception("scheduled_runner.tick_failed")
            await asyncio.sleep(self._settings.scheduler_tick_seconds)

    async def _tick(self) -> None:
        async with self._sessionmaker() as session:
            # Both clock-driven triggers: schedule fires on ABSENCE at a
            # deadline; trailing fires on the FALLING edge (a sustained
            # activity that has now ended). Neither fires per-tick reactively.
            stmt = (
                select(Detector)
                .where(
                    Detector.alert_trigger.in_(("absence", "trailing")),
                    Detector.is_on.is_(True),
                    Detector.identifier.is_not(None),
                )
            )
            detectors = (await session.execute(stmt)).scalars().all()
            for det in detectors:
                try:
                    if det.alert_trigger == "trailing":
                        await self._evaluate_trailing(session, det)
                    else:
                        await self._evaluate_one(session, det)
                except Exception:  # noqa: BLE001
                    log.exception("scheduled_runner.evaluate_failed detector=%s", det.id)

    async def _resolve_tz(self, session: AsyncSession, patient) -> ZoneInfo:
        """Windows are the HOME's local time: "medication by 5pm" means
        5pm where the patient lives, not where the caregiver is reading
        from. Falls back to the caregiver's zone (the old behaviour)."""
        cg = await session.get(Caregiver, patient.caregiver_id)
        return patient_tz.zone_for(
            getattr(patient, "timezone", None), cg.timezone if cg else None)

    async def _evaluate_one(self, session: AsyncSession, det: Detector) -> None:
        # schedule_config dissolution (2026-07-18) + MULTI-WINDOW support
        # (Aryaman 2026-07-28): when_scope is THE truth, and every window
        # in it gets its own deadline evaluation — "8 AM, 8 PM +-1H" is
        # two daily checks. (parse_schedule's old first-window view
        # silently dropped the evening one.) Per-window single-fire is
        # stateless: an absence event's occurred_at IS its window's
        # deadline, so an existing event at that exact instant means
        # this window already fired today. last_scheduled_eval_date now
        # marks WHOLE-DAY completion (all deadlines processed) only.
        from src.services.detection.scheduling.when_scope import (
            ScheduleWindow, parse_schedules,
        )

        sws = list(parse_schedules(det.when_scope))
        if not sws:
            cfg = det.schedule_config or {}
            cadence = cfg.get("cadence")
            ws_t = _parse_hhmm(cfg.get("window_start"))
            we_t = _parse_hhmm(cfg.get("window_end"))
            dl_t = _parse_hhmm(cfg.get("deadline"))
            days_cfg = cfg.get("days_of_week") or []
            # Graceful defaults (2026-07-28, "Missed evening medication"
            # incident): deadline-only configs mean "seen at any point
            # today before the deadline" — window_end defaults to the
            # deadline, window_start to midnight. Only a truly unusable
            # config (no deadline at all) bails, and loudly.
            if dl_t is None and we_t is not None:
                dl_t = we_t
            if we_t is None and dl_t is not None:
                we_t = dl_t
            if ws_t is None and we_t is not None:
                ws_t = time(0, 0)
            if cadence not in ("daily", "weekly") or ws_t is None or we_t is None:
                log.error(
                    "scheduled_runner.unusable_config detector=%s cfg=%s — "
                    "this ABSENCE detector will NEVER evaluate; fix its "
                    "schedule_config",
                    det.id, cfg,
                )
                return
            if ws_t >= we_t:
                log.warning("scheduled_runner.inverted_window detector=%s", det.id)
                return
            sws = [ScheduleWindow(
                start_min=ws_t.hour * 60.0 + ws_t.minute,
                end_min=we_t.hour * 60.0 + we_t.minute,
                days=tuple(days_cfg) if days_cfg else None,
            )]

        patient = await session.get(Patient, det.patient_id)
        if patient is None:
            return
        tz = await self._resolve_tz(session, patient)
        now_local = datetime.now(tz)
        today = now_local.date()

        # Whole day already processed.
        if det.last_scheduled_eval_date == today:
            return

        def _t(minutes: float) -> time:
            return time(int(minutes) // 60, int(minutes) % 60)

        due_today = [
            w for w in sws
            if w.days is None or now_local.weekday() in w.days
        ]
        if not due_today:
            return

        for w in due_today:
            ws, we = _t(w.start_min), _t(w.end_min)
            deadline_local = datetime.combine(today, we, tzinfo=tz)
            if now_local < deadline_local:
                continue
            deadline_utc = deadline_local.astimezone(timezone.utc)

            # Per-window single-fire, stateless: occurred_at == deadline.
            already = (await session.execute(
                select(Event.id).where(
                    Event.detector_id == det.id,
                    Event.occurred_at == deadline_utc,
                ).limit(1)
            )).scalar_one_or_none()
            if already is not None:
                continue

            start_utc = datetime.combine(today, ws, tzinfo=tz).astimezone(timezone.utc)
            end_utc = deadline_utc
            detected_key = f"{det.identifier}_detected"

            if getattr(self._settings, "detection_v2_decision_enabled", False):
                from src.services.detection.scheduling.decision import (
                    qualifying_tick_count,
                )
                seen = (
                    await qualifying_tick_count(
                        session,
                        patient_id=patient.id,
                        detected_field=detected_key,
                        conf_field=f"{det.identifier}_confidence",
                        threshold=None,
                        since=start_utc,
                        before=end_utc,
                    )
                ) > 0
            else:
                seen_stmt = (
                    select(Analysis.id)
                    .where(
                        Analysis.patient_id == patient.id,
                        Analysis.captured_at >= start_utc,
                        Analysis.captured_at < end_utc,
                        Analysis.detector_fields[detected_key].astext == "true",
                    )
                    .limit(1)
                )
                seen = (await session.execute(seen_stmt)).scalar_one_or_none() is not None

            if seen:
                # debug, not info: seen windows re-check every tick until
                # the day's final deadline stamps completion.
                log.debug(
                    "scheduled_runner.seen detector=%s window_end=%s — no alert",
                    det.id, we,
                )
                continue

            pct = await self._uptime_pct(session, patient.id, start_utc, end_utc)
            if pct == 0:
                log.info(
                    "scheduled_runner.no_monitoring detector=%s window_end=%s — "
                    "suppressed (0%% coverage)",
                    det.id, we,
                )
                continue

            event = self._build_absence_event(
                det=det, patient=patient, ws=ws, we=we,
                deadline_utc=deadline_utc, uptime_pct=pct,
            )
            session.add(event)
            await session.commit()
            log.info(
                "scheduled_runner.fired detector=%s date=%s window_end=%s "
                "event=%s uptime=%d%%",
                det.id, today, we, event.id, pct,
            )
            asyncio.create_task(push.dispatch_event_pushes(
                sessionmaker=self._sessionmaker, event_ids=[event.id],
                settings=self._settings,
            ))
            asyncio.create_task(email.dispatch_event_emails(
                sessionmaker=self._sessionmaker, event_ids=[event.id],
                settings=self._settings,
            ))

        # Whole-day completion: stamp only after the LAST deadline passed,
        # so earlier windows can't block later ones.
        last_deadline = max(
            datetime.combine(today, _t(w.end_min), tzinfo=tz) for w in due_today
        )
        if now_local >= last_deadline:
            det.last_scheduled_eval_date = today
            await session.commit()

    async def _evaluate_trailing(self, session: AsyncSession, det: Detector) -> None:
        """Fire once a sustained activity has FINISHED — detected recently,
        then ABSENT for >= its hysteresis gap. The falling-edge counterpart of
        the rising-edge promotion. One-shot per episode: guarded by the last
        trailing event vs the episode's last detection, so a whole session
        yields exactly one "finished" alert, and a fresh session re-arms."""
        patient = await session.get(Patient, det.patient_id)
        if patient is None:
            return
        # Same exit gap the rising edge uses, so the two edges agree on
        # when an episode ended (2026-07-29). Previously read the
        # per-detector `episode_gap_seconds`, an LLM-guessed activity
        # duration — with the two edges free to disagree, a session could
        # report "finished" while the rising edge still considered it
        # ongoing. Criticality-derived, so nothing is guessed per detector.
        gap = episode_exit_gap_for(det)
        now_utc = datetime.now(timezone.utc)
        detected_key = f"{det.identifier}_detected"

        # Most recent qualifying detection (bounded lookback so a long-dead
        # detector doesn't scan all history every tick).
        last_seen = (await session.execute(
            select(func.max(Analysis.captured_at)).where(
                Analysis.patient_id == patient.id,
                Analysis.captured_at >= now_utc - timedelta(seconds=_TRAILING_LOOKBACK_SECONDS),
                Analysis.detector_fields[detected_key].astext == "true",
            )
        )).scalar_one_or_none()
        if last_seen is None:
            return  # nothing ongoing recently
        # Still within the gap since the last sighting → the episode hasn't
        # ended yet; wait.
        if (now_utc - last_seen).total_seconds() < gap:
            return
        # Episode ended. Skip if a trailing event already fired for it (any
        # event for this detector dated at/after the episode's last sighting).
        already = (await session.execute(
            select(Event.id)
            .where(Event.detector_id == det.id, Event.occurred_at >= last_seen)
            .limit(1)
        )).scalar_one_or_none()
        if already is not None:
            return

        ended_at = last_seen + timedelta(seconds=gap)
        event = self._build_trailing_event(det=det, patient=patient, ended_at=ended_at)
        session.add(event)
        await session.commit()
        log.info(
            "trailing_runner.fired detector=%s ended=%s event=%s",
            det.id, ended_at.isoformat(), event.id,
        )
        asyncio.create_task(push.dispatch_event_pushes(
            sessionmaker=self._sessionmaker, event_ids=[event.id], settings=self._settings,
        ))
        asyncio.create_task(email.dispatch_event_emails(
            sessionmaker=self._sessionmaker, event_ids=[event.id], settings=self._settings,
        ))

    def _build_trailing_event(
        self, *, det: Detector, patient: Patient, ended_at: datetime,
    ) -> Event:
        return Event(
            id=uuid4(),
            patient_id=patient.id,
            detector_id=det.id,
            camera_id=None,
            category=det.category,
            source=det.source,
            icon_name=det.icon_name,
            title=f"Finished: {det.name}",
            location="Activity ended",
            occurred_at=ended_at,
            has_clip=False,
            narrative=f"\"{det.name}\" has finished — no longer detected.",
            opened=False,
            reviewed=False,
            review_choice=None,
            review_notes=None,
            reviewed_at=None,
            created_at=datetime.now(timezone.utc),
        )

    async def _uptime_pct(self, session, patient_id, start_utc, end_utc) -> int:
        """% of the window covered by ACTIVELY-ANALYZING ticks (provider
        != 'skipped'). A rough proxy for how much of the window we could
        actually have seen the signal in — surfaced in the alert so a
        caregiver can judge how much to trust an absence."""
        count = (await session.execute(
            select(func.count(Analysis.id)).where(
                Analysis.patient_id == patient_id,
                Analysis.captured_at >= start_utc,
                Analysis.captured_at < end_utc,
                Analysis.provider != "skipped",
            )
        )).scalar_one()
        window_seconds = (end_utc - start_utc).total_seconds()
        tick = max(1, self._settings.livekit_worker_tick_seconds)
        expected = max(1.0, window_seconds / tick)
        return int(min(1.0, count / expected) * 100)

    def _build_absence_event(
        self, *, det: Detector, patient: Patient, ws: time, we: time,
        deadline_utc: datetime, uptime_pct: int,
    ) -> Event:
        # "Possible <name>" (Aryaman 2026-07-28) — the SAME title shape
        # as every other event. Absence detectors are naturally NAMED as
        # the miss ("Missed evening medication"), so the system must add
        # NO negation of its own: not "Not detected:", not "not seen by".
        title = f"Possible {det.name}"
        coverage_note = (
            " Coverage was low during this window, so treat this as uncertain."
            if uptime_pct < 60 else ""
        )
        # Narrative semantics (Aryaman 2026-07-28): the detector WATCHES
        # for a routine happening; the event reports that it didn't.
        # Never embed the detector's (miss-phrased) name in a negated
        # sentence — "didn't detect 'Missed medication'" reads as the
        # medication having been taken.
        narrative = (
            f"Memoria watched for this between {ws.strftime('%-I:%M %p')} "
            f"and {we.strftime('%-I:%M %p')} today, and the routine it "
            f"looks for didn't happen by the deadline. Cameras were "
            f"actively monitoring about {uptime_pct}% of that window."
            f"{coverage_note}"
        )
        return Event(
            id=uuid4(),
            patient_id=patient.id,
            detector_id=det.id,
            camera_id=None,                 # absence spans the patient, not one camera
            category=det.category,
            source=det.source,
            icon_name=det.icon_name,
            title=title,
            location="Scheduled check",
            occurred_at=deadline_utc,
            has_clip=False,
            # No camera, no clip, EVER — stamp the suppression contract
            # so apps show the no-clip placeholder instead of an endless
            # "Generating clip…" (2026-07-28, the med-absence event).
            clip_suppressed=True,
            narrative=narrative,
            opened=False,
            reviewed=False,
            review_choice=None,
            review_notes=None,
            reviewed_at=None,
            created_at=datetime.now(timezone.utc),
        )
