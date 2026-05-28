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

Timezone is per-caregiver (detector → patient → caregiver.timezone).

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
from src.services import email, push
from src.settings import Settings

log = logging.getLogger(__name__)

_FALLBACK_TZ = "America/New_York"


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
            stmt = (
                select(Detector)
                .where(
                    Detector.shape == "scheduled",
                    Detector.is_on.is_(True),
                    Detector.identifier.is_not(None),
                )
            )
            detectors = (await session.execute(stmt)).scalars().all()
            for det in detectors:
                try:
                    await self._evaluate_one(session, det)
                except Exception:  # noqa: BLE001
                    log.exception("scheduled_runner.evaluate_failed detector=%s", det.id)

    async def _resolve_tz(self, session: AsyncSession, caregiver_id) -> ZoneInfo:
        cg = await session.get(Caregiver, caregiver_id)
        name = (cg.timezone if cg else None) or _FALLBACK_TZ
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            log.warning("scheduled_runner.bad_tz %r — falling back", name)
            return ZoneInfo(_FALLBACK_TZ)

    async def _evaluate_one(self, session: AsyncSession, det: Detector) -> None:
        cfg = det.schedule_config or {}
        cadence = cfg.get("cadence")
        ws = _parse_hhmm(cfg.get("window_start"))
        we = _parse_hhmm(cfg.get("window_end"))
        dl = _parse_hhmm(cfg.get("deadline"))
        if cadence not in ("daily", "weekly") or ws is None or we is None or dl is None:
            log.warning("scheduled_runner.incomplete_config detector=%s", det.id)
            return
        if ws >= we:
            # v1 doesn't support windows crossing midnight.
            log.warning("scheduled_runner.inverted_window detector=%s", det.id)
            return

        patient = await session.get(Patient, det.patient_id)
        if patient is None:
            return
        tz = await self._resolve_tz(session, patient.caregiver_id)
        now_local = datetime.now(tz)
        today = now_local.date()

        # Weekly: only on configured weekdays (0=Mon..6=Sun).
        if cadence == "weekly":
            days = cfg.get("days_of_week") or []
            if now_local.weekday() not in days:
                return

        # Deadline not yet reached today → wait.
        deadline_local = datetime.combine(today, dl, tzinfo=tz)
        if now_local < deadline_local:
            return

        # Single-fire: already evaluated today.
        if det.last_scheduled_eval_date == today:
            return

        # Window → UTC for the analyses query (captured_at is UTC).
        start_utc = datetime.combine(today, ws, tzinfo=tz).astimezone(timezone.utc)
        end_utc = datetime.combine(today, we, tzinfo=tz).astimezone(timezone.utc)
        detected_key = f"{det.identifier}_detected"

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

        # Mark evaluated regardless of outcome (single-fire for the day).
        det.last_scheduled_eval_date = today

        if seen:
            await session.commit()
            log.info("scheduled_runner.seen detector=%s date=%s — no alert", det.id, today)
            return

        # Absent → uptime context, then fire an absence event.
        pct = await self._uptime_pct(session, patient.id, start_utc, end_utc)
        event = self._build_absence_event(
            det=det, patient=patient, ws=ws, we=we, deadline_utc=deadline_local.astimezone(timezone.utc),
            uptime_pct=pct,
        )
        session.add(event)
        await session.commit()
        log.info(
            "scheduled_runner.fired detector=%s date=%s event=%s uptime=%d%%",
            det.id, today, event.id, pct,
        )

        # Reuse the existing fire-and-forget dispatchers (both work with
        # camera_id=None — they key off the event's patient).
        asyncio.create_task(push.dispatch_event_pushes(
            sessionmaker=self._sessionmaker, event_ids=[event.id], settings=self._settings,
        ))
        asyncio.create_task(email.dispatch_event_emails(
            sessionmaker=self._sessionmaker, event_ids=[event.id], settings=self._settings,
        ))

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
        title = f"{det.name}: possibly not seen today"
        narrative = (
            f"No '{det.name}' was detected today between "
            f"{ws.strftime('%H:%M')} and {we.strftime('%H:%M')}. "
            f"Detection was active about {uptime_pct}% of that window — "
            f"{'treat with caution if low' if uptime_pct < 60 else 'good coverage'}."
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
            narrative=narrative,
            opened=False,
            reviewed=False,
            review_choice=None,
            review_notes=None,
            reviewed_at=None,
            created_at=datetime.now(timezone.utc),
        )
