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

# Trailing trigger (Phase E): default hysteresis gap when a detector has no
# episode_gap_seconds, and how far back the falling-edge sweep looks for the
# episode's last sighting (bounded so a long-idle detector doesn't scan all
# history each tick).
_DEFAULT_TRAILING_GAP_SECONDS = 300
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

    async def _resolve_tz(self, session: AsyncSession, caregiver_id) -> ZoneInfo:
        cg = await session.get(Caregiver, caregiver_id)
        name = (cg.timezone if cg else None) or _FALLBACK_TZ
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            log.warning("scheduled_runner.bad_tz %r — falling back", name)
            return ZoneInfo(_FALLBACK_TZ)

    async def _evaluate_one(self, session: AsyncSession, det: Detector) -> None:
        # schedule_config dissolution (Aryaman 2026-07-18): when_scope
        # is THE windows mechanism. The runner reads it first —
        # "MON,WED 8:00 AM - 9:00 AM" ⇒ window 8-9, weekly on those
        # days; no day tokens ⇒ daily; deadline ≡ window end (the
        # window(hit|miss) fire rule evaluates at close). Legacy rows
        # whose when_scope is ALWAYS-flavored fall back to their old
        # schedule_config until the app migration backfills them.
        from src.services.detection.scheduling.when_scope import parse_schedule

        sw = parse_schedule(det.when_scope)
        days: list[int] | None
        if sw is not None:
            ws = time(int(sw.start_min // 60), int(sw.start_min % 60))
            we = time(int(sw.end_min // 60), int(sw.end_min % 60))
            dl = we
            days = list(sw.days) if sw.days is not None else None
            cadence = "weekly" if days else "daily"
        else:
            cfg = det.schedule_config or {}
            cadence = cfg.get("cadence")
            ws = _parse_hhmm(cfg.get("window_start"))
            we = _parse_hhmm(cfg.get("window_end"))
            dl = _parse_hhmm(cfg.get("deadline"))
            days = cfg.get("days_of_week") or []
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
            if now_local.weekday() not in (days or []):
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

        if getattr(self._settings, "detection_v2_decision_enabled", False):
            # v2 (P4, parallel-systems rule): same semantics, but the
            # counting goes through the ONE shared primitive the state
            # entry also uses (scheduling/decision.py) — a single
            # implementation to soak. threshold=None = legacy's
            # no-confidence-gate behavior exactly. Cleanup Ledger row 2
            # retires this runner entirely once proven.
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

        # Mark evaluated regardless of outcome (single-fire for the day).
        det.last_scheduled_eval_date = today

        if seen:
            await session.commit()
            log.info("scheduled_runner.seen detector=%s date=%s — no alert", det.id, today)
            return

        # Absent. But "not detected" is only meaningful if we were actually
        # watching. With no sensor connected — or AI paused / the camera
        # disconnected for the whole window — there are zero analyzing ticks
        # and uptime is 0%. Firing then is a false alarm: you can't say a thing
        # "didn't happen" when you weren't looking at all. Suppress that case.
        # Partial coverage still fires (with the low-coverage caveat in the
        # narrative) — we only drop the genuinely-blind case.
        pct = await self._uptime_pct(session, patient.id, start_utc, end_utc)
        if pct == 0:
            await session.commit()
            log.info(
                "scheduled_runner.no_monitoring detector=%s date=%s — suppressed (0%% coverage)",
                det.id, today,
            )
            return

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

    async def _evaluate_trailing(self, session: AsyncSession, det: Detector) -> None:
        """Fire once a sustained activity has FINISHED — detected recently,
        then ABSENT for >= its hysteresis gap. The falling-edge counterpart of
        the rising-edge promotion. One-shot per episode: guarded by the last
        trailing event vs the episode's last detection, so a whole session
        yields exactly one "finished" alert, and a fresh session re-arms."""
        patient = await session.get(Patient, det.patient_id)
        if patient is None:
            return
        gap = det.episode_gap_seconds or _DEFAULT_TRAILING_GAP_SECONDS
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
        title = f"Not detected: {det.name}"
        coverage_note = (
            " Coverage was low during this window, so treat this as uncertain."
            if uptime_pct < 60 else ""
        )
        narrative = (
            f"Memoria didn't detect \"{det.name}\" between "
            f"{ws.strftime('%H:%M')} and {we.strftime('%H:%M')} today. "
            f"Cameras were actively monitoring about {uptime_pct}% of that window."
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
            narrative=narrative,
            opened=False,
            reviewed=False,
            review_choice=None,
            review_notes=None,
            reviewed_at=None,
            created_at=datetime.now(timezone.utc),
        )
