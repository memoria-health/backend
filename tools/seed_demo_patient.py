"""Demo seed: Margaret Ellis + historical data for every widget shape.

Local/dev only — creates a SYNTHETIC patient on the caregiver account that owns
'Aryaman Darda' (falls back to the first caregiver),
detectors across all four categories (presets + two customs, incl. a
Care custom since Care has no built-ins), compiled widget specs, and
~60 days of history shaped so each chart has something honest to show:

  fall (streak 47d) · wandering (night exits, 2 of 14 nights) ·
  stove (occurrence dots) · medication (2 missed evening doses) ·
  meals (occurrence + ratio) · sleep (7 nights, 2 restless) ·
  activity (30d trend, recent dip) · agitation (4-8 PM cluster, up
  vs prior week) · "Company at meals" care custom (ratio) ·
  "TV after 11 PM" behaviour custom (duration bars)

    uv run python tools/seed_demo_patient.py            # refuses if Jane exists
    uv run python tools/seed_demo_patient.py --wipe     # delete + reseed
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services.dashboard.widget_compiler import apply_widget
from src.services.detection import catalog, catalog_v2

NOW = datetime.now(timezone.utc)
rng = random.Random(20260815)

PRESET_KEYS = ["fall", "agitation", "wandering", "medication",
               "sleep", "activity", "meal_eaten", "stove_left_on"]


def _preset(patient_id, key, i, **over):
    e = next(x for x in catalog.CATALOG if x.key == key)
    ts = NOW - timedelta(days=60) + timedelta(microseconds=i)
    det = Detector(
        id=uuid4(), patient_id=patient_id, name=e.name, icon_name=e.icon_name,
        description=e.description, category=e.category, source="preset",
        alert_trigger=e.alert_trigger, is_sustained=e.is_sustained,
        duration_seconds=e.duration_seconds,
        where_rooms=None, when_scope=over.pop("when_scope", e.when_scope),
        sensitivity="standard", is_on=True, created_at=ts, updated_at=ts,
        custom_prompt=None, identifier=e.key, prompt_fragment=e.prompt_fragment,
        schema_json=e.schema_json, always_on=False,
        schedule_config=over.pop("schedule_config", None),
        **catalog_v2.row_kwargs(e.key),
    )
    apply_widget(det)
    return det


def _custom(patient_id, name, category, i, *, alert_trigger="leading",
            is_sustained=False, duration_seconds=None, when_scope="ALWAYS",
            severity=40, icon="sparkles"):
    ts = NOW - timedelta(days=45) + timedelta(microseconds=i)
    det = Detector(
        id=uuid4(), patient_id=patient_id, name=name, icon_name=icon,
        description=name, category=category, source="custom",
        alert_trigger=alert_trigger, is_sustained=is_sustained,
        duration_seconds=duration_seconds, where_rooms=None,
        when_scope=when_scope, sensitivity="standard", is_on=True,
        created_at=ts, updated_at=ts, custom_prompt=name,
        identifier=f"custom_{uuid4().hex[:8]}",
        prompt_fragment=f"Detect: {name}.",
        schema_json=None, always_on=False, schedule_config=None,
        subject_scope="patient", severity=severity,
    )
    apply_widget(det)
    return det


def _event(patient_id, det, occurred_at, camera, narrative):
    return Event(
        id=uuid4(), patient_id=patient_id, category=det.category,
        source=det.source, icon_name=det.icon_name,
        title=f"Possible {det.name}", location=camera.room,
        occurred_at=occurred_at, has_clip=False, narrative=narrative,
        opened=True, reviewed=True, created_at=occurred_at,
        detector_id=det.id, camera_id=camera.id, review_choice="correct",
    )


async def run(wipe: bool) -> None:
    sm = get_sessionmaker()
    async with sm() as s:
        caregiver_id = (await s.execute(
            text("select coalesce((select caregiver_id from patients where full_name like 'Aryaman%' limit 1), (select id from caregivers order by created_at limit 1))"),
        )).scalar_one()
        existing = (await s.execute(text(
            "select id from patients where full_name in ('Jane Patel', 'Margaret Ellis')"),
        )).scalars().all()
        if existing and not wipe:
            print("Margaret Ellis already exists — pass --wipe to reseed.")
            return
        for pid in existing:
            await s.execute(text("delete from patients where id = :p"), {"p": pid})

        pid = uuid4()
        s.add(Patient(
            id=pid, caregiver_id=caregiver_id, full_name="Margaret Ellis",
            preferred_name="Margaret", relation="parent", state="calm",
            status_line="All quiet", cameras_paused=False,
            created_at=NOW - timedelta(days=60), updated_at=NOW,
        ))
        await s.flush()   # children below FK onto the patient row
        # Access flows through patient_members, NOT patients.caregiver_id —
        # without this row the roster never lists her.
        await s.execute(text(
            "insert into patient_members (patient_id, caregiver_id, role) "
            "values (:p, :c, 'owner')"), {"p": pid, "c": caregiver_id})

        cams = []
        for i, room in enumerate(["Living Room", "Bedroom", "Kitchen"]):
            cam = Camera(id=uuid4(), patient_id=pid, display_id=i + 1,
                         room=room, icon_name="web.camera", online=True,
                         created_at=NOW - timedelta(days=60))
            cams.append(cam)
            s.add(cam)
        living, bedroom, kitchen = cams

        dets = {k: _preset(pid, k, i) for i, k in enumerate(PRESET_KEYS)}
        # Wandering watches the night; medication has two daily deadlines.
        dets["wandering"].when_scope = "10 PM - 6 AM"
        dets["medication"].schedule_config = {
            "cadence": "daily", "window_start": "07:00",
            "window_end": "21:00", "deadline": "21:00",
        }
        for k in ("wandering", "medication"):
            apply_widget(dets[k])
        company = _custom(pid, "Company at meals", "care", 100)
        tv = _custom(pid, "TV after 11 PM", "behaviour", 101,
                     alert_trigger="duration", is_sustained=True,
                     duration_seconds=900, when_scope="11 PM - 3 AM",
                     icon="tv")
        for d in [*dets.values(), company, tv]:
            s.add(d)
        await s.flush()   # events FK onto detectors + cameras

        events: list[Event] = []
        # Falls: one 47 days ago, one 61 days before that → streak + gap.
        events.append(_event(pid, dets["fall"], NOW - timedelta(days=47),
                             living, "Margaret lowered herself to the floor."))
        # Night exits: 2 of the last 14 nights, small hours.
        for day, hour in ((3, 2), (3, 3), (9, 23), (9, 2)):
            events.append(_event(
                pid, dets["wandering"],
                (NOW - timedelta(days=day)).replace(
                    hour=hour, minute=rng.randrange(50)),
                living, "Front door opened."))
        # Agitation: sundowning cluster 4-8 PM, heavier this week.
        for day in (1, 2, 2, 4, 5, 9, 11):
            hour = rng.choice((16, 17, 18, 19))
            events.append(_event(
                pid, dets["agitation"],
                (NOW - timedelta(days=day)).replace(
                    hour=hour, minute=rng.randrange(60)),
                living, "Pacing between the couch and the window."))
        # Stove left on: 3 scattered days.
        for day in (2, 6, 11):
            events.append(_event(
                pid, dets["stove_left_on"],
                (NOW - timedelta(days=day)).replace(hour=12, minute=40),
                kitchen, "Burner lit with no pan."))
        # Medication misses: adherence dots — 2 missed evenings.
        for day in (4, 9):
            events.append(_event(
                pid, dets["medication"],
                (NOW - timedelta(days=day)).replace(hour=21, minute=5),
                kitchen, "Evening dose not seen by 9 PM."))
        # Meals seen most days (11 of 14).
        for day in range(1, 15):
            if day in (5, 8, 12):
                continue
            events.append(_event(
                pid, dets["meal_eaten"],
                (NOW - timedelta(days=day)).replace(hour=13, minute=10),
                kitchen, "Margaret ate lunch at the table."))
        # Care custom: company at meals, 4 of last 7 days.
        for day in (1, 2, 4, 6):
            events.append(_event(
                pid, company,
                (NOW - timedelta(days=day)).replace(hour=13, minute=20),
                kitchen, "Someone sat with Margaret through lunch."))
        # TV after 11 PM: 5 of last 7 nights, 1-3 instances each.
        for day in (1, 2, 3, 5, 6):
            for _ in range(rng.randint(1, 3)):
                events.append(_event(
                    pid, tv,
                    (NOW - timedelta(days=day)).replace(
                        hour=23, minute=rng.randrange(55)),
                    living, "Television on past 11 PM."))
        for ev in events:
            s.add(ev)

        # Raw SQL below needs the ORM rows on disk first.
        await s.flush()

        # Sleep nights: 7 rows, 2 restless.
        for day in range(1, 8):
            night = (NOW - timedelta(days=day)).date()
            bed = datetime(night.year, night.month, night.day, 22,
                           rng.randrange(15, 55), tzinfo=timezone.utc)
            wake = bed + timedelta(hours=rng.uniform(7.4, 8.6))
            restless = day in (2, 5)
            await s.execute(text("""
                insert into sleep_nights (patient_id, night_date, bedtime,
                    sleep_onset, wake_time, duration_minutes, disturbances,
                    out_of_bed_events, flags)
                values (:p, :d, :b, :o, :w, :m, :dist, '[]'::jsonb, '[]'::jsonb)
                on conflict (patient_id, night_date) do nothing
                """), {"p": pid, "d": night, "b": bed,
                       "o": bed + timedelta(minutes=12), "w": wake,
                       "m": int((wake - bed).total_seconds() // 60),
                       "dist": 3 if restless else rng.randint(0, 1)})

        # Analyses: 30 days x hourly ticks → the activity trend, with a
        # dip over the last 5 days.
        analyses = []
        for day in range(30):
            base = 0.55 if day >= 5 else 0.34 - 0.03 * (5 - day)
            for hour in range(8, 22):
                captured = (NOW - timedelta(days=day)).replace(
                    hour=hour, minute=30)
                active = rng.random() < base
                analyses.append(Analysis(
                    id=uuid4(), caregiver_id=caregiver_id, patient_id=pid,
                    camera_id=living.id, provider="skipped",
                    description="seeded tick", detector_fields={},
                    people_present=True, motion_in_clip=active,
                    state_after="open", captured_at=captured,
                    created_at=captured))
        s.add_all(analyses)

        await s.commit()
        print(f"Seeded Margaret Ellis ({pid}) — {len(events)} events, "
              f"7 sleep nights, {len(analyses)} analyses, "
              f"{2 + len(dets)} detectors.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wipe", action="store_true")
    asyncio.run(run(wipe=p.parse_args().wipe))
