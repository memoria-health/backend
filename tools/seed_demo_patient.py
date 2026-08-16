"""Demo seed: Margaret Ellis, deck-quality — a hero day for screenshots.

Creates a SYNTHETIC patient (age 82) on the caregiver account that owns
'Aryaman Darda', detectors across all four categories, and ~2 weeks of
believable history. Yesterday is the HERO day: gapped sleep band, a
2:14 AM front-door moment to review, an afternoon agitation spell,
meals + company at lunch, medication done, water counted, TV in the
evening. One real pre-recorded fall clip is attached to the fall event
47 days back so the Events page plays actual video.

    uv run python tools/seed_demo_patient.py            # refuses if exists
    uv run python tools/seed_demo_patient.py --wipe     # delete + reseed
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from datetime import date as date_t, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zoneinfo import ZoneInfo

from sqlalchemy import text

from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.detector import Detector
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services import storage
from src.services.detection import catalog, catalog_v2
from src.settings import get_settings

NOW = datetime.now(timezone.utc)
# Filled in run() from the caregiver row — every seeded moment is
# constructed on the CAREGIVER'S clock, because the journal renders in
# that timezone and "2:14 AM" must stay 2:14 AM.
TZ = timezone.utc
rng = random.Random(20260815)

PRESET_KEYS = ["fall", "agitation", "wandering", "medication",
               "sleep", "activity", "meal_eaten", "stove_left_on"]

WATER_IDENT = "custom_water01"

# A real pre-recorded clip already in the local incident-clips bucket —
# reused on Margaret's fall event so the detail sheet plays actual video.
FALL_CLIP_KEY = "events/e5a0d6c7-7af8-457c-b1a3-eab0fbed553b.mp4"


def _preset(patient_id, key, i):
    e = next(x for x in catalog.CATALOG if x.key == key)
    ts = NOW - timedelta(days=60) + timedelta(microseconds=i)
    return Detector(
        id=uuid4(), patient_id=patient_id, name=e.name, icon_name=e.icon_name,
        description=e.description, category=e.category, source="preset",
        alert_trigger=e.alert_trigger, is_sustained=e.is_sustained,
        duration_seconds=e.duration_seconds,
        where_rooms=None, when_scope=e.when_scope,
        sensitivity="standard", is_on=True, created_at=ts, updated_at=ts,
        custom_prompt=None, identifier=e.key, prompt_fragment=e.prompt_fragment,
        schema_json=e.schema_json, always_on=False, schedule_config=None,
        **catalog_v2.row_kwargs(e.key),
    )


def _custom(patient_id, name, category, i, *, identifier=None,
            alert_trigger="leading", is_sustained=False,
            duration_seconds=None, when_scope="ALWAYS", severity=40,
            icon="sparkles"):
    ts = NOW - timedelta(days=45) + timedelta(microseconds=i)
    return Detector(
        id=uuid4(), patient_id=patient_id, name=name, icon_name=icon,
        description=name, category=category, source="custom",
        alert_trigger=alert_trigger, is_sustained=is_sustained,
        duration_seconds=duration_seconds, where_rooms=None,
        when_scope=when_scope, sensitivity="standard", is_on=True,
        created_at=ts, updated_at=ts, custom_prompt=name,
        identifier=identifier or f"custom_{uuid4().hex[:8]}",
        prompt_fragment=f"Detect: {name}.",
        schema_json=None, always_on=False, schedule_config=None,
        subject_scope="patient", severity=severity,
    )


def _event(patient_id, det, occurred_at, camera, narrative, *,
           reviewed=True, clip_key=None):
    return Event(
        id=uuid4(), patient_id=patient_id, category=det.category,
        source=det.source, icon_name=det.icon_name,
        title=f"Possible {det.name}", location=camera.room,
        occurred_at=occurred_at, has_clip=clip_key is not None,
        narrative=narrative,
        opened=reviewed, reviewed=reviewed, created_at=occurred_at,
        detector_id=det.id, camera_id=camera.id,
        review_choice="correct" if reviewed else None,
        clip_storage_key=clip_key,
    )


def _at(days_back: int, h: int, m: int = 0) -> datetime:
    local = datetime.now(TZ) - timedelta(days=days_back)
    return local.replace(hour=h, minute=m, second=0, microsecond=0)


async def run(wipe: bool) -> None:
    sm = get_sessionmaker()
    async with sm() as s:
        caregiver_id = (await s.execute(text(
            "select coalesce((select caregiver_id from patients where "
            "full_name like 'Aryaman%' limit 1), (select id from caregivers "
            "order by created_at limit 1))"))).scalar_one()
        global TZ
        tzname = (await s.execute(text(
            "select timezone from caregivers where id = :c"),
            {"c": caregiver_id})).scalar_one_or_none()
        try:
            TZ = ZoneInfo(tzname or "UTC")
        except Exception:
            TZ = timezone.utc
        existing = (await s.execute(text(
            "select id from patients where full_name in "
            "('Jane Patel', 'Margaret Ellis')"))).scalars().all()
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
            date_of_birth=date_t(1944, 3, 9), gender="female",
        ))
        await s.flush()
        await s.execute(text(
            "insert into patient_members (patient_id, caregiver_id, role) "
            "values (:p, :c, 'owner')"), {"p": pid, "c": caregiver_id})

        # Portrait survives reseeds: drop a JPEG at this path once and
        # every seed run re-attaches it to the fresh row.
        portrait = Path("/tmp/margaret-portrait.jpg")
        if portrait.exists():
            key = await storage.upload_patient_photo(
                patient_id=pid, image_bytes=portrait.read_bytes(),
                content_type="image/jpeg", settings=get_settings())
            await s.execute(text(
                "update patients set photo_storage_key = :k where id = :p"),
                {"k": key, "p": pid})

        cams = []
        for i, room in enumerate(["Living Room", "Bedroom", "Kitchen"]):
            cam = Camera(id=uuid4(), patient_id=pid, display_id=i + 1,
                         room=room, icon_name="web.camera", online=True,
                         created_at=NOW - timedelta(days=60),
                         # Paired + recently seen, or the startup sweep
                         # flips the demo sensors offline.
                         api_key_hash="seeded-demo", app_present=True)
            cams.append(cam)
            s.add(cam)
        living, bedroom, kitchen = cams

        dets = {k: _preset(pid, k, i) for i, k in enumerate(PRESET_KEYS)}
        dets["wandering"].when_scope = "10 PM - 6 AM"
        dets["medication"].schedule_config = {
            "cadence": "daily", "window_start": "07:00",
            "window_end": "21:00", "deadline": "21:00",
        }
        company = _custom(pid, "Company at meals", "care", 100,
                          icon="person.2")
        tv = _custom(pid, "TV after 11 PM", "behaviour", 101,
                     alert_trigger="duration", is_sustained=True,
                     duration_seconds=900, when_scope="11 PM - 3 AM",
                     icon="tv")
        water = _custom(pid, "Drinking water", "routine", 102,
                        identifier=WATER_IDENT,
                        alert_trigger="record_only", icon="drop")
        for d in [*dets.values(), company, tv, water]:
            s.add(d)
        await s.flush()

        events: list[Event] = []

        # ── The fall, 47 days ago — with the real recorded clip.
        events.append(_event(
            pid, dets["fall"], _at(47, 14, 12), living,
            "Margaret lost her balance stepping away from the armchair and "
            "sat down hard on the rug. She stayed alert, and got up with "
            "help from the sofa arm after about a minute.",
            clip_key=FALL_CLIP_KEY))

        # ── HERO DAY (yesterday) ─────────────────────────────────
        events.append(_event(
            pid, dets["wandering"], _at(1, 2, 14), living,
            "The front door opened at 2:14 AM and Margaret stepped toward "
            "the porch in her nightgown. She came back inside on her own "
            "after about a minute.", reviewed=False))
        events.append(_event(
            pid, dets["agitation"], _at(1, 17, 10), living,
            "Margaret paced between the couch and the window for around "
            "twenty minutes, wringing her hands and checking the door.",
            reviewed=False))
        events.append(_event(
            pid, dets["meal_eaten"], _at(1, 12, 35), kitchen,
            "Margaret ate lunch at the kitchen table, finishing most of "
            "her soup and toast."))
        events.append(_event(
            pid, dets["meal_eaten"], _at(1, 18, 20), kitchen,
            "Margaret had dinner at the table and cleared her own plate."))
        events.append(_event(
            pid, company, _at(1, 12, 40), kitchen,
            "Her daughter sat with her through lunch."))
        events.append(_event(
            pid, tv, _at(1, 23, 25), living,
            "The television stayed on past 11 PM while Margaret dozed on "
            "the couch.", reviewed=False))

        # ── History (days 2-14): believable texture, all reviewed.
        for day in (2, 3, 5, 6, 8, 9, 10, 12, 13, 14):
            events.append(_event(
                pid, dets["meal_eaten"], _at(day, 12, rng.randrange(20, 55)),
                kitchen, "Margaret ate lunch at the kitchen table."))
        for day in (2, 4, 6, 9, 11, 13):
            events.append(_event(
                pid, company, _at(day, 12, rng.randrange(30, 59)), kitchen,
                "Someone sat with Margaret through lunch."))
        for day, h in ((3, 17), (7, 16), (12, 18)):
            events.append(_event(
                pid, dets["agitation"], _at(day, h, rng.randrange(59)),
                living,
                "A stretch of restless pacing in the late afternoon.",
                reviewed=day != 3))
        events.append(_event(
            pid, dets["stove_left_on"], _at(4, 12, 48), kitchen,
            "A burner stayed lit with nothing on it for twelve minutes "
            "after lunch."))
        events.append(_event(
            pid, dets["wandering"], _at(9, 23, 5), living,
            "The front door opened late; Margaret checked the porch and "
            "came back in."))
        events.append(_event(
            pid, dets["wandering"], _at(2, 23, 48), living,
            "The front door opened just before midnight. Margaret stood in "
            "the doorway for a moment before closing it.", reviewed=False))
        for day in (2, 3, 6, 8, 11):
            events.append(_event(
                pid, tv, _at(day, 23, rng.randrange(5, 50)), living,
                "Television on past 11 PM.", reviewed=day != 2))
        # ── So-far-today texture (guarded to the past only).
        for cand in (
            _event(pid, dets["meal_eaten"], _at(0, 8, 10), kitchen,
                   "Margaret had breakfast at the kitchen table."),
            _event(pid, company, _at(0, 8, 15), kitchen,
                   "Her daughter shared coffee with her at breakfast."),
            _event(pid, dets["stove_left_on"], _at(0, 11, 5), kitchen,
                   "A burner stayed lit with an empty pan for eight "
                   "minutes after breakfast.", reviewed=False),
            _event(pid, dets["agitation"], _at(0, 10, 20), living,
                   "Margaret circled the living room several times, "
                   "picking things up and putting them down.",
                   reviewed=False),
        ):
            if cand.occurred_at < datetime.now(TZ):
                events.append(cand)
        for ev in events:
            s.add(ev)
        await s.flush()

        # ── Sleep: 15 nights incl. last night; hero night (day 1)
        # carries two real wake intervals.
        for day in range(0, 15):
            night = (datetime.now(TZ) - timedelta(days=day + 1)).date()
            bed = datetime(night.year, night.month, night.day, 22,
                           rng.randrange(20, 55), tzinfo=TZ)
            wake = bed + timedelta(hours=rng.uniform(7.4, 8.4))
            oob = []
            dist = rng.randint(0, 1)
            if day == 1:
                oob = [
                    {"start": (bed + timedelta(hours=3, minutes=20)).isoformat(),
                     "end": (bed + timedelta(hours=3, minutes=38)).isoformat()},
                    {"start": (bed + timedelta(hours=6, minutes=5)).isoformat(),
                     "end": (bed + timedelta(hours=6, minutes=17)).isoformat()},
                ]
                dist = 2
            elif day in (4, 9):
                oob = [
                    {"start": (bed + timedelta(hours=4)).isoformat(),
                     "end": (bed + timedelta(hours=4, minutes=14)).isoformat()},
                ]
                dist = 1
            await s.execute(text("""
                insert into sleep_nights (patient_id, night_date, bedtime,
                    sleep_onset, wake_time, duration_minutes, disturbances,
                    out_of_bed_events, flags)
                values (:p, :d, :b, :o, :w, :m, :dist,
                        cast(:oob as jsonb), '[]'::jsonb)
                on conflict (patient_id, night_date) do nothing
                """), {"p": pid, "d": night, "b": bed,
                       "o": bed + timedelta(minutes=12), "w": wake,
                       "m": int((wake - bed).total_seconds() // 60),
                       "dist": dist,
                       "oob": __import__("json").dumps(oob)})

        # ── Analyses: 14 days of daytime ticks. Activity states shape a
        # believable ratio (~2 h active); water ticks feed the counter.
        analyses = []
        now_local = datetime.now(TZ)
        for day in range(0, 15):
            water_times = sorted(rng.sample(range(8 * 60, 20 * 60),
                                            rng.randint(4, 7)))
            water_marks = {t // 60: t % 60 for t in water_times}
            for half in range(15, 42):   # 7:30 → 21:00 local half-hours
                h, m = divmod(half * 30, 60)
                base = _at(day, h, m)
                for j in range(3):       # one tick / 10 min keeps rows sane
                    t = base + timedelta(minutes=10 * j)
                    hour_frac = h + m / 60
                    active_p = (0.55 if 8 <= hour_frac < 10
                                else 0.45 if 16 <= hour_frac < 17.5
                                else 0.10)
                    state = ("active" if rng.random() < active_p
                             else rng.choice(("sitting_still", "lying",
                                              "sitting_moving")))
                    if t >= now_local:
                        continue   # today only fills up to "now"
                    fields = {"activity_state": state}
                    if water_marks.get(h) is not None and m <= water_marks[h] < m + 30 and j == 0:
                        fields[f"{WATER_IDENT}_detected"] = True
                    analyses.append(Analysis(
                        id=uuid4(), caregiver_id=caregiver_id, patient_id=pid,
                        camera_id=living.id, provider="skipped",
                        description="seeded tick", detector_fields=fields,
                        people_present=True,
                        motion_in_clip=state == "active",
                        state_after="open", captured_at=t, created_at=t))
        s.add_all(analyses)

        await s.commit()
        print(f"Seeded Margaret Ellis ({pid}) — {len(events)} events, "
              f"14 sleep nights, {len(analyses)} analyses, "
              f"{3 + len(dets)} detectors, fall clip attached.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wipe", action="store_true")
    asyncio.run(run(wipe=p.parse_args().wipe))
