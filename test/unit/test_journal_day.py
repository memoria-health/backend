"""Journal day builders — the pure half of the Daily Journal.

Contract (docs/plans/2026-08-16-daily-journal.md): each detector
contributes one line whose shape is a pure function of its taxonomy.
Strips are sparse and only where the map is clean; sleep rides a
noon-to-noon axis; activity blocks follow the pinned definition
(moving = standing+active; a block = 30-min bucket with >=10 moving
minutes); no summary sentence anywhere.
"""

from datetime import datetime, timezone, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from src.services.journal import day as jd

TZ = ZoneInfo("America/Los_Angeles")
DAY = datetime(2026, 8, 15, tzinfo=TZ).date()


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def _local(h, mi=0, day=DAY):
    return datetime(day.year, day.month, day.day, h, mi, tzinfo=TZ)


def _det(**over):
    base = dict(
        id="d1", name="Falls", category="safety", alert_trigger="leading",
        is_sustained=False, schedule_config=None, severity=95,
        identifier="fall",
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── axis math ────────────────────────────────────────────────────────

def test_hour_x_midnight_axis_is_local_fractional_hours() -> None:
    # 2:30 PM local == 14.5 on the midnight axis
    assert jd.hour_x(_local(14, 30), TZ, axis="midnight") == 14.5


def test_hour_x_noon_axis_wraps_evening_before_morning() -> None:
    # 10:42 PM local = 10.7 h after noon; 6:47 AM = 18.78 h after noon
    assert abs(jd.hour_x(_local(22, 42), TZ, axis="noon") - 10.7) < 0.01
    assert abs(jd.hour_x(_local(6, 47), TZ, axis="noon") - 18.783) < 0.01


# ── activity (pinned definition) ─────────────────────────────────────

def test_activity_blocks_need_ten_moving_minutes_per_half_hour() -> None:
    # 9:00-9:30 bucket: 70 moving ticks (11.7 min) → block.
    # 10:00-10:30 bucket: 30 moving ticks (5 min) → no block.
    ticks = [( _local(9, 0) + timedelta(seconds=10 * i), "active")
             for i in range(70)]
    ticks += [(_local(10, 0) + timedelta(seconds=10 * i), "standing")
              for i in range(30)]
    ticks += [(_local(10, 5) + timedelta(seconds=10 * i), "lying")
              for i in range(60)]
    entry = jd.activity_line(ticks, TZ)
    assert entry["strip"]["spans"] == [[9.0, 9.5]]
    # 100 moving ticks ≈ 16.7 min → "about 15 m" rounding to 15-min steps
    assert "up and moving" in entry["line"]["primary"]


def test_activity_reports_hours_when_large() -> None:
    ticks = [(_local(9, 0) + timedelta(seconds=10 * i), "active")
             for i in range(1080)]  # 3 h of moving ticks
    entry = jd.activity_line(ticks, TZ)
    assert entry["line"]["primary"] == "About 3 h up and moving"


# ── sleep (noon axis) ────────────────────────────────────────────────

def test_sleep_line_spans_noon_axis_and_counts_wakeups() -> None:
    row = {
        "bedtime": _local(22, 42, DAY - timedelta(days=1)),
        "wake_time": _local(6, 47),
        "duration_minutes": 485, "disturbances": 2,
    }
    entry = jd.sleep_line(row, TZ)
    s, e = entry["strip"]["spans"][0]
    assert abs(s - 10.7) < 0.01 and abs(e - 18.783) < 0.01
    assert entry["strip"]["axis"] == "noon"
    assert "10:42" in entry["line"]["primary"]
    assert entry["line"]["meta"] == "2 wake-ups"


def test_sleep_line_absent_row_reports_no_data() -> None:
    entry = jd.sleep_line(None, TZ)
    assert entry["line"]["primary"] == "No sleep recorded"
    assert entry["strip"] is None


# ── point alerts ─────────────────────────────────────────────────────

def test_quiet_point_line_has_no_strip() -> None:
    entry = jd.point_line(_det(), [], TZ, streak_days=47)
    assert entry["line"]["primary"] == "No falls"
    assert entry["line"]["status"] == "ok"
    assert entry["line"]["meta"] == "47 days"
    assert entry["strip"] is None


def test_event_day_gets_marks_and_review_status() -> None:
    ev = SimpleNamespace(id="e1", occurred_at=_local(2, 14), location="Living Room",
                        reviewed=False)
    entry = jd.point_line(_det(name="Front door"), [ev], TZ, streak_days=None)
    assert entry["line"]["status"] == "review"
    assert "2:14 AM" in entry["line"]["meta"]
    assert abs(entry["strip"]["marks"][0] - (2 + 14 / 60)) < 0.01
    assert entry["review_event_id"] == "e1"


def test_reviewed_event_day_is_calm() -> None:
    ev = SimpleNamespace(id="e1", occurred_at=_local(9, 0), location="Kitchen",
                        reviewed=True)
    entry = jd.point_line(_det(), [ev], TZ, streak_days=None)
    assert entry["line"]["status"] == "none"


# ── absence (deadline) ───────────────────────────────────────────────

def test_absence_taken_when_no_miss_event() -> None:
    det = _det(name="Medication", category="routine", alert_trigger="absence",
               schedule_config={"deadline": "21:00"})
    entry = jd.absence_line(det, [], TZ)
    assert entry["line"]["status"] == "ok"
    assert "by 9 PM" in entry["line"]["primary"]


def test_absence_missed_when_event_fired() -> None:
    det = _det(name="Medication", alert_trigger="absence",
               schedule_config={"deadline": "21:00"})
    ev = SimpleNamespace(id="e2", occurred_at=_local(21, 5), location="Kitchen",
                        reviewed=False)
    entry = jd.absence_line(det, [ev], TZ)
    assert entry["line"]["status"] == "review"
    assert "Missed" in entry["line"]["primary"]


# ── episodes + counting records ──────────────────────────────────────

def test_episode_line_lists_times_no_fake_spans() -> None:
    evs = [SimpleNamespace(id="e3", occurred_at=_local(17, 10),
                           location="Living Room", reviewed=True)]
    entry = jd.episode_line(_det(name="Agitation", category="behaviour"),
                            evs, TZ)
    assert entry["line"]["primary"] == "1 episode"
    assert "5:10 PM" in entry["line"]["meta"]
    assert entry["strip"]["marks"]          # dots, not fabricated spans
    assert entry["strip"]["spans"] == []


def test_count_line_counts_detected_ticks() -> None:
    times = [_local(10, 0), _local(13, 30), _local(16, 45)]
    entry = jd.count_line(_det(name="Water", category="routine"), times, TZ)
    assert entry["line"]["primary"] == "3 times"
    assert len(entry["strip"]["marks"]) == 3


# ── assembly ─────────────────────────────────────────────────────────

def test_sections_fixed_order_and_severity_within() -> None:
    entries = [
        {"category": "behaviour", "severity": 40, "title": "B"},
        {"category": "safety", "severity": 95, "title": "Falls"},
        {"category": "safety", "severity": 100, "title": "Choking"},
    ]
    sections = jd.assemble_sections(entries)
    assert [s["category"] for s in sections] == ["safety", "behaviour"]
    assert [e["title"] for e in sections[0]["entries"]] == ["Choking", "Falls"]


def test_duration_line_merges_contiguous_ticks_into_sessions() -> None:
    # 40 min of contiguous TV ticks + a separate 20-min run later.
    times = [_local(23, 20) + timedelta(seconds=10 * i) for i in range(240)]
    times += [_local(21, 0) + timedelta(seconds=10 * i) for i in range(120)]
    det = _det(name="TV after 11 PM", category="behaviour",
               alert_trigger="record_only", is_sustained=True)
    entry = jd.duration_line(det, sorted(times), TZ)
    assert entry["line"]["primary"] == "About 1 h"
    assert len(entry["strip"]["spans"]) == 2
    s, e = entry["strip"]["spans"][1]
    assert abs(s - (23 + 20 / 60)) < 0.01


def test_build_entries_dispatches_by_taxonomy() -> None:
    dets = [
        _det(id="p1", name="Falls", severity=95),
        _det(id="a1", name="Medication", category="routine",
             alert_trigger="absence", schedule_config={"deadline": "21:00"}),
        _det(id="s1", name="Sleep", category="routine", identifier="sleep",
             alert_trigger="record_only"),
        _det(id="c1", name="Water", category="routine",
             identifier="custom_ab", alert_trigger="record_only"),
        _det(id="e1", name="Agitation", category="behaviour",
             alert_trigger="leading", is_sustained=True, severity=60),
    ]
    entries = jd.build_entries(
        detectors=dets, events_by_detector={}, tz=TZ,
        sleep_row=None, activity_ticks=[],
        detected_times_by_identifier={"custom_ab": [_local(10, 0)]},
        streak_days_by_detector={"p1": 12},
    )
    kinds = {e["title"]: e["kind"] for e in entries}
    assert kinds["Falls"] == "point"
    assert kinds["Medication"] == "absence"
    assert kinds["Sleep"] == "record_sleep"
    assert kinds["Water"] == "record_count"
    assert kinds["Agitation"] == "episode"


def test_legacy_absence_routes_by_deadline_not_trigger() -> None:
    det = _det(id="m1", name="Medication adherence", category="routine",
               alert_trigger="leading",
               schedule_config={"deadline": "21:00"})
    entries = jd.build_entries(
        detectors=[det], events_by_detector={}, tz=TZ, sleep_row=None,
        activity_ticks=[], detected_times_by_identifier={},
        streak_days_by_detector={})
    assert entries[0]["kind"] == "absence"


def test_one_day_streak_is_singular() -> None:
    entry = jd.point_line(_det(), [], TZ, streak_days=1)
    assert entry["line"]["meta"] == "1 day"


def test_unwatched_day_never_claims_clean(  ) -> None:
    dets = [_det(id="p1", name="Falls", severity=95),
            _det(id="a1", name="Medication", category="routine",
                 alert_trigger="absence",
                 schedule_config={"deadline": "21:00"})]
    entries = jd.build_entries(
        detectors=dets, events_by_detector={}, tz=TZ, sleep_row=None,
        activity_ticks=[], detected_times_by_identifier={},
        streak_days_by_detector={}, coverage_hours=0)
    for e in entries:
        assert e["line"]["primary"] == "Not watched"
        assert e["line"]["status"] == "none"


def test_unwatched_day_still_shows_events() -> None:
    ev = SimpleNamespace(id="e9", occurred_at=_local(2, 14),
                         location="Living Room", reviewed=False)
    entries = jd.build_entries(
        detectors=[_det(id="p1")], events_by_detector={"p1": [ev]}, tz=TZ,
        sleep_row=None, activity_ticks=[], detected_times_by_identifier={},
        streak_days_by_detector={}, coverage_hours=0)
    assert entries[0]["line"]["status"] == "review"


def test_off_detector_keeps_its_event_history() -> None:
    ev = SimpleNamespace(id="e8", occurred_at=_local(9, 0),
                         location="Kitchen", reviewed=False)
    on_quiet = _det(id="q1", name="Choking", is_on=True)
    off_with_event = _det(id="p1", name="Falls", is_on=False)
    off_quiet = _det(id="p2", name="Wandering", is_on=False)
    entries = jd.build_entries(
        detectors=[on_quiet, off_with_event, off_quiet],
        events_by_detector={"p1": [ev]}, tz=TZ, sleep_row=None,
        activity_ticks=[], detected_times_by_identifier={},
        streak_days_by_detector={}, coverage_hours=14)
    titles = [e["title"] for e in entries]
    assert "Falls" in titles          # history survives the toggle
    assert "Wandering" not in titles  # no quiet claims for off detectors
    assert "Choking" in titles


def test_entries_carry_the_detector_icon() -> None:
    entry = jd.point_line(_det(icon_name="figure.fall"), [], TZ, streak_days=5)
    assert entry["icon"] == "figure.fall"
    assert jd.sleep_line(None, TZ)["icon"] == "bed.double"
