"""Unit tests for the sleep state machine (pure function, no I/O)."""

from datetime import datetime, timedelta

from src.services.detection.activity_summary import ActivityTick, summarize_day
from src.services.detection.sleep_summary import SleepTick, summarize_night

_N = datetime(2026, 7, 28, 22, 0)          # 10pm, naive-UTC like the DB


def _t(mins: int, in_bed: bool, activity: str = "none", conf: float = 0.9):
    return SleepTick(
        at=_N + timedelta(minutes=mins), in_bed=in_bed,
        activity=activity, confidence=conf,
    )


def test_full_night_with_disturbances_and_out_of_bed() -> None:
    ticks = [
        _t(0, True, "settling"),           # 22:00 into bed
        _t(5, True, "stirring"),           # settling churn — onset scans past
        # silence 22:05 → 01:00 = asleep at 22:05
        _t(185, True, "stirring"),         # 01:05 disturbance
        _t(300, False, "leaving_bed"),     # 03:00 bathroom
        _t(310, True, "settling"),         # 03:10 back in bed
        _t(540, False, "leaving_bed"),     # 07:00 up for the day
    ]
    out = summarize_night(ticks, window_end=_N + timedelta(hours=12))
    assert out["bedtime"] == _t(0, True).at
    assert out["sleep_onset"] == _t(5, True).at
    assert out["disturbances"] == 1
    assert len(out["out_of_bed_events"]) == 1
    assert out["out_of_bed_events"][0]["duration_minutes"] == 10
    assert out["wake_time"] == _t(540, False).at
    # 22:05 → 07:00 = 535 min, minus 10 away
    assert out["duration_minutes"] == 525
    assert out["flags"] == []


def test_no_bed_entry_yields_flag_not_fabrication() -> None:
    out = summarize_night(
        [_t(0, False, "none")], window_end=_N + timedelta(hours=12))
    assert out["flags"] == ["no_bed_entry"]
    assert out["duration_minutes"] is None


def test_incomplete_night_when_window_ends_in_bed() -> None:
    ticks = [_t(0, True, "settling")]
    out = summarize_night(ticks, window_end=_N + timedelta(hours=2))
    assert out["sleep_onset"] == _t(0, True).at
    assert "incomplete_night" in out["flags"]
    assert out["wake_time"] is None


def test_low_confidence_ticks_ignored() -> None:
    out = summarize_night(
        [_t(0, True, "settling", conf=0.3)],
        window_end=_N + timedelta(hours=12))
    assert out["flags"] == ["no_bed_entry"]


def test_wake_needs_confirmation_gap() -> None:
    # Leaving bed 5 min before the window ends: too soon to call wake.
    ticks = [_t(0, True, "settling"), _t(100, False, "leaving_bed")]
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=105))
    assert "incomplete_night" in out["flags"]


def test_activity_day_rollup() -> None:
    base = datetime(2026, 7, 28, 9, 0)
    ticks = [
        ActivityTick(base + timedelta(minutes=i * 10), cam, present, motion)
        for i, (cam, present, motion) in enumerate([
            ("LR", True, True), ("LR", True, False), ("BR", True, True),
            ("LR", True, True), ("LR", False, False),
        ])
    ]
    out = summarize_day(ticks)
    assert out["occupied_ticks"] == 4
    assert out["moving_ticks"] == 3
    assert out["active_fraction"] == 0.75
    assert out["room_transitions"] == 2      # LR→BR→LR
    assert out["coverage_ticks"] == 5
