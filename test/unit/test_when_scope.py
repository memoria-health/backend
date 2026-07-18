"""Unit: when_scope parsing + tick eligibility (recalibration 2026-07-14).

Pins the fail-open law (unparseable/empty/always ⇒ eligible, not
windowed), anchor±slack windows, explicit ranges, shared-meridiem
ranges, minute slacks, and midnight-crossing windows.
"""

from __future__ import annotations

from datetime import datetime

from src.services.detection.scheduling.when_scope import is_active, parse_windows


def _at(hh: int, mm: int = 0) -> datetime:
    return datetime(2026, 7, 14, hh, mm, 0)


def test_always_flavors_parse_to_none_and_never_window() -> None:
    for text in (None, "", "ALWAYS", "ALL DAY", "all-day", "24/7", "Any time"):
        assert parse_windows(text) is None
        assert is_active(text, _at(3)) == (True, False)


def test_unparseable_text_fails_open() -> None:
    assert parse_windows("whenever grandma is up") is None
    assert is_active("whenever grandma is up", _at(4)) == (True, False)


def test_builtin_anchor_pair_with_explicit_slack() -> None:
    scope = "8 AM, 8 PM ±1H"
    assert is_active(scope, _at(8, 30)) == (True, True)     # inside AM
    assert is_active(scope, _at(19, 15)) == (True, True)    # inside PM
    assert is_active(scope, _at(17, 0)) == (False, True)    # 5 PM — outside
    assert is_active(scope, _at(23, 0)) == (False, True)


def test_anchor_without_slack_defaults_to_one_hour() -> None:
    scope = "5:45 PM"
    assert is_active(scope, _at(17, 0)) == (True, True)      # 16:45–18:45
    assert is_active(scope, _at(15, 0)) == (False, True)


def test_minute_slack() -> None:
    scope = "9 AM ±30 min"
    assert is_active(scope, _at(9, 20)) == (True, True)
    assert is_active(scope, _at(10, 0)) == (False, True)


def test_explicit_range_and_shared_meridiem_range() -> None:
    assert is_active("5 PM - 7 PM", _at(18, 0)) == (True, True)
    assert is_active("5 PM - 7 PM", _at(16, 0)) == (False, True)
    assert is_active("5-7 PM", _at(18, 30)) == (True, True)
    assert is_active("5-7 PM", _at(19, 30)) == (False, True)


def test_window_crossing_midnight() -> None:
    scope = "11:30 PM ±1H"                                   # 22:30–00:30
    assert is_active(scope, _at(0, 15)) == (True, True)
    assert is_active(scope, _at(23, 0)) == (True, True)
    assert is_active(scope, _at(2, 0)) == (False, True)


# ─── schedule_config dissolution (2026-07-18) ────────────────────


def test_parse_days_tokens_and_absence() -> None:
    from src.services.detection.scheduling.when_scope import parse_days
    assert parse_days("MON,WED,FRI 8:00 AM - 9:00 AM") == (0, 2, 4)
    assert parse_days("tuesdays 5 PM - 7 PM") == (1,)
    assert parse_days("8:00 AM - 9:00 AM") is None
    assert parse_days(None) is None


def test_parse_schedule_window_and_deadline_is_end() -> None:
    from src.services.detection.scheduling.when_scope import parse_schedule
    sw = parse_schedule("MON,WED 8:00 AM - 9:30 AM")
    assert sw is not None
    assert (sw.start_min, sw.end_min, sw.days) == (480.0, 570.0, (0, 2))
    daily = parse_schedule("8:00 AM - 9:00 AM")
    assert daily is not None and daily.days is None
    assert parse_schedule("ALWAYS") is None
    assert parse_schedule("9 PM - 2 AM") is None  # midnight-cross rejected


def test_is_active_gates_on_weekday() -> None:
    from datetime import datetime

    from src.services.detection.scheduling.when_scope import is_active
    scope = "MON 8:00 AM - 9:00 AM"
    monday_in = datetime(2026, 7, 20, 8, 30)     # a Monday
    tuesday_in = datetime(2026, 7, 21, 8, 30)    # a Tuesday
    assert is_active(scope, monday_in) == (True, True)
    assert is_active(scope, tuesday_in) == (False, True)
