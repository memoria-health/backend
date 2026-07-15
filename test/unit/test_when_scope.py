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
