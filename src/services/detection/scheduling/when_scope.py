"""when_scope → tick eligibility (recalibration, Aryaman 2026-07-14).

`detectors.when_scope` is operator-facing text ("ALL DAY",
"8 AM, 8 PM ±1H", "5 PM - 7 PM"). Until now it rode only into the
prompt; this module turns it into a scheduling gate: outside the
window the detector is skipped for the tick (skip reason
`outside_when_scope`), inside it the due entry is marked `windowed`
so the packer gives windowed detectors their own call — they never
share a call with always-on detectors (Aryaman: "when scope detectors
should be their own sub group that have a distinct call").

Fail-open is the law here: anything empty, unparseable, or
"always"-flavored means ALWAYS ELIGIBLE (and windowed=False). A typo
in a free-text field must never silently disable a safety detector.

Scheduled-shape rows are exempt by design — their per-tick fields feed
the absence runner all day; their own schedule_config governs timing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

_ALWAYS = frozenset({
    "", "always", "all day", "all-day", "allday", "24/7", "24x7",
    "anytime", "any time", "all rooms",
})

# Weekday prefix grammar (2026-07-18, schedule_config dissolution):
# "MON,WED,FRI 8:00 AM - 9:00 AM" — day tokens may appear anywhere;
# no day tokens = every day. 0=Mon..6=Sun (Python weekday()).
_DAY_MAP = {
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
    "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}
_DAY_TOKEN = re.compile(
    r"\b(mon|tues?|wed|thur?s?|fri|sat|sun)(?:day)?s?\b", re.IGNORECASE,
)

_TIME = re.compile(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", re.IGNORECASE)
_SLACK = re.compile(
    r"±\s*(\d+(?:\.\d+)?)\s*(h|hr|hrs|hour|hours|m|min|mins|minutes)",
    re.IGNORECASE,
)
_RANGE = re.compile(
    r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*[-–]\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)",
    re.IGNORECASE,
)

# An anchor time with no explicit ± gets this window either side —
# matches the builtin convention ("8 AM, 8 PM ±1H").
_DEFAULT_SLACK_MIN = 60.0


def _minutes(hh: str, mm: str | None, ap: str) -> float:
    h = int(hh) % 12
    if ap.lower() == "pm":
        h += 12
    return h * 60.0 + (int(mm) if mm else 0)


def parse_windows(text: str | None) -> tuple[tuple[float, float], ...] | None:
    """Minute-of-day windows for a when_scope, or None for always-eligible.

    Ranges ("5 PM - 7 PM", "5-7 PM") become [start, end]; anchor times
    ("8 AM, 8 PM ±1H") become [anchor-slack, anchor+slack]. Windows may
    cross midnight (start > end). None (not empty tuple) is the
    always-eligible answer so callers can distinguish "no window" from
    "window I couldn't read" — both fail open, only the former is
    windowed=False."""
    if text is None:
        return None
    if text.strip().lower() in _ALWAYS:
        return None
    windows: list[tuple[float, float]] = []
    remaining = text
    for m in _RANGE.finditer(text):
        ap_start = m.group(3) or m.group(6)
        start = _minutes(m.group(1), m.group(2), ap_start)
        end = _minutes(m.group(4), m.group(5), m.group(6))
        windows.append((start, end))
    if windows:
        remaining = _RANGE.sub(" ", text)
    slack = _DEFAULT_SLACK_MIN
    sm = _SLACK.search(text)
    if sm:
        n = float(sm.group(1))
        slack = n * 60.0 if sm.group(2).lower().startswith("h") else n
    for m in _TIME.finditer(remaining):
        anchor = _minutes(m.group(1), m.group(2), m.group(3))
        windows.append(((anchor - slack) % 1440.0, (anchor + slack) % 1440.0))
    return tuple(windows) or None


def parse_days(text: str | None) -> tuple[int, ...] | None:
    """Weekdays named in the scope (0=Mon..6=Sun), or None for every
    day. Same fail-open law: no recognizable day tokens = no day gate."""
    if not text:
        return None
    days = sorted({
        _DAY_MAP[m.group(1).lower()] for m in _DAY_TOKEN.finditer(text)
    })
    return tuple(days) or None


@dataclass(frozen=True)
class ScheduleWindow:
    """One evaluable window for the absence runner (schedule_config
    dissolution, 2026-07-18): start/end as minute-of-day, deadline ≡
    window end (Aryaman's fire-rule model: window(hit|miss) evaluates
    at close), days=None means daily."""

    start_min: float
    end_min: float
    days: tuple[int, ...] | None


def parse_schedule(text: str | None) -> ScheduleWindow | None:
    """The absence runner's view of a when_scope. None when the text is
    always-flavored, unparseable, or midnight-crossing (the runner's
    v1 parity: inverted windows are rejected, never guessed)."""
    windows = parse_windows(text)
    if not windows:
        return None
    start, end = windows[0]
    if start >= end:
        return None
    return ScheduleWindow(start_min=start, end_min=end, days=parse_days(text))


def is_active(text: str | None, local_now: datetime) -> tuple[bool, bool]:
    """(eligible, windowed) for a when_scope at caregiver-local time.

    Always-eligible scopes → (True, False). Real windows → (inside,
    True). `windowed=True` is what drives the packer's distinct-call
    grouping, so it is returned even when currently outside. Weekday
    tokens gate eligibility the same way hours do."""
    windows = parse_windows(text)
    if windows is None:
        return True, False
    days = parse_days(text)
    if days is not None and local_now.weekday() not in days:
        return False, True
    now = local_now.hour * 60.0 + local_now.minute + local_now.second / 60.0
    for start, end in windows:
        if start <= end:
            if start <= now <= end:
                return True, True
        elif now >= start or now <= end:  # window crosses midnight
            return True, True
    return False, True
