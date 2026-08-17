"""Journal day builders — pure functions, no I/O.

Each detector contributes one journal line whose shape is a pure
function of its taxonomy (docs/plans/2026-08-16-daily-journal.md).
The endpoint (src/api/journal.py) does the queries and hands plain
rows here. Rules that live in this file:

  - Strips are SPARSE: only sleep, episodes, and event days carry one;
    a strip never aggregates or infers, it only places recorded
    timestamps on a 24-h line.
  - Sleep rides a NOON-TO-NOON axis so the night band is contiguous;
    everything else is midnight-to-midnight.
  - Activity (pinned definition): moving = standing + active minutes
    from the summarizer's state vocabulary; a block renders where a
    30-minute bucket holds >= 10 moving minutes; solid blocks only.
  - Episodes have no persisted end time yet, so they render as marks —
    never fabricated spans.
  - No summary sentence anywhere.
"""

from __future__ import annotations

from datetime import datetime, tzinfo
from typing import Any

# Analysis cadence: one tick ~ every 10 seconds.
_TICK_SECONDS = 10
_MOVING_STATES = {"standing", "active"}
_BLOCK_BUCKET_HOURS = 0.5
_BLOCK_MIN_MOVING_MINUTES = 10

_CATEGORY_ORDER = ["safety", "routine", "behaviour", "care"]


# ── shared helpers ───────────────────────────────────────────────────

def hour_x(dt: datetime, tz: tzinfo, axis: str = "midnight") -> float:
    """Fractional-hour position of a moment on the day-strip axis.
    axis="midnight": 0..24 from local midnight. axis="noon": 0..24 from
    local noon, so a night (10 PM -> 7 AM) is one contiguous span."""
    local = dt.astimezone(tz)
    hours = local.hour + local.minute / 60 + local.second / 3600
    if axis == "noon":
        return (hours - 12) % 24
    return hours


def _clock(dt: datetime, tz: tzinfo) -> str:
    local = dt.astimezone(tz)
    out = local.strftime("%I:%M %p").lstrip("0")
    return out


def _deadline_words(deadline: str | None) -> str:
    if not deadline:
        return "in its window"
    try:
        h, m = (int(p) for p in deadline.split(":")[:2])
    except ValueError:
        return "in its window"
    ampm = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"by {h12} {ampm}" if m == 0 else f"by {h12}:{m:02d} {ampm}"


def _review_state(events: list) -> tuple[str, Any]:
    """(status, review_event_id) from the day's events."""
    for ev in events:
        if not getattr(ev, "reviewed", True):
            return "review", ev.id
    return "none", None


def _entry(det, kind: str, primary: str, meta: str | None, status: str,
           strip: dict | None, review_event_id=None) -> dict:
    return {
        "detector_id": str(det.id) if det is not None else None,
        "title": det.name if det is not None else kind,
        "icon": getattr(det, "icon_name", None) if det is not None else None,
        "category": det.category if det is not None else "routine",
        "severity": getattr(det, "severity", None) if det is not None else None,
        "kind": kind,
        "line": {"primary": primary, "meta": meta, "status": status},
        "strip": strip,
        "review_event_id": review_event_id,
    }


def _strip(axis: str, spans: list, marks: list, flag: float | None = None) -> dict:
    return {"axis": axis, "spans": spans, "marks": marks, "flag": flag}


# ── point alerts (leading, rare) ─────────────────────────────────────

def point_line(det, events: list, tz: tzinfo, streak_days: int | None) -> dict:
    if not events:
        return _entry(
            det, "point",
            f"No {det.name.lower()}",
            (f"{streak_days} day{'' if streak_days == 1 else 's'}"
             if streak_days else None),
            "ok", None,
        )
    status, review_id = _review_state(events)
    first = events[0]
    n = len(events)
    if n == 1:
        meta = f"{_clock(first.occurred_at, tz)}, {first.location}"
    else:
        word = "Twice" if n == 2 else f"{n} times"
        meta = f"{word}, first {_clock(first.occurred_at, tz)}"
    marks = [hour_x(ev.occurred_at, tz) for ev in events]
    return _entry(det, "point", det.name, meta, status,
                  _strip("midnight", [], marks), review_id)


# ── sustained episodes (alerting) ────────────────────────────────────

def episode_line(det, events: list, tz: tzinfo) -> dict:
    """Episode ends aren't persisted yet, so episodes render as marks
    at their start time — never fabricated spans."""
    if not events:
        return _entry(det, "episode", f"No {det.name.lower()}", None, "ok", None)
    status, review_id = _review_state(events)
    n = len(events)
    meta = f"{_clock(events[0].occurred_at, tz)}, {events[0].location}"
    marks = [hour_x(ev.occurred_at, tz) for ev in events]
    return _entry(det, "episode",
                  f"{n} episode{'' if n == 1 else 's'}", meta, status,
                  _strip("midnight", [], marks), review_id)


# ── absence (deadline) — an event IS the miss ────────────────────────

def absence_line(det, events: list, tz: tzinfo) -> dict:
    deadline = _deadline_words((det.schedule_config or {}).get("deadline"))
    if not events:
        return _entry(det, "absence", f"Done {deadline}", None, "ok", None)
    status, review_id = _review_state(events)
    return _entry(det, "absence", f"Missed, not seen {deadline}",
                  _clock(events[0].occurred_at, tz),
                  "review" if status == "review" else "none",
                  None, review_id)


# ── record streams ───────────────────────────────────────────────────

def sleep_line(row: dict | None, tz: tzinfo) -> dict:
    det = None
    if row is None or not row.get("bedtime") or not row.get("wake_time"):
        e = _entry(det, "record_sleep", "No sleep recorded", None, "none", None)
        e["title"], e["category"], e["icon"] = "Sleep", "routine", "bed.double"
        return e
    bed, wake = row["bedtime"], row["wake_time"]
    n = int(row.get("disturbances") or 0)
    meta = "no wake-ups" if n == 0 else f"{n} wake-up{'' if n == 1 else 's'}"
    spans = _sleep_spans(bed, wake, row.get("out_of_bed_events"), tz)
    e = _entry(
        det, "record_sleep",
        f"{_clock(bed, tz)} – {_clock(wake, tz)}", meta, "none",
        _strip("noon", spans, []),
    )
    e["title"], e["category"], e["icon"] = "Sleep", "routine", "bed.double"
    return e


def _sleep_spans(bed, wake, out_of_bed, tz: tzinfo) -> list[list[float]]:
    """Split the night band at recorded out-of-bed intervals. The
    summarizer owns the interval shape; anything unparseable degrades
    to the unbroken band, never a fabricated gap."""
    b, w = hour_x(bed, tz, "noon"), hour_x(wake, tz, "noon")
    gaps: list[tuple[float, float]] = []
    for item in (out_of_bed or []):
        try:
            gs = datetime.fromisoformat(str(item["start"]))
            ge = datetime.fromisoformat(str(item.get("end") or item["start"]))
            g0, g1 = hour_x(gs, tz, "noon"), hour_x(ge, tz, "noon")
            if b < g0 < w:
                gaps.append((g0, max(g0 + 0.15, min(g1, w))))
        except Exception:
            continue
    spans, cursor = [], b
    for g0, g1 in sorted(gaps):
        if g0 > cursor:
            spans.append([cursor, g0])
        cursor = max(cursor, g1)
    if cursor < w:
        spans.append([cursor, w])
    return spans or [[b, w]]


def _moving_words(minutes: float) -> str:
    m15 = int(round(minutes / 15) * 15)
    if m15 <= 0:
        return "Under 15 m active"
    if m15 < 60:
        return f"About {m15} m active"
    h, rem = divmod(m15, 60)
    return f"About {h} h active" if rem == 0 \
        else f"About {h} h {rem} m active"


def activity_line(ticks: list, tz: tzinfo) -> dict:
    """ticks: [(captured_at, activity_state)] for the local day.
    Renders as a two-color tug-of-war: active minutes vs the rest of
    the observed, labeled time — a ratio, not a timeline, so there is
    no axis to misread."""
    moving = sum(1 for _, state in ticks if state in _MOVING_STATES)
    labeled = len(ticks)
    minutes = moving * _TICK_SECONDS / 60
    e = _entry(None, "record_activity", _moving_words(minutes), None, "none",
               None)
    e["title"], e["category"], e["icon"] = "Activity", "routine", "figure.walk"
    if labeled:
        e["ratio"] = {
            "active_minutes": round(minutes),
            "labeled_minutes": round(labeled * _TICK_SECONDS / 60),
        }
    return e


def count_line(det, times: list, tz: tzinfo) -> dict:
    if not times:
        return _entry(det, "record_count", "None", None, "none", None)
    n = len(times)
    return _entry(det, "record_count", f"{n} time{'' if n == 1 else 's'}",
                  None, "none",
                  _strip("midnight", [], [hour_x(t, tz) for t in times]))


# ── assembly ─────────────────────────────────────────────────────────

def assemble_sections(entries: list[dict]) -> list[dict]:
    """Fixed category order, severe-first within a section; categories
    with no entries do not render."""
    out = []
    for cat in _CATEGORY_ORDER:
        rows = [e for e in entries if e.get("category") == cat]
        if not rows:
            continue
        rows.sort(key=lambda e: -(e.get("severity") or 0))
        out.append({"category": cat, "entries": rows})
    return out


def _duration_words(minutes: float) -> str:
    m15 = int(round(minutes / 15) * 15)
    if m15 <= 0:
        return "Under 15 m"
    if m15 < 60:
        return f"About {m15} m"
    h, rem = divmod(m15, 60)
    return f"About {h} h" if rem == 0 else f"About {h} h {rem} m"


def duration_line(det, tick_times: list, tz: tzinfo) -> dict:
    """Sustained record streams (TV, time alone): contiguous detected
    ticks are RECORDED timestamps, so merging runs (gap <= 60 s) into
    session spans is a clean map, not an inference."""
    if not tick_times:
        return _entry(det, "record_duration", "None", None, "none", None)
    spans: list[list[float]] = []
    run_start = prev = tick_times[0]
    for t in tick_times[1:]:
        if (t - prev).total_seconds() > 60:
            spans.append([hour_x(run_start, tz), hour_x(prev, tz)])
            run_start = t
        prev = t
    spans.append([hour_x(run_start, tz), hour_x(prev, tz)])
    minutes = len(tick_times) * _TICK_SECONDS / 60
    return _entry(det, "record_duration", _duration_words(minutes), None,
                  "none", _strip("midnight", spans, []))


def orphan_group_line(events: list, tz: tzinfo) -> dict:
    """Events whose detector was later DELETED still happened, and the
    journal for THAT day must keep narrating them (Aryaman 2026-08-17:
    "if a detector existed in the past, its events should still be
    recorded for that day"). The event row stores its own title/icon/
    category precisely to outlive its detector. Grouped by stored
    title, one line per title per day; days after the deletion have no
    events and therefore show nothing."""
    first = events[0]
    n = len(events)
    status, review_id = _review_state(events)
    if n == 1:
        primary = f"At {_clock(first.occurred_at, tz)}"
        if first.location:
            primary += f", {first.location}"
    else:
        word = "Twice" if n == 2 else f"{n} times"
        primary = f"{word}, first at {_clock(first.occurred_at, tz)}"
    return {
        "detector_id": None,
        "title": first.title,
        "icon": first.icon_name,
        "category": first.category,
        "severity": None,
        "kind": "point",
        "line": {"primary": primary, "meta": "detector removed",
                 "status": status},
        "strip": None,
        "review_event_id": review_id,
    }


def build_entries(*, detectors: list, events_by_detector: dict, tz: tzinfo,
                  sleep_row: dict | None, activity_ticks: list,
                  detected_times_by_identifier: dict,
                  streak_days_by_detector: dict,
                  coverage_hours: int = 24,
                  orphan_events: list | None = None) -> list[dict]:
    """Route each detector to its taxonomy's builder. Pure — the
    endpoint does the queries, this does the shape.

    Honesty rules (phase review, Aug 16):
      - A detector that is OFF still contributes its line on days it
        HAS events — history never vanishes on a toggle. Off detectors
        with nothing that day are skipped (no quiet claims for them).
      - When the day had zero watched hours, quiet lines say
        "Not watched" instead of asserting a clean bill that was never
        observed.
    """
    entries: list[dict] = []
    for det in detectors:
        events = events_by_detector.get(str(det.id), [])
        if not getattr(det, "is_on", True):
            had_ticks = bool(detected_times_by_identifier.get(
                det.identifier or ""))
            if not events and not had_ticks:
                continue
        if det.alert_trigger == "record_only":
            if det.identifier == "sleep":
                e = sleep_line(sleep_row, tz)
                e["detector_id"] = str(det.id)
            elif det.identifier == "activity":
                e = activity_line(activity_ticks, tz)
                e["detector_id"] = str(det.id)
            else:
                times = detected_times_by_identifier.get(det.identifier or "", [])
                e = (duration_line(det, times, tz) if det.is_sustained
                     else count_line(det, times, tz))
        elif det.alert_trigger == "absence" or (
            (det.schedule_config or {}).get("deadline")
        ):
            # Legacy absence detectors (medication) predate the absence
            # trigger value: leading + a schedule deadline.
            e = absence_line(det, events, tz)
        elif det.is_sustained:
            e = episode_line(det, events, tz)
        else:
            e = point_line(det, events, tz,
                           streak_days_by_detector.get(str(det.id)))
        entries.append(e)
    # Orphaned events (detector deleted) — grouped by their stored
    # title, appended after the living detectors. See orphan_group_line.
    if orphan_events:
        by_title: dict[str, list] = {}
        for ev in orphan_events:
            by_title.setdefault(ev.title or "Event", []).append(ev)
        for evs in by_title.values():
            entries.append(orphan_group_line(evs, tz))
    if coverage_hours == 0:
        for e in entries:
            quiet = (e["line"]["status"] == "ok"
                     or (e["kind"].startswith("record") and e["strip"] is None))
            if quiet:
                e["line"] = {"primary": "Not watched", "meta": None,
                             "status": "none"}
                e["strip"] = None
    return entries
