"""Nightly sleep summarizer — Aryaman's state machine (2026-07-28).

Ticks are motion-gated, so a still sleeper generates NO ticks: the
labeled burst ticks (from the "sleep" detector's prompt fields) mark the
moments, and the SILENCE between them is the asleep signal. This module
is a pure function over one night's labeled ticks — no I/O, no clock
reads — so the whole state machine is unit-testable and the runner
(tools/run_daily_summaries.py) stays a thin query-and-insert shell.

State machine:
  bedtime     = first confident in-bed tick in the night window
  sleep onset = first in-bed tick followed by >= ONSET_QUIET of tick
                silence (nothing moved -> nothing fired -> asleep)
  disturbance = stirring / sitting_up tick while the session is active
  out-of-bed  = leaving_bed tick with a RETURN (a later in-bed tick
                within OUT_RETURN_MAX); duration recorded, session
                continues
  wake        = leaving_bed / sitting_up tick after which no in-bed
                tick occurs for WAKE_CONFIRM (or the window ends)
  duration    = wake - onset - sum(out-of-bed durations)

Honesty rules: a window that ends with the person still in bed yields
wake=None + an "incomplete_night" flag (the runner re-runs later); zero
qualifying ticks yields a "no_bed_entry" flag, never a fabricated night.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

ONSET_QUIET = timedelta(minutes=15)
WAKE_CONFIRM = timedelta(minutes=30)
MIN_CONFIDENCE = 0.6

_DISTURB = ("stirring", "sitting_up")


@dataclass(frozen=True)
class SleepTick:
    """One labeled burst tick (from analyses detector_fields)."""

    at: datetime
    in_bed: bool                 # sleep_detected
    activity: str                # settling|stirring|sitting_up|leaving_bed|none
    confidence: float            # sleep_confidence


def summarize_night(
    ticks: list[SleepTick],
    *,
    window_end: datetime,
) -> dict:
    """Fold one night's ticks (chronological) into the nightly record.

    `window_end` is the end of the observation window the runner queried
    (needed to score trailing silence — we can only call "asleep at
    onset" or "still in bed" relative to how far observation extends).
    """
    ts = sorted(
        (t for t in ticks if t.confidence >= MIN_CONFIDENCE),
        key=lambda t: t.at,
    )
    out: dict = {
        "bedtime": None, "sleep_onset": None, "wake_time": None,
        "duration_minutes": None, "disturbances": 0,
        "out_of_bed_events": [], "flags": [],
    }
    in_bed_ticks = [t for t in ts if t.in_bed]
    if not in_bed_ticks:
        out["flags"].append("no_bed_entry")
        return out
    out["bedtime"] = in_bed_ticks[0].at

    def _next_at(i: int) -> datetime:
        """Time of the tick after index i, or the window end."""
        return ts[i + 1].at if i + 1 < len(ts) else window_end

    # ── onset: first in-bed tick followed by ONSET_QUIET of silence ──
    onset_i: int | None = None
    for i, t in enumerate(ts):
        if t.in_bed and _next_at(i) - t.at >= ONSET_QUIET:
            onset_i = i
            out["sleep_onset"] = t.at
            break
    if onset_i is None:
        out["flags"].append("no_sleep_onset")
        return out

    # ── walk the session ─────────────────────────────────────────────
    wake: datetime | None = None
    i = onset_i + 1
    while i < len(ts):
        t = ts[i]
        if t.in_bed and t.activity in _DISTURB:
            out["disturbances"] += 1
            i += 1
            continue
        if t.activity == "leaving_bed" or (not t.in_bed):
            # Return-scan (reopen fix, Aryaman 2026-07-28): ANY later
            # in-bed tick makes this an out-of-bed event — including a
            # 3am wander that lasts hours, clinically the headline
            # event. Wake is ONLY a leave with no return before the
            # window ends. (v1 capped returns at 60min and truncated
            # wander nights with a false wake.)
            ret = next(
                (j for j in range(i + 1, len(ts)) if ts[j].in_bed),
                None,
            )
            if ret is not None:
                out["out_of_bed_events"].append({
                    "at": t.at.isoformat(),
                    "duration_minutes": round(
                        (ts[ret].at - t.at).total_seconds() / 60),
                })
                i = ret + 1
                continue
            # No return: this is the wake IF observation extends far
            # enough past it to be sure (WAKE_CONFIRM), else the night
            # is still in progress.
            if window_end - t.at >= WAKE_CONFIRM:
                wake = t.at
            break
        i += 1                     # settling / none — neutral

    if wake is None:
        out["flags"].append("incomplete_night")
        return out
    out["wake_time"] = wake
    away = sum(e["duration_minutes"] for e in out["out_of_bed_events"])
    out["duration_minutes"] = max(
        0, round((wake - out["sleep_onset"]).total_seconds() / 60) - away)
    return out
