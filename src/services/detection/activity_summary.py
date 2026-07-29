"""Daily activity summarizer v2 — VLM state timeline (Aryaman 2026-07-28).

v1 counted raw motion ticks, which mislabels hand-fidgeting-while-seated
as activity. v2 consumes the Activity detector's per-tick STATE labels
(lying / sitting / standing / active — seated exercise IS active) and
applies the same persistence principle as sleep: a label holds until the
next label, because a state change is physically a motion event and
motion fires a tick. The daily record is therefore TIME-WEIGHTED state
minutes, not tick counts — strictly more accurate than any motion proxy.

Coverage honesty: time before the first label of the day is UNOBSERVED
(not attributed to any state), and `labeled_minutes` rides the record so
a low-observation day can never masquerade as a low-activity day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

MIN_CONFIDENCE = 0.6
STATES = ("lying", "sitting", "standing", "active")


@dataclass(frozen=True)
class StateTick:
    """One labeled tick from the Activity detector."""

    at: datetime
    camera: str
    state: str                   # activity_state
    confidence: float            # activity_confidence


def summarize_day(
    ticks: list[StateTick],
    *,
    window_end: datetime,
) -> dict:
    """Fold one day's state labels into time-weighted state minutes.

    Each label persists until the next label (or `window_end` for the
    last one — bounded by how far observation actually extends).
    """
    ts = sorted(
        (t for t in ticks
         if t.confidence >= MIN_CONFIDENCE and t.state in STATES),
        key=lambda t: t.at,
    )
    minutes = {s: 0.0 for s in STATES}
    transitions = 0
    last_cam: str | None = None
    for i, t in enumerate(ts):
        until = ts[i + 1].at if i + 1 < len(ts) else window_end
        minutes[t.state] += max(0.0, (until - t.at).total_seconds() / 60)
        if last_cam is not None and t.camera != last_cam:
            transitions += 1
        last_cam = t.camera
    labeled = round(sum(minutes.values()), 1)
    return {
        "state_minutes": {s: round(m, 1) for s, m in minutes.items()},
        "active_fraction": (
            round(minutes["active"] / labeled, 3) if labeled else None
        ),
        "room_transitions": transitions,
        "labeled_minutes": labeled,      # observation coverage, not a score
        "label_ticks": len(ts),
    }
