"""Daily activity summarizer — pure telemetry, zero VLM cost.

Folds one day's analyzed ticks (columns the pipeline already records:
people_present, motion_in_clip, camera) into a daily activity record:
per-camera occupied/moving tick counts, an active fraction, and room
transitions (consecutive presence ticks on different cameras).

Deliberately trend-oriented: the number that matters downstream is
today's active fraction vs the patient's own trailing baseline (Whoop
principle — personal baselines, not universal thresholds). The baseline
comparison lives with the reader; this module just produces honest
daily numbers plus coverage so low-observation days are never read as
low-activity days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ActivityTick:
    at: datetime
    camera: str                  # camera id prefix or name
    people_present: bool
    motion: bool                 # motion_in_clip


def summarize_day(ticks: list[ActivityTick]) -> dict:
    ts = sorted(ticks, key=lambda t: t.at)
    per_cam: dict[str, dict] = {}
    transitions = 0
    last_cam: str | None = None
    presence = moving = 0
    for t in ts:
        c = per_cam.setdefault(t.camera, {"occupied": 0, "moving": 0})
        if not t.people_present:
            continue
        c["occupied"] += 1
        presence += 1
        if t.motion:
            c["moving"] += 1
            moving += 1
        if last_cam is not None and t.camera != last_cam:
            transitions += 1
        last_cam = t.camera
    return {
        "per_camera": per_cam,
        "occupied_ticks": presence,
        "moving_ticks": moving,
        "active_fraction": round(moving / presence, 3) if presence else None,
        "room_transitions": transitions,
        "coverage_ticks": len(ts),   # low coverage != low activity
    }
