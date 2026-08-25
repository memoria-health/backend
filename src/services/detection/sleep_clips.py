"""Save the tick a sleep disturbance was marked on, and the one after.

Sleep is `alert_trigger="record_only"`: its fields feed the summarizers
and it fires NO event, so `clip_archival` — which hangs off a fired
event — never runs for it. A night of sleep data therefore leaves
nothing to look at, and no way to tell a real disturbance from a
hallucinated one.

This is that missing evidence, and nothing more. It does not promote
sleep to an event, page anyone, or touch attribution — all of which
`record_only` exists to prevent. It writes the mp4 the tick was already
judged on to disk, next to the fields that judged it.

TWO ticks per disturbance, not one. The mark lands on the tick where the
movement was VISIBLE, so the tick after it is what says whether the
person settled or stayed up — which is the whole question when reading
a night back. The next tick is captured whatever it contains, including
`none`, because "it stopped" is the answer half the time.

DEV INSTRUMENT. Gated on `sleep_clips_dir` being set, off by default,
and it writes raw frames to local disk with no sensitivity screening —
so it belongs on a local stack recording its own operator, never on a
household's.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

log = logging.getLogger(__name__)

# `sleep_activity` values that mean the person moved. `none` is the
# steady state and is not a disturbance; `entering_bed` / `leaving_bed`
# are the two crossings that carry the night, and are the most valuable
# to see, so they count too.
DISTURBANCE_ACTIVITIES = frozenset({
    "entering_bed", "leaving_bed", "stirring", "sitting_up",
})

# Cameras armed to also save their NEXT tick, by camera id.
_follow_up: set[str] = set()


def _clip_name(activity: str, *, follow_up: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = "after" if follow_up else activity
    return f"{stamp}_{suffix}.mp4"


async def save_if_disturbance(
    *,
    camera_id: UUID,
    detector_fields: Mapping[str, Any],
    mp4_bytes: bytes | None,
    settings: Any,
) -> str | None:
    """Write this tick's mp4 if it marks a disturbance, or if the
    previous tick did. Returns the path written, or None.

    Never raises: a missing clip must not cost the tick that produced
    it. The whole point is observing the night, not shaping it.
    """
    directory = getattr(settings, "sleep_clips_dir", "") or ""
    if not directory or not mp4_bytes:
        return None

    cam_key = str(camera_id)
    activity = str(detector_fields.get("sleep_activity") or "none")
    is_disturbance = activity in DISTURBANCE_ACTIVITIES
    owed_follow_up = cam_key in _follow_up

    if not is_disturbance and not owed_follow_up:
        return None

    # Arm (or re-arm) the next tick when THIS one is a disturbance. A
    # run of movement therefore keeps saving rather than capturing the
    # first frame of a long episode and going quiet.
    _follow_up.discard(cam_key)
    if is_disturbance:
        _follow_up.add(cam_key)

    try:
        cam_dir = Path(directory) / cam_key[:8]
        cam_dir.mkdir(parents=True, exist_ok=True)
        path = cam_dir / _clip_name(activity, follow_up=not is_disturbance)
        path.write_bytes(mp4_bytes)
        # The fields beside the footage: reading a clip back is only
        # useful next to what the model claimed while watching it.
        (path.with_suffix(".txt")).write_text(
            "\n".join(
                f"{k}={detector_fields.get(k)}"
                for k in ("sleep_detected", "sleep_activity",
                          "sleep_confidence", "people_present",
                          "people_present_src", "identity_patient_in_clip")
            ),
            encoding="utf-8",
        )
        log.info(
            "sleep_clips.saved camera=%s activity=%s follow_up=%s -> %s",
            cam_key[:8], activity, not is_disturbance, path.name,
        )
        return str(path)
    except Exception:  # noqa: BLE001 — evidence must never break a tick
        log.exception("sleep_clips.save_failed camera=%s", cam_key[:8])
        return None
