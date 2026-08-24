"""Pick the camera that should receive caregiver Talk audio.

Used by `POST /v1/patients/{id}/talk/start` to figure out *which* wall
phone in a patient's household should play the caregiver's voice. The
heuristic is simple: the camera whose most-recent `analyses` row has
`people_present=true` within the last 30 seconds is the "active" one.
If no qualifying row exists (newly-paired household, everyone asleep,
gap between detector ticks), the caller is expected to fan out the
audio to every camera in the room.

Why 30 seconds: the detector pipeline ticks every ~10s per camera, so
two missed ticks is the longest "still present but quiet" gap we'd
expect for a person in-frame. Beyond that, falling back to fan-out is
the safe choice — Talk is a barge-in interaction, the caregiver
wants to be heard *somewhere* immediately.

The query touches `analyses(patient_id, captured_at DESC)` which is
covered by `analyses_camera_created_idx` (camera_id) plus a sequential
scan on the recent slice — fine for v0 scale; if it becomes a
hotspot, add `analyses(patient_id, captured_at DESC)` explicitly.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.analysis import Analysis
from src.services.camera_coverage import covered_cameras

# How recent an "active" analyses row must be to count. Two detector
# ticks (~20s) plus a buffer for clock skew + LLM latency.
_ACTIVE_WINDOW = timedelta(seconds=30)


async def get_active_camera(
    session: AsyncSession,
    *,
    patient_id: UUID,
    now: datetime | None = None,
) -> UUID | None:
    """Return the camera_id with the freshest `people_present=true`
    analyses row in the last 30 seconds, or None if no qualifying row.

    `now` is parameterized so tests can pin the clock.
    """
    cutoff = (now or datetime.now(timezone.utc)) - _ACTIVE_WINDOW
    stmt = (
        select(Analysis.camera_id)
        .where(
            Analysis.camera_id.in_(covered_cameras(patient_id)),
            Analysis.people_present.is_(True),
            Analysis.captured_at >= cutoff,
        )
        .order_by(Analysis.captured_at.desc())
        .limit(1)
    )
    return (await session.execute(stmt)).scalar_one_or_none()
