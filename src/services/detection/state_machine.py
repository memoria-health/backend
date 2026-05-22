"""Per-camera open/closed state machine.

Verbatim port of `nemara-temp-dashboard/lib/camera-state.ts:applyTickUpdate`.
The state controls when the LLM runs:

  - `open`:    LLM runs every tick (10s).
  - `closed`:  LLM is skipped until people OR motion reappear.

Transitions:
  - `closed → open`: any tick with `people_present=true` OR `motion_in_clip=true`.
  - `open → closed`: `close_threshold` consecutive ticks with both false.

Persistent state lives on the `cameras` row (state, no_people_count,
state_changed_at) so Phase 2's restart-resilient worker doesn't lose
the counter when it crashes mid-shift.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera


@dataclass(frozen=True)
class CameraStateAfter:
    """The fields callers care about after a tick — used to populate
    `analyses.state_after` and the route response. Mirrors the
    dashboard's `CameraState` shape for the response payload.
    """
    state: str               # 'open' | 'closed'
    no_people_count: int
    close_threshold: int


async def apply_tick_update(
    session: AsyncSession,
    *,
    camera_id: UUID,
    people_present: bool,
    motion_in_clip: bool,
) -> CameraStateAfter:
    """Read-modify-write of the camera state for one tick.

    Caller owns the transaction (we don't commit here). At nemara's
    scale (one camera publishes sequentially per phone), the
    read-modify-write race window is non-existent — there's only one
    writer per row at a time.

    Raises ValueError if the camera row doesn't exist (signals a bug;
    the api-key auth path should have already resolved it).
    """
    cam = await session.get(Camera, camera_id)
    if cam is None:
        raise ValueError(
            f"[state_machine] camera {camera_id} does not exist; "
            f"api-key auth path should have rejected the request"
        )

    now = datetime.now(timezone.utc)
    close_threshold = cam.close_threshold
    current_state = cam.state
    current_count = cam.no_people_count

    next_state: str = current_state
    next_count: int = current_count

    if people_present or motion_in_clip:
        next_count = 0
        if current_state == "closed":
            next_state = "open"
            cam.state_changed_at = now
    else:
        if current_state == "closed":
            # Already closed: clamp the counter at the threshold so it
            # doesn't grow unboundedly (matches dashboard M2).
            next_count = close_threshold
        else:
            next_count = current_count + 1
            if next_count >= close_threshold:
                next_state = "closed"
                cam.state_changed_at = now

    cam.state = next_state
    cam.no_people_count = next_count
    cam.last_seen_at = now
    await session.flush()

    return CameraStateAfter(
        state=next_state,
        no_people_count=next_count,
        close_threshold=close_threshold,
    )
