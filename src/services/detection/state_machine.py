"""Per-camera open/closed state machine.

The state controls when the LLM runs:

  - `open`:    LLM runs every tick (10s).
  - `closed`:  LLM is **completely off**. Motion is the wake-up.

Transitions:
  - `closed → open`: any tick with `people_present=true` (LLM verdict)
                     OR any tick with `motion_in_clip=true` (wall-phone
                     wake-up signal). Motion-driven reactivation means
                     the LLM resumes on this same tick — the caller
                     transitions state *before* the LLM call, so the
                     tick that detected motion runs as an open-state
                     tick.
  - `open → closed`: `close_threshold` consecutive ticks with
                     `people_present=false` (and the LLM ran on each).

Why motion is now allowed to reopen: under the previous
"people-only" gate the state could only flip closed → open when the
LLM said `people_present=true`. But the LLM never ran in closed
state with no motion — and closed-with-motion still cost a full
LLM call. With motion as the wake-up, closed truly means *no LLM
calls at all* until something in the room moves. The trade-off is
that a noisy on-phone motion detector (low threshold, pets,
curtains, lighting flicker) can keep flipping the camera back
open. That knob lives in `NemaraCam/Camera/MotionDetector.swift`
(`motionThreshold`); raise it if false reopens become common.

Persistent state lives on the `cameras` row (state, no_people_count,
state_changed_at).

## Concurrency note (matters since tick loop went fire-and-forget)

This module used to be a read-modify-write Python loop:

    cam = session.get(Camera, ...)   # snapshot read
    cam.no_people_count = current + 1
    session.flush()                   # write

That was safe when the tick loop serialized ticks behind a per-camera
asyncio.Lock. After tick scheduling moved to fire-and-forget
(`camera_ingest._tick_loop`), two ticks for the same camera can be
in flight simultaneously — each in its own DB session — and the RMW
loses increments: both read N, both write N+1, instead of one writing
N+2. Symptom was the camera taking ~10 ticks instead of 6 to close
because roughly 40% of increments were getting dropped.

The fix is to do the whole state transition in a **single atomic
UPDATE statement**. PostgreSQL's row-level lock serializes concurrent
UPDATEs on the same row, and the CASE expressions inside the UPDATE
see the latest committed value at execution time — no lost increments,
no torn writes.

We bypass the ORM for this one write (no Camera object change-tracking
needed because callers don't reuse the in-memory Camera after this).
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class CameraStateAfter:
    """The fields callers care about after a tick — used to populate
    `analyses.state_after` and the route response.
    """
    state: str               # 'open' | 'closed'
    no_people_count: int
    close_threshold: int


# Single atomic UPDATE encoding the entire state machine. The CASE
# expressions reference `cameras.no_people_count` etc. — PostgreSQL
# evaluates them against the LATEST COMMITTED row value at UPDATE
# time, and acquires a row lock for the duration of the statement.
# This makes the increment race-free even when two backend tasks
# call `apply_tick_update` for the same camera concurrently.
#
# Logic (motion-as-wake-up + people-confirmation):
#   people_present (LLM saw a person this tick):
#     count → 0, state → 'open', changed_at → now if transition
#   else if state == 'closed' AND motion_in_clip (wall-phone wake-up):
#     count → 0, state → 'open', changed_at → now (transition)
#   else if state == 'closed' (closed + no motion — should never be
#     reached via apply_tick_update since the ingest worker writes
#     the synthetic skipped row directly without calling this; kept
#     defensively to clamp the counter):
#     count → close_threshold
#   else (state == 'open', no people, motion either way):
#     count → LEAST(count + 1, close_threshold)
#     state → 'closed' if count + 1 >= close_threshold else 'open'
#     changed_at → now if transition
_ATOMIC_UPDATE_SQL = text(
    """
    UPDATE public.cameras
    SET
        no_people_count = CASE
            WHEN :people_present THEN 0
            WHEN state = 'closed' AND :motion_in_clip THEN 0
            WHEN state = 'closed' THEN close_threshold
            ELSE LEAST(no_people_count + 1, close_threshold)
        END,
        state = CASE
            WHEN :people_present THEN 'open'
            WHEN state = 'closed' AND :motion_in_clip THEN 'open'
            WHEN state = 'open' AND no_people_count + 1 >= close_threshold
                THEN 'closed'
            ELSE state
        END,
        state_changed_at = CASE
            WHEN :people_present AND state = 'closed' THEN :now
            WHEN state = 'closed' AND :motion_in_clip THEN :now
            WHEN NOT :people_present AND state = 'open'
                AND no_people_count + 1 >= close_threshold THEN :now
            ELSE state_changed_at
        END,
        last_seen_at = :now
    WHERE id = :camera_id
    RETURNING state, no_people_count, close_threshold
    """
)


async def apply_tick_update(
    session: AsyncSession,
    *,
    camera_id: UUID,
    people_present: bool,
    motion_in_clip: bool,
) -> CameraStateAfter:
    """Atomic state machine update for one tick. Caller owns the
    transaction (we don't commit here).

    Both `people_present` (LLM verdict) and `motion_in_clip` (wall-phone
    motion-detector signal) drive transitions — see the module
    docstring for the rules. Motion is the cheap wake-up trigger that
    lets `closed → open` flip without needing the LLM to first say
    "person here"; people_present is the authoritative gate that
    keeps the camera open in the absence of motion noise.

    Race-free under concurrent ticks — see module docstring.

    Raises ValueError if the camera row doesn't exist (signals a bug;
    the api-key auth path should have already resolved it).
    """
    now = datetime.now(timezone.utc)

    result = await session.execute(
        _ATOMIC_UPDATE_SQL,
        {
            "camera_id": camera_id,
            "people_present": people_present,
            "motion_in_clip": motion_in_clip,
            "now": now,
        },
    )
    row = result.one_or_none()
    if row is None:
        raise ValueError(
            f"[state_machine] camera {camera_id} does not exist; "
            f"api-key auth path should have rejected the request"
        )

    # The raw SQL UPDATE bypasses SQLAlchemy's identity map. Any
    # previously-fetched Camera ORM object in this session now holds
    # stale state/count/changed_at/last_seen_at values. Expire the
    # identity map so subsequent reads in the same session (callers,
    # tests, or downstream queries that join through Camera) see the
    # committed values from our UPDATE.
    session.expire_all()

    state, no_people_count, close_threshold = row
    return CameraStateAfter(
        state=state,
        no_people_count=no_people_count,
        close_threshold=close_threshold,
    )
