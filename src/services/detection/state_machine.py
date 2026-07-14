"""Per-camera open/closed state machine.

The state controls when the LLM runs:

  - `open`:    LLM runs every tick (10s).
  - `closed`:  LLM is off. Motion in this state triggers ONE
               verification LLM call; the LLM's verdict decides
               whether the camera actually transitions to open.

Transitions (ALL driven by `people_present` — the LLM verdict):

  - `closed → open`: any tick where the LLM returned
                     `people_present=true`. Motion alone never flips
                     the state. The LLM call that returned this
                     verdict was triggered by motion (the gate in
                     `camera_ingest.py:219` allows closed+motion to
                     run the LLM), but the people verdict — not the
                     motion signal — is what writes the transition.

  - `open → closed`: `close_threshold` consecutive ticks with
                     `people_present=false` while in the `open`
                     state. The counter increments on each no-people
                     tick, clamped at `close_threshold`. Any single
                     tick with `people_present=true` resets it to 0.

The "first LLM call verification" semantic falls out naturally:

  closed + motion → camera_ingest runs the LLM (one call)
    → LLM says people=true  → state flips to open, count=0,
                                normal open-state ticking begins
    → LLM says people=false → state stays closed, count clamped at
                                close_threshold. No false 7-call
                                open window for a phantom wake-up.

The motion signal lives entirely in `camera_ingest.py` (the
closed-state "should we even call the LLM this tick?" gate). Once
the LLM has been called, motion is irrelevant to the state
transition. This module accepts `motion_in_clip` purely for
logging in the analyses row; the SQL ignores it. Since v2 P2 that
value is the UNION of the phone/gateway SAD packet (now a hint)
and the backend's own frame-diff reading (tick_signals) — so
gateway cameras without SAD packets participate too.

If false motion wake-ups become too frequent, the backend knob is
`signals_motion_min_active_fraction` (settings.py); the on-phone
SAD knob remains `motionThreshold` in
`NemaraCam/Camera/MotionDetector.swift`. Each false wake costs one
LLM call (the verification) — not the seven it cost under the
previous "motion-auto-opens" rule.

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
# Logic (motion = LLM wake-up; people verdict = state transition):
#   people_present=true (LLM saw a person this tick):
#     count → 0
#     state → 'open' (transition from closed if applicable)
#     changed_at → now if a closed→open transition happened
#   else if state == 'closed' (this is the "motion-wake-up verification
#     failed" case: the camera was closed, motion fired, the ingest
#     worker ran the LLM, and the LLM came back with no people. We
#     stay closed — no false open window).
#     count → close_threshold (clamp)
#     state → unchanged ('closed')
#     changed_at → unchanged
#   else (state == 'open', no people):
#     count → LEAST(count + 1, close_threshold)
#     state → 'closed' if count + 1 >= close_threshold else 'open'
#     changed_at → now if a transition happened
#
# Motion is NOT a state input here. The on-phone motion signal only
# gates whether `camera_ingest._tick_loop` decides to call the LLM
# this tick (see camera_ingest.py:219). Once the LLM has been called,
# its `people_present` verdict is the sole driver of state. This means
# a closed camera + false-positive motion → 1 verification LLM call,
# people=false → stays closed (no 7-call open window for a phantom
# wake-up).
_ATOMIC_UPDATE_SQL = text(
    """
    UPDATE public.cameras
    SET
        no_people_count = CASE
            WHEN :people_present THEN 0
            WHEN state = 'closed' THEN close_threshold
            ELSE LEAST(no_people_count + 1, close_threshold)
        END,
        state = CASE
            WHEN :people_present THEN 'open'
            WHEN state = 'open' AND no_people_count + 1 >= close_threshold
                THEN 'closed'
            ELSE state
        END,
        state_changed_at = CASE
            WHEN :people_present AND state = 'closed' THEN :now
            WHEN NOT :people_present AND state = 'open'
                AND no_people_count + 1 >= close_threshold THEN :now
            ELSE state_changed_at
        END
    WHERE id = :camera_id
    RETURNING state, no_people_count, close_threshold
    """
)

# The `online` column is no longer written by the state machine.
# `LiveKitRoomReconciler` is the sole writer, sourced from
# `lkapi.room.list_participants` and the per-ingest `_last_frame_at`
# clock. Keeping the state-machine UPDATE focused on open/closed
# state means a paused detector tick does not pretend the camera is
# live, and a live publisher whose camera is AI-paused still gets
# `online=true` via the reconciler.


async def apply_tick_update(
    session: AsyncSession,
    *,
    camera_id: UUID,
    people_present: bool,
    motion_in_clip: bool,
) -> CameraStateAfter:
    """Atomic state machine update for one tick. Caller owns the
    transaction (we don't commit here).

    `people_present` alone drives transitions (the SQL never reads
    `motion_in_clip` — it is accepted purely for the analyses log).
    Motion's real job is upstream in camera_ingest: the cheap
    closed-state wake that decides whether a tick runs at all.

    Race-free under concurrent ticks — see module docstring.

    Raises ValueError if the camera row doesn't exist (signals a bug;
    the api-key auth path should have already resolved it).
    """
    # `now` is still bound for the `state_changed_at` CASE in the
    # UPDATE — the column is stamped only on a real open<->closed
    # transition, not on every tick.
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
    # stale state/count/changed_at values. Expire the identity map so
    # subsequent reads in the same session (callers, tests, or
    # downstream queries that join through Camera) see the committed
    # values from our UPDATE.
    session.expire_all()

    state, no_people_count, close_threshold = row
    return CameraStateAfter(
        state=state,
        no_people_count=no_people_count,
        close_threshold=close_threshold,
    )
