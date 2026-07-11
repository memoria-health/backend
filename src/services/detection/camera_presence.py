"""Centralized owner of the `cameras.online` column.

`cameras.online` answers exactly ONE question: "is this camera's video
actually flowing into its LiveKit room right now?" Every write to the
column goes through THIS module so the rules live in one place and the
trigger points are explicit. Grep the codebase for `online =` and you
should only find SQL here.

Trigger points — who writes `online`, when, and why:

  1. participant_joined webhook (cam_*)  -> NO direct write here.
     `livekit_webhooks` spawns/keeps a RoomWorker whose reconciler
     writes online=true ONCE frames actually arrive (the two-signal
     liveness rule). "Joined but no frames yet" must read as
     "connecting", not "online", so the true-write is deliberately
     deferred to the reconciler.

  2. participant_left webhook (cam_*)     -> mark_offline_by_identity().
     LiveKit Cloud has authoritatively confirmed the publisher is gone.
     This is THE load-bearing offline trigger and the fix for the
     long-standing "stuck online" bug: the reconciler used to be the
     sole offline writer, but it gets torn down with the RoomWorker on
     leave (livekit_webhooks -> ensure_room_left_if_empty), and on a
     clean leave the SDK's participant_disconnected handler pops the
     ingest before the reconciler's next poll — so neither path ever
     wrote false. The webhook is the one signal that always fires.

  3. room_finished webhook                -> mark_room_offline().
     The whole room went away; every camera in it is offline.

  4. periodic reconciler (per RoomWorker) -> set_camera_online_* /
     set_camera_offline_by_id() under the two-signal rule. This is the
     steady-state source of truth for a CONNECTED camera and the
     backstop that re-derives `online` from LiveKit REST presence when
     an SDK event was lost. See `room_reconciler.py`.

  5. backend startup sweep                -> sweep_stale_online().
     After a restart, any camera whose room is not currently hosting a
     publisher is forced offline so a flag that was true when we went
     down doesn't linger true forever (no reconciler ran while we were
     gone). See `DetectorWorker._reconcile_active_rooms`.

What is deliberately NOT a trigger point: caregiver disconnect
(`desired_state='disconnected'`). That flips `desired_state`; the wall
phone then leaves the room and trigger #2 fires. The clients already
render "offline" instantly off `desired_state`, so there is no need for
a redundant `online` write on the API path — keeping it out of
`camera_repo` keeps THIS module the single owner of the column.

Every public function here is best-effort and never raises: a webhook
ack or a startup must not be blocked on a transient DB failure. The
periodic reconciler and the startup sweep are the backstops if any one
write is lost.
"""

from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = logging.getLogger(__name__)


# ── SQL owned by this module ──────────────────────────────────────────
#
# The three statements below are issued by the reconciler (which manages
# its own session + no-op gating + fresh-start decision). Their exact
# text is contract: `test/unit/test_room_reconciler.py` string-matches on
# the column-set keywords, and `test/integration/test_reconciler_lifecycle.py`
# pins the real-Postgres effect. Do not reformat without updating both.

# online=true + state-machine reset. The reset (state='open',
# no_people_count=0) gives a fresh camera session a clean evaluation;
# without it a camera that previously closed after N no-people ticks
# would inherit state='closed' across reconnects and the LLM would skip
# almost every tick. The `AND online = false` guard makes this idempotent:
# the reconciler re-issues it every pass, so it is a DB no-op once the row
# is online (no redundant Realtime echo) AND self-heals a row that drifted
# false under a stale in-memory cache — the "stuck offline" bug.
SET_ONLINE_TRUE_FRESH_START_SQL = text(
    """
    UPDATE public.cameras
    SET online = true,
        state = 'open',
        no_people_count = 0,
        state_changed_at = :now
    WHERE id = :camera_id AND online = false
    """
)

# online=true WITHOUT the state-machine reset — the self-heal path for a
# row that drifted false while we still believed it was online. Same
# `AND online = false` guard: idempotent no-op once the row is online.
SET_ONLINE_TRUE_KEEP_STATE_SQL = text(
    """
    UPDATE public.cameras
    SET online = true
    WHERE id = :camera_id AND online = false
    """
)

# online=false by camera id. Used by the reconciler's absent/stale paths.
# `AND online = true` guard: idempotent no-op once the row is offline.
SET_ONLINE_FALSE_SQL = text(
    """
    UPDATE public.cameras
    SET online = false
    WHERE id = :camera_id AND online = true
    """
)


async def set_camera_online_fresh_start(
    session: AsyncSession, camera_id: UUID, now: datetime,
) -> bool:
    """online=true + state-machine reset, within the caller's session.
    Returns True iff the row actually flipped (was offline). Caller owns
    commit."""
    res = await session.execute(
        SET_ONLINE_TRUE_FRESH_START_SQL, {"camera_id": camera_id, "now": now},
    )
    return (res.rowcount or 0) > 0


async def set_camera_online_keep_state(
    session: AsyncSession, camera_id: UUID,
) -> bool:
    """online=true without the state reset. Returns True iff the row
    actually flipped (was offline). Caller owns commit."""
    res = await session.execute(
        SET_ONLINE_TRUE_KEEP_STATE_SQL, {"camera_id": camera_id},
    )
    return (res.rowcount or 0) > 0


async def set_camera_offline_by_id(
    session: AsyncSession, camera_id: UUID,
) -> bool:
    """online=false by camera id, within the caller's session. Returns
    True iff the row actually flipped (was online). Caller owns commit."""
    res = await session.execute(
        SET_ONLINE_FALSE_SQL, {"camera_id": camera_id},
    )
    return (res.rowcount or 0) > 0


# ── Webhook / startup trigger points (own their own session) ──────────
#
# These manage their own session+commit and swallow all errors, because
# their callers (the webhook route, FastAPI lifespan) must not fail or
# block on a transient DB issue.

# Idempotent: only flips a row that is currently online, so we never emit
# a no-op Realtime echo (the clients re-render off every cameras UPDATE)
# when the camera was already offline.
_OFFLINE_BY_IDENTITY_SQL = text(
    """
    UPDATE public.cameras
    SET online = false
    WHERE livekit_identity = :identity AND online = true
    """
)

_OFFLINE_BY_ROOM_SQL = text(
    """
    UPDATE public.cameras
    SET online = false
    WHERE livekit_room = :room AND online = true
    """
)

_SELECT_ONLINE_CAMERAS_WITH_ROOM_SQL = text(
    """
    SELECT id, livekit_room FROM public.cameras WHERE online = true
    """
)


async def mark_offline_by_identity(
    sessionmaker: async_sessionmaker, livekit_identity: str,
) -> bool:
    """participant_left trigger. Force the camera with this LiveKit
    identity offline. Returns True iff a row actually flipped (i.e. it
    was online). Never raises.
    """
    if not livekit_identity:
        return False
    try:
        async with sessionmaker() as session:
            res = await session.execute(
                _OFFLINE_BY_IDENTITY_SQL, {"identity": livekit_identity},
            )
            await session.commit()
        flipped = (res.rowcount or 0) > 0
        if flipped:
            log.info("camera_presence.offline identity=%s", livekit_identity)
        return flipped
    except Exception:
        log.exception(
            "camera_presence.mark_offline_by_identity_failed identity=%s",
            livekit_identity,
        )
        return False


async def mark_room_offline(
    sessionmaker: async_sessionmaker, livekit_room: str,
) -> int:
    """room_finished trigger. Force every camera in this room offline.
    Returns the number of rows flipped. Never raises.
    """
    if not livekit_room:
        return 0
    try:
        async with sessionmaker() as session:
            res = await session.execute(_OFFLINE_BY_ROOM_SQL, {"room": livekit_room})
            await session.commit()
        n = res.rowcount or 0
        if n:
            log.info(
                "camera_presence.room_offline room=%s flipped=%d", livekit_room, n,
            )
        return n
    except Exception:
        log.exception(
            "camera_presence.mark_room_offline_failed room=%s", livekit_room,
        )
        return 0


async def sweep_stale_online(
    sessionmaker: async_sessionmaker, active_rooms: set[str],
) -> int:
    """Startup trigger. Force offline any camera that is online=true in
    the DB but whose `livekit_room` is NOT currently hosting a publisher
    (i.e. not in `active_rooms`). Clears flags left true by an unclean
    shutdown — no reconciler ran while the backend was down.

    Cameras whose room IS in `active_rooms` are left alone: the
    freshly-spawned RoomWorker reconciler re-derives their true online
    state from LiveKit REST within one poll interval (~10s), so we don't
    blip a genuinely-live tile offline on every restart.

    Returns the number of rows flipped. Never raises.
    """
    try:
        async with sessionmaker() as session:
            rows = (
                await session.execute(_SELECT_ONLINE_CAMERAS_WITH_ROOM_SQL)
            ).all()
            stale_ids = [
                row[0]
                for row in rows
                if row[1] is None or row[1] not in active_rooms
            ]
            if not stale_ids:
                return 0
            for camera_id in stale_ids:
                await session.execute(
                    SET_ONLINE_FALSE_SQL, {"camera_id": camera_id},
                )
            await session.commit()
        log.info(
            "camera_presence.sweep_stale_online flipped=%d active_rooms=%d",
            len(stale_ids), len(active_rooms),
        )
        return len(stale_ids)
    except Exception:
        log.exception("camera_presence.sweep_stale_online_failed")
        return 0
