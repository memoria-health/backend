"""Single owner of the `cameras.app_present` column.

`app_present` answers ONE question: is the Memoria Sensor app currently
RUNNING on this camera's phone (polling us), as opposed to closed /
force-quit / phone-off? It is independent of `online` (publishing) and
`desired_state` (caregiver intent):

  online=true                  -> LIVE       (app is obviously running)
  online=false, present=true   -> STANDBY    ("Disconnected — tap Reconnect")
  online=false, present=false  -> APP CLOSED ("Open Memoria Sensor")

Like `camera_presence` owns `online`, THIS module is the single writer
of `app_present`. The poll handler only ever touches an in-memory
`PresenceCache` (no DB write on the 1s hot path); this sweeper compares
that cache to the DB every few seconds and flips the column ONLY on
transitions, so each write is a real change → one Realtime broadcast.

iPhone-ONLY scope: only paired iPhone sensors poll the desired-state
endpoint (Pi cameras authenticate with an ingest token and never hit
it; unpaired rows have no app at all). So the sweep restricts its reads
and writes to `source_type='iphone' AND api_key_hash IS NOT NULL`. A Pi
camera therefore stays at the column default (true) forever and can
never show "app closed" — it has no app to close.

Re-introducing a standalone sweep is deliberate and justified: unlike
the removed `StaleOnlineSweeper` (online is owned by each RoomWorker's
reconciler), presence is POLL-based and ROOM-INDEPENDENT — a camera
whose app is alive but not publishing has no RoomWorker, so nothing
else can observe it. Presence MUST be swept globally. Do not
"consolidate" this back out.

Best-effort: the loop swallows DB errors so a transient blip never
crashes the lifespan; the next tick re-derives the correct state.
"""

from __future__ import annotations

import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.services.detection.presence_cache import PresenceCache
from src.settings import Settings

log = logging.getLogger(__name__)


# Paired iPhone cameras only (the only rows that poll desired-state).
# We read id + current app_present, diff against the live set in Python,
# and issue one single-row UPDATE per transition. Doing the diff in
# Python (rather than an `id = ANY(:array)` UPDATE) keeps the SQL
# trivial and dodges array-binding edge cases; the row count is tiny
# (a few cameras per backend).
_SELECT_PRESENCE_STATE_SQL = text(
    """
    SELECT id, app_present
    FROM public.cameras
    WHERE source_type = 'iphone' AND api_key_hash IS NOT NULL
    """
)

# One-row transition writes. Each flips exactly one row, which the
# Realtime publication broadcasts to every subscribed client (web + iOS)
# for free over the existing `cameras` channel.
_SET_PRESENT_SQL = text(
    "UPDATE public.cameras SET app_present = true WHERE id = :camera_id"
)
_SET_ABSENT_SQL = text(
    "UPDATE public.cameras SET app_present = false WHERE id = :camera_id"
)


class PresenceSweeper:
    """Periodic task that reconciles `cameras.app_present` from the
    in-memory heartbeat cache. Owns the column; nothing else writes it.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        sessionmaker: async_sessionmaker[AsyncSession],
        cache: PresenceCache,
    ) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._cache = cache
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Seed the cache (restart grace) and launch the sweep loop.
        No-op when the feature is disabled — the column then stays at
        its true default and the poll handler's touch is harmless.
        """
        if not self._settings.sensor_presence_enabled:
            log.info("presence_sweeper.disabled")
            return
        await self._seed()
        self._task = asyncio.create_task(self._run())
        log.info(
            "presence_sweeper.started sweep=%ds stale=%ds",
            self._settings.sensor_presence_sweep_seconds,
            self._settings.sensor_presence_stale_seconds,
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _seed(self) -> None:
        """Mark every currently-paired iPhone camera as alive at startup
        so a restart never flashes a live sensor to "app closed" before
        its first poll. Genuinely-closed ones go stale within
        `stale_seconds`; the next sweep corrects them.
        """
        try:
            async with self._sessionmaker() as session:
                rows = (await session.execute(_SELECT_PRESENCE_STATE_SQL)).all()
            self._cache.seed(row[0] for row in rows)
            log.info("presence_sweeper.seeded count=%d", len(rows))
        except Exception:
            log.exception("presence_sweeper.seed_failed")

    async def _run(self) -> None:
        interval = self._settings.sensor_presence_sweep_seconds
        while True:
            try:
                await asyncio.sleep(interval)
                await self._sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Never let one bad tick kill the loop.
                log.exception("presence_sweeper.tick_failed")

    async def _sweep_once(self) -> None:
        """Compare the heartbeat cache to the DB and write only the rows
        whose app_present value actually changed.
        """
        alive = self._cache.alive_ids()
        try:
            async with self._sessionmaker() as session:
                rows = (await session.execute(_SELECT_PRESENCE_STATE_SQL)).all()
                to_present: list = []
                to_absent: list = []
                for camera_id, present in rows:
                    is_alive = camera_id in alive
                    if is_alive and not present:
                        to_present.append(camera_id)
                    elif not is_alive and present:
                        to_absent.append(camera_id)
                if not to_present and not to_absent:
                    return
                for camera_id in to_present:
                    await session.execute(
                        _SET_PRESENT_SQL, {"camera_id": camera_id}
                    )
                for camera_id in to_absent:
                    await session.execute(
                        _SET_ABSENT_SQL, {"camera_id": camera_id}
                    )
                await session.commit()
            log.info(
                "presence_sweeper.flipped present=%d absent=%d alive=%d",
                len(to_present), len(to_absent), len(alive),
            )
        except Exception:
            log.exception("presence_sweeper.sweep_failed")
