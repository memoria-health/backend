"""`PiIngestManager`: lifecycle controller for all Pi camera relays.

Driven by the standalone worker process at
`src/workers/pi_ingest_worker.py`. At startup it queries the
`cameras` table for every row with `source_type='pi'` and a non-null
`pi_stream_url`. For each, it spawns a `RelayPublisher.run()` task.

Periodic re-sync: every `db_poll_seconds` the manager re-queries the
table, starts relays for newly-added Pi cameras, and cancels relays
for cameras that have been deleted or had their source_type flipped
away from 'pi'. This is the only mechanism that connects "caregiver
adds a Pi via /v1/cameras/pi-provision" to "relay starts streaming".

Reconnect: each per-camera relay task is supervised. On any
exception (MJPEG read error, LiveKit disconnect, JPEG decode storm)
the manager restarts the relay with exponential backoff capped at
60 s. A clean stream end (return from `run()`) is also re-attempted,
since it usually means the Pi rebooted briefly.

The manager itself runs forever until cancelled. Cancellation
propagates to all child relay tasks, which clean up their LiveKit
connections via the existing async-with-room pattern.

Testability: the relay coroutine factory is injectable. Tests can
pass a fake factory that records "I would have started this camera"
without actually opening a LiveKit connection. The DB poll is also
exposed so tests can drive it deterministically rather than waiting
for the polling timer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.db.camera import Camera
from src.services.ingest.pi.relay_publisher import (
    RelayConfig,
    default_relay_coroutine,
)
from src.settings import Settings

log = logging.getLogger(__name__)


# Backoff schedule used when a relay task fails. Doubles up to a 60s
# cap, resets to the floor only after the relay has stayed healthy
# for `_RELAY_HEALTHY_THRESHOLD_S` seconds. Without that threshold,
# a Pi that ends connections cleanly after a tenth of a second
# (e.g., accept-then-close in a tight loop) would never accumulate
# backoff and would hammer the Pi at the floor rate forever.
_RELAY_BACKOFF_INITIAL_S: float = 2.0
_RELAY_BACKOFF_MAX_S: float = 60.0
# A relay must have been running for at least this long before its
# exit is considered a transient and backoff is reset to the floor.
# 30 s is well past the 5 s motion-heartbeat cadence and far longer
# than any reasonable Pi MJPEG handshake, so a fast-failing Pi cannot
# keep tricking the manager into resetting.
_RELAY_HEALTHY_THRESHOLD_S: float = 30.0

# How often the manager re-queries the DB for new / removed Pi
# cameras. 30 s is fast enough that a caregiver provisioning a new
# camera sees it come online within a minute, slow enough that the
# DB query cost is negligible.
DEFAULT_DB_POLL_SECONDS: float = 30.0


@dataclass(frozen=True)
class _RelayState:
    """Tracks one running camera's task + its current backoff."""

    task: asyncio.Task[None]
    backoff_s: float


RelayCoroutineFactory = Callable[
    [RelayConfig, Settings], Awaitable[None],
]


class PiIngestManager:
    """Supervisor for all Pi camera relays.

    Owns one asyncio task per camera plus one DB-poll task. All tasks
    are children of the asyncio context the manager was started in;
    cancelling the manager cancels everything cleanly.
    """

    def __init__(
        self,
        settings: Settings,
        session_factory: async_sessionmaker,
        *,
        relay_coroutine_factory: RelayCoroutineFactory | None = None,
        db_poll_seconds: float = DEFAULT_DB_POLL_SECONDS,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._relay_coroutine_factory = (
            relay_coroutine_factory or default_relay_coroutine
        )
        self._db_poll_seconds = db_poll_seconds
        self._relays: dict[UUID, _RelayState] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._stopped: bool = False

    async def start(self) -> None:
        """Begin the lifecycle. Does an immediate DB sync so relays
        for existing Pi cameras start without waiting for the first
        poll tick.
        """
        if self._poll_task is not None:
            raise RuntimeError("PiIngestManager already started")
        await self.reconcile_once()
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="pi_ingest_db_poll"
        )

    async def stop(self) -> None:
        """Cancel the poll task and every per-camera relay task.
        Waits for all tasks to drain before returning so the caller
        can be sure no LiveKit connections remain open.
        """
        self._stopped = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
            self._poll_task = None

        for state in list(self._relays.values()):
            state.task.cancel()
        for state in list(self._relays.values()):
            try:
                await state.task
            except (asyncio.CancelledError, Exception):
                pass
        self._relays.clear()

    async def reconcile_once(self) -> None:
        """One DB poll cycle. Public so tests can drive sync explicitly
        instead of waiting for the polling timer.
        """
        rows = await self._load_pi_cameras()
        wanted_ids = {row.id for row in rows}
        # Cancel relays whose camera rows have been deleted or had
        # source_type flipped off 'pi'.
        for camera_id in list(self._relays.keys()):
            if camera_id not in wanted_ids:
                log.info(
                    "pi_ingest.relay_removed camera=%s (no longer in DB)",
                    camera_id,
                )
                self._relays[camera_id].task.cancel()
                # We leave the entry in `_relays` until the task done
                # callback removes it, so a concurrent reconcile does
                # not race to re-create it.

        # Start relays for newly-present camera rows.
        for row in rows:
            if row.id in self._relays:
                continue
            self._spawn_relay_for(row)

    # ── internals ────────────────────────────────────────────────

    def _spawn_relay_for(
        self, row: Camera, *, initial_backoff_s: float = _RELAY_BACKOFF_INITIAL_S,
    ) -> None:
        """Start one supervised relay task for a camera row.

        The task is a thin wrapper that calls `_run_relay_with_backoff`
        which owns the retry loop for THIS camera. When that loop
        exits (only on manager stop or DB removal), the done callback
        clears the entry from `_relays`.
        """
        assert row.pi_stream_url is not None  # enforced by DB CHECK
        config = RelayConfig(
            camera_id=row.id,
            camera_identity=row.livekit_identity or f"cam_{row.id.hex[:8]}",
            room_name=row.livekit_room or f"patient_{row.patient_id}",
            pi_stream_url=row.pi_stream_url,
            pi_audio_url=row.pi_audio_url,
        )
        task = asyncio.create_task(
            self._run_relay_with_backoff(config),
            name=f"pi_relay_{row.id}",
        )
        self._relays[row.id] = _RelayState(
            task=task, backoff_s=initial_backoff_s,
        )
        task.add_done_callback(
            lambda _t, cid=row.id: self._relays.pop(cid, None)
        )
        log.info(
            "pi_ingest.relay_started camera=%s url=%s",
            row.id, row.pi_stream_url,
        )

    async def _run_relay_with_backoff(self, config: RelayConfig) -> None:
        """Per-camera supervisor: runs the relay, on failure or clean
        exit waits backoff seconds and tries again. Exits cleanly
        only when the task is cancelled (manager stop or DB removal).

        Backoff reset rule: a relay must have run for at least
        `_RELAY_HEALTHY_THRESHOLD_S` before its exit is treated as a
        transient (and backoff reset to the floor). A relay that
        fails or exits cleanly within that window keeps escalating
        backoff up to the cap, so a misbehaving Pi cannot induce a
        tight reconnect loop.
        """
        loop = asyncio.get_event_loop()
        backoff = _RELAY_BACKOFF_INITIAL_S
        ran_seconds: float = 0.0
        while not self._stopped:
            start_time = loop.time()
            try:
                await self._relay_coroutine_factory(config, self._settings)
                ran_seconds = loop.time() - start_time
                log.info(
                    "pi_ingest.relay_ended_clean camera=%s ran=%.1fs, restarting",
                    config.camera_id, ran_seconds,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                ran_seconds = loop.time() - start_time
                log.warning(
                    "pi_ingest.relay_failed camera=%s ran=%.1fs err=%r retry_in=%.1fs",
                    config.camera_id, ran_seconds, exc, backoff,
                )

            if self._stopped:
                return
            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                raise

            # Reset backoff only if the relay was healthy long enough
            # to convince us the underlying issue was transient.
            if ran_seconds >= _RELAY_HEALTHY_THRESHOLD_S:
                backoff = _RELAY_BACKOFF_INITIAL_S
            else:
                backoff = min(backoff * 2, _RELAY_BACKOFF_MAX_S)

    async def _poll_loop(self) -> None:
        """Periodic DB reconciliation. Cancellation-safe (asyncio
        sleep raises CancelledError which we re-raise to terminate).
        """
        while not self._stopped:
            try:
                await asyncio.sleep(self._db_poll_seconds)
            except asyncio.CancelledError:
                raise
            try:
                await self.reconcile_once()
            except Exception:
                log.exception("pi_ingest.reconcile_failed")

    async def _load_pi_cameras(self) -> list[Camera]:
        """Load every Pi camera row from the DB. Filters at the SQL
        layer so the manager never sees iPhone cameras.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(Camera)
                .where(Camera.source_type == "pi")
                .where(Camera.pi_stream_url.is_not(None))
            )
            return list(result.scalars().all())

    # ── introspection (for tests + debugging) ────────────────────

    def active_camera_ids(self) -> set[UUID]:
        """Set of camera_ids the manager currently has a running
        relay task for. Useful for tests asserting reconciliation
        produced the expected set without inspecting private state.
        """
        return {
            cid for cid, state in self._relays.items()
            if not state.task.done()
        }
