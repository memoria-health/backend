"""Top-level detector worker.

`DetectorWorker` is the asyncio entry point started by FastAPI's
lifespan. It owns:

  - one `MotionCache` shared across every `RoomWorker`,
  - a map of `{room_name: RoomWorker}` for currently-joined rooms,
  - the lock that serializes joins/leaves so the webhook handler and
    a future periodic refresh can't collide.

The worker doesn't poll the DB on a timer. Discovery is webhook-driven
— `src/api/livekit_webhooks.py` calls `ensure_room_joined()` and
`ensure_room_left_if_empty()` as LiveKit Cloud fires
`participant_joined` / `participant_left`. Until a wall phone is
paired and publishing, the worker is genuinely zero-cost: no rooms
joined, no SDK connections open.

`start()` short-circuits if `LIVEKIT_URL` / `LIVEKIT_API_KEY` /
`LIVEKIT_API_SECRET` aren't configured — the worker has nothing to
mint tokens with, so we log loudly and skip the lifespan attach
rather than letting the first webhook crash.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.services.detection.motion_cache import MotionCache
from src.services.detection.room_worker import RoomWorker

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)

# Upper bound on how long a RoomWorker.stop() can hold the DetectorWorker
# lock during `ensure_room_left_if_empty`. 10s is generous for a clean
# LiveKit disconnect (typically <500ms) but short enough that a hung
# disconnect doesn't park subsequent join/leave operations forever.
_STOP_TIMEOUT: float = 10.0


class DetectorWorker:
    """Single instance per FastAPI process. Wired into the lifespan
    in `src/main.py`.
    """

    def __init__(
        self,
        *,
        settings: "Settings",
        sessionmaker: "async_sessionmaker",
    ) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._motion_cache = MotionCache(
            ttl_seconds=settings.livekit_motion_state_ttl_seconds,
        )
        self._rooms: dict[str, RoomWorker] = {}
        self._lock = asyncio.Lock()
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if not (self._settings.livekit_url
                and self._settings.livekit_api_key
                and self._settings.livekit_api_secret):
            log.error(
                "detector_worker.start cannot start — "
                "LIVEKIT_URL/KEY/SECRET not configured",
            )
            return
        self._started = True
        log.info("detector_worker.start awaiting webhook discovery")

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        async with self._lock:
            rooms = list(self._rooms.values())
            self._rooms.clear()
        # Stop every room worker concurrently — they don't depend on
        # each other and we'd rather not serialize their disconnect
        # round-trips at shutdown.
        await asyncio.gather(
            *(room.stop() for room in rooms),
            return_exceptions=True,
        )
        log.info("detector_worker.stop done")

    # ── webhook-driven discovery ───────────────────────────────────

    async def ensure_room_joined(self, room_name: str) -> None:
        """Idempotent — if the worker is already in this room, no-op.
        Otherwise spawn a `RoomWorker`. Called from
        `participant_joined` webhooks.
        """
        if not self._started:
            log.debug(
                "detector_worker.ensure_joined ignored room=%s (worker not started)",
                room_name,
            )
            return
        async with self._lock:
            if room_name in self._rooms:
                return
            worker = RoomWorker(
                room_name=room_name,
                settings=self._settings,
                sessionmaker=self._sessionmaker,
                motion_cache=self._motion_cache,
            )
            self._rooms[room_name] = worker
        # Connect outside the lock — `Room.connect` does network I/O
        # and we don't want to block other webhook calls behind it.
        try:
            await worker.start()
        except Exception:
            log.exception(
                "detector_worker.ensure_joined failed room=%s — backing out",
                room_name,
            )
            async with self._lock:
                self._rooms.pop(room_name, None)

    async def ensure_room_left_if_empty(self, room_name: str) -> None:
        """Called from `participant_left` webhooks for cam_* identities.
        Tears the room worker down. The next camera join re-fires the
        webhook and we rejoin from scratch.

        `worker.stop()` runs **inside** `self._lock` so a concurrent
        `ensure_room_joined` for the same room can't spawn a new
        RoomWorker (and a new LiveKit connection) while the old one is
        still mid-disconnect. Before this was outside the lock, every
        cam-leave-then-rejoin race left a zombie detector_worker
        participant in the LiveKit room — they accumulated over
        Railway redeploys and started producing duplicate
        `public.analyses` rows once their respective track subscribers
        each got their own CameraIngest.

        The stop is bounded by `_STOP_TIMEOUT` so a hung LiveKit
        disconnect can't park the lock forever. On timeout we drop the
        reference and log loudly; the LiveKit server eventually times
        out the orphaned participant on its own (~5min).
        """
        if not self._started:
            return
        async with self._lock:
            worker = self._rooms.pop(room_name, None)
            if worker is None:
                return
            try:
                await asyncio.wait_for(worker.stop(), timeout=_STOP_TIMEOUT)
            except asyncio.TimeoutError:
                log.warning(
                    "detector_worker.ensure_left stop() timed out "
                    "after %ss room=%s — dropping reference, LiveKit "
                    "participant will time out server-side",
                    _STOP_TIMEOUT, room_name,
                )
            except Exception:
                log.exception(
                    "detector_worker.ensure_left failed room=%s",
                    room_name,
                )

    # ── diagnostics ────────────────────────────────────────────────

    def active_rooms(self) -> list[str]:
        """Snapshot of currently-joined room names. Used by the test
        suite and future admin endpoints.
        """
        return list(self._rooms.keys())
