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
from uuid import UUID

from livekit import api as lkapi

from src.services.detection import camera_presence
from src.services.detection.motion_cache import MotionCache
from src.services.detection.room_worker import RoomWorker
from sqlalchemy import select, text

from src.db.camera import Camera
from src.db.session import get_engine
from src.services.identity.identity_router import IdentityRouter

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncConnection, async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)

# Postgres advisory-lock key for "I am the authoritative detector
# worker." Two backends pointed at the same database (a common dev
# footgun: forgetting to kill an old uvicorn before starting a new one)
# both spin up reconcilers that write `cameras.online` and both spawn
# detector_worker_* identities into LiveKit. They then fight, and the
# `online` column ping-pongs as each backend's `_last_frame_at` falls
# stale relative to the other. The advisory lock makes the second
# starter refuse cleanly instead of silently corrupting state.
#
# Key is a stable signed 64-bit integer. Any stable value works as
# long as it doesn't collide with other advisory locks the application
# takes. We don't take any others today, so collision risk is zero.
# `0x4E454D52444C434B` is ASCII "NEMRDLCK" — 8 bytes exactly, fits in
# Postgres bigint (which `pg_try_advisory_lock` requires).
_DETECTOR_WORKER_LOCK_KEY: int = 0x4E454D52444C434B  # ASCII "NEMRDLCK"


class DetectorWorkerLockUnavailableError(RuntimeError):
    """Raised by `DetectorWorker.start()` when another process already
    holds the detector-worker advisory lock. The FastAPI lifespan lets
    this propagate so uvicorn surfaces the failure loudly at startup
    rather than running degraded.
    """

# Upper bound on how long a RoomWorker.stop() can hold the DetectorWorker
# lock during `ensure_room_left_if_empty`. 10s is generous for a clean
# LiveKit disconnect (typically <500ms) but short enough that a hung
# disconnect doesn't park subsequent join/leave operations forever.
_STOP_TIMEOUT: float = 10.0

# Wall-clock upper bound on the startup-time LiveKit reconciliation.
# list_rooms is one HTTPS round-trip; each ensure_room_joined does an
# async LiveKit Room.connect. For a couple dozen patient rooms this
# usually completes in <10s; 30s gives headroom for a degraded network
# without indefinitely blocking the FastAPI process from accepting
# webhook traffic.
_RECONCILE_TIMEOUT: float = 30.0

# Prefix on LiveKit room names that camera publishers join. Only these
# rooms are candidates for reconciliation — egress / talk / test rooms
# don't need detector workers.
_PATIENT_ROOM_PREFIX: str = "patient_"

# Identity prefix that every RoomWorker uses (mirrored from
# room_worker._WORKER_IDENTITY_PREFIX). Defined here so the reaper
# doesn't have to import a private from room_worker.
_DETECTOR_WORKER_IDENTITY_PREFIX: str = "detector_worker_"


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
        # Identity provider: the sidecar, period (2026-07-26 — the
        # legacy in-process BodyReID brain was deleted once the sidecar
        # proved out live). Per-tick identity is synthesized from the
        # GPU-host identity service's /state; the backend does zero
        # video identity processing. Fail-open on an unreachable
        # service is the sidecar's contract.
        # IdentityRouter, not IdentityClient: a home is served by whichever
        # GPU node the roster assigned it to, so the endpoint is per-patient
        # rather than global. A single client pointed at one URL silently
        # returns no identity at all for every home on any other node.
        # The router is a drop-in — `label_clip` already carries patient_id.
        self._reid = IdentityRouter(
            sessionmaker=sessionmaker,
            settings=settings,
        )
        log.info("identity provider: router (per-patient, roster-resolved)")
        self._rooms: dict[str, RoomWorker] = {}
        self._lock = asyncio.Lock()
        self._started = False
        # Shared `lkapi.LiveKitAPI` client passed into every RoomWorker
        # so its `LiveKitRoomReconciler` can poll `list_participants`
        # without each pod opening + closing its own HTTPS client per
        # poll. Constructed in `start()` after the creds check; closed
        # in `stop()`. None when LIVEKIT_* settings are not configured.
        self._lk_api: object | None = None
        # Dedicated long-lived connection holding the
        # `_DETECTOR_WORKER_LOCK_KEY` advisory lock for the lifetime of
        # this worker. Postgres releases advisory locks when the
        # session ends, so we keep the connection open until `stop()`.
        self._lock_conn: "AsyncConnection | None" = None

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
        # Acquire the singleton advisory lock BEFORE any other
        # side-effecting work (LiveKit connections, room reconciliation,
        # DB writes). Failure here means another backend is already
        # writing to this Postgres — refuse to start so we don't
        # corrupt state.
        await self._acquire_singleton_lock()
        self._started = True
        # Single shared LiveKit REST client for every RoomWorker's
        # reconciler. Avoids per-poll HTTPS client construction.
        self._lk_api = lkapi.LiveKitAPI(
            self._settings.livekit_url,
            self._settings.livekit_api_key,
            self._settings.livekit_api_secret,
        )
        log.info("detector_worker.start awaiting webhook discovery")
        # Reconcile with LiveKit: at backend startup, query for active
        # rooms and spawn workers for any whose participant_joined
        # webhook we missed (e.g. because backend was redeploying when
        # the camera connected). MUST run AFTER `_started=True` so the
        # ensure_room_joined calls inside don't early-return on the
        # not-started check.
        #
        # Bounded so a slow / unreachable LiveKit at boot doesn't park
        # FastAPI's lifespan. Failures are logged + swallowed because
        # the webhook path keeps working for any NEW connects — we'd
        # rather come up degraded than not come up at all.
        try:
            await asyncio.wait_for(
                self._reconcile_active_rooms(),
                timeout=_RECONCILE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(
                "detector_worker.reconcile timed out after %.0fs — "
                "proceeding; webhook path handles new connects",
                _RECONCILE_TIMEOUT,
            )
        except Exception:
            log.exception(
                "detector_worker.reconcile crashed — proceeding; "
                "webhook path handles new connects",
            )

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
        # Close the sidecar's HTTP pool — same close-what-you-open
        # discipline as _lk_api below (Phase 0 review finding #1).
        try:
            await self._reid.aclose()
        except Exception:
            log.exception("detector_worker.stop reid.aclose failed")
        # Close the shared LiveKit REST client AFTER every reconciler
        # has stopped using it. Best-effort; an aclose failure
        # mid-shutdown is logged but does not block the lifespan.
        if self._lk_api is not None:
            try:
                await self._lk_api.aclose()
            except Exception:
                log.exception("detector_worker.stop lkapi.aclose failed")
            self._lk_api = None
        # Release the advisory lock LAST so any in-flight
        # `cameras.online` writes from the reconciler complete first
        # under our exclusive hold. Closing the connection
        # auto-releases the lock; the explicit unlock call is for
        # symmetry + cleaner Postgres logs.
        await self._release_singleton_lock()
        log.info("detector_worker.stop done")

    # ── singleton advisory lock ───────────────────────────────────

    async def _acquire_singleton_lock(self) -> None:
        """Acquire the Postgres advisory lock that gates this process
        as the authoritative detector worker.

        `pg_try_advisory_lock` returns True if we got the lock, False
        if another session holds it. Held for the lifetime of
        `self._lock_conn`; Postgres releases it automatically when the
        connection closes.
        """
        engine = get_engine()
        conn = await engine.connect()
        try:
            result = await conn.execute(
                text("SELECT pg_try_advisory_lock(:k)"),
                {"k": _DETECTOR_WORKER_LOCK_KEY},
            )
            acquired = bool(result.scalar())
        except Exception:
            await conn.close()
            raise
        if not acquired:
            await conn.close()
            raise DetectorWorkerLockUnavailableError(
                "Another nemara-backend process already holds the "
                "detector-worker advisory lock on this Postgres. "
                "Refusing to start to avoid dual-write conflicts on "
                "cameras.online + duplicate detector_worker_* identities "
                "kicking each other off LiveKit. Stop the other process "
                "and retry."
            )
        self._lock_conn = conn
        log.info("detector_worker.singleton_lock_acquired key=%d", _DETECTOR_WORKER_LOCK_KEY)

    async def _release_singleton_lock(self) -> None:
        """Release the advisory lock and close the holder connection.
        Idempotent."""
        conn = self._lock_conn
        if conn is None:
            return
        self._lock_conn = None
        try:
            await conn.execute(
                text("SELECT pg_advisory_unlock(:k)"),
                {"k": _DETECTOR_WORKER_LOCK_KEY},
            )
        except Exception:
            log.exception("detector_worker.singleton_unlock_failed")
        try:
            await conn.close()
        except Exception:
            log.exception("detector_worker.singleton_lock_conn_close_failed")

    # ── webhook fast-path signal ───────────────────────────────────

    async def signal_reconcile(self, room_name: str) -> None:
        """Webhook fast-path hint: trigger an immediate reconciler poll
        for the given room. No-op if we don't have a RoomWorker for
        the room yet — the next webhook (or the startup reconciler
        the periodic poll itself) will catch it.
        """
        async with self._lock:
            worker = self._rooms.get(room_name)
        if worker is None:
            return
        try:
            await worker._reconciler.signal_tick()
        except Exception:
            log.exception(
                "detector_worker.signal_reconcile failed room=%s",
                room_name,
            )

    # ── startup reconciliation ─────────────────────────────────────

    async def _owned_room_names(self) -> set[str]:
        """Return the set of LiveKit room names that THIS backend's
        Supabase owns — derived from `cameras.livekit_room` across
        all rows. Used to scope reconciliation + zombie reaper so we
        don't accidentally touch rooms belonging to a different
        backend instance pointed at the same LiveKit deployment
        (a common dev situation when two engineers share a LiveKit
        Cloud project but have separate local Supabases).

        Empty result is fine — means we have no cameras yet, so
        reconciliation has nothing to do.
        """
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(
                    select(Camera.livekit_room).where(
                        Camera.livekit_room.is_not(None)
                    )
                )
            ).scalars().all()
        # livekit_room is nullable, but the SQL filters NULLs. Cast
        # to set for O(1) membership checks downstream.
        return {r for r in rows if r}

    async def _reconcile_active_rooms(self) -> None:
        """One-shot at startup: enumerate LiveKit rooms and join each
        `patient_*` room that has at least one participant. Closes the
        redeploy-orphan gap where the backend restarts while a camera
        is already connected — LiveKit only fires `participant_joined`
        on the actual join, so without this we'd silently sit idle
        with no detector worker for the live stream.

        Failures here MUST NOT raise to the caller (start() catches
        anyway, but we double-up here so a single bad room can't
        abort the loop over the others). The webhook path keeps
        working for any NEW connects, so degraded > broken.

        Per the user's design choice ("spawn workers for ALL active
        rooms — not filtered"), we don't inspect participant identities
        before joining. A patient room with no `cam_*` participants is
        a benign idle case: the RoomWorker connects, no track_subscribed
        fires, no ticks happen, zero analyses written. When the room
        actually empties, the next `participant_left` webhook tears it
        down via `ensure_room_left_if_empty`.
        """
        # Scope reconciliation to rooms WE own. The LiveKit deployment
        # may be shared with other backend instances (sibling devs, a
        # different env pointed at the same LiveKit project, etc).
        # Joining their rooms would pollute their state with our worker
        # bots and our DB schema can't see their patients anyway.
        owned = await self._owned_room_names()
        log.info(
            "detector_worker.reconcile own %d patient room(s) in our DB",
            len(owned),
        )
        lk = lkapi.LiveKitAPI(
            self._settings.livekit_url,
            self._settings.livekit_api_key,
            self._settings.livekit_api_secret,
        )
        try:
            try:
                rooms_resp = await lk.room.list_rooms(lkapi.ListRoomsRequest())
            except Exception:
                log.exception(
                    "detector_worker.reconcile list_rooms failed — "
                    "no rooms reconciled this boot",
                )
                return
            candidates: list[str] = []
            skipped_foreign = 0
            for r in rooms_resp.rooms:
                if not r.name.startswith(_PATIENT_ROOM_PREFIX):
                    continue
                if r.num_participants == 0:
                    continue
                if r.name not in owned:
                    # Belongs to another backend's Supabase. Leaving
                    # it alone is the right call — they own its
                    # lifecycle, not us.
                    skipped_foreign += 1
                    continue
                candidates.append(r.name)
            log.info(
                "detector_worker.reconcile found %d active OWNED room(s); "
                "skipped %d foreign room(s)",
                len(candidates), skipped_foreign,
            )
            # STALE-ONLINE SWEEP: clear `cameras.online` left true by an
            # unclean shutdown. Any owned camera whose room is NOT a
            # candidate (no participants right now) cannot be live, so
            # force it offline — no reconciler ran while we were down to
            # do it. Cameras in candidate rooms are left alone: the
            # RoomWorker we're about to spawn re-derives their true
            # online state from REST within one poll. Only runs when
            # `list_rooms` succeeded (we're past its early-return), so we
            # never blanket-offline on a transient LiveKit failure.
            # Guarded on sessionmaker for the unit tests that build a
            # worker with sessionmaker=None.
            if self._sessionmaker is not None:
                await camera_presence.sweep_stale_online(
                    self._sessionmaker, set(candidates),
                )
            # ZOMBIE REAPER: before joining, kick any leftover
            # detector_worker_* participants in rooms WE own. At THIS
            # moment in start() we own zero RoomWorker instances, so
            # any worker bot present in OUR rooms must be from a
            # previous instance of THIS backend that got SIGKILL'd
            # or crashed before lifespan could clean up. We never
            # touch detector_workers in foreign rooms — those belong
            # to whatever backend owns that room. Bounded per-room:
            # if remove_participant fails, log and continue — ghost
            # will eventually time out on LiveKit's side (~5 min).
            await self._reap_zombie_workers(lk, candidates)
            for room_name in candidates:
                try:
                    await self.ensure_room_joined(room_name)
                except Exception:
                    log.exception(
                        "detector_worker.reconcile join_failed room=%s — "
                        "continuing with remaining rooms",
                        room_name,
                    )
        finally:
            try:
                await lk.aclose()
            except Exception:
                log.exception(
                    "detector_worker.reconcile aclose_failed — "
                    "reconciliation considered successful",
                )

    async def _reap_zombie_workers(
        self, lk: "lkapi.LiveKitAPI", room_names: list[str],
    ) -> None:
        """Per-room: list participants, kick any leftover
        `detector_worker_*` identities. Called from
        `_reconcile_active_rooms` at startup, when by definition our
        process has not yet spawned any workers, so every
        `detector_worker_*` present in LiveKit is a zombie from a
        prior process that didn't shut down cleanly.

        Failures (list_participants raises, remove_participant raises)
        are caught per-room/per-participant — the loop keeps going so
        one bad room/zombie doesn't abort the rest. Logged loudly so
        production can alert on persistent reaper failures.
        """
        total_reaped = 0
        for room_name in room_names:
            try:
                parts_resp = await lk.room.list_participants(
                    lkapi.ListParticipantsRequest(room=room_name),
                )
            except Exception:
                log.exception(
                    "detector_worker.reap list_participants_failed room=%s — "
                    "skipping reap for this room",
                    room_name,
                )
                continue
            for p in parts_resp.participants:
                if not p.identity.startswith(_DETECTOR_WORKER_IDENTITY_PREFIX):
                    continue
                try:
                    await lk.room.remove_participant(
                        lkapi.RoomParticipantIdentity(
                            room=room_name, identity=p.identity,
                        ),
                    )
                    total_reaped += 1
                    log.info(
                        "detector_worker.reap kicked zombie room=%s identity=%s",
                        room_name, p.identity,
                    )
                except Exception:
                    log.exception(
                        "detector_worker.reap remove_participant_failed "
                        "room=%s identity=%s — will rely on LiveKit "
                        "server-side timeout",
                        room_name, p.identity,
                    )
        if total_reaped > 0:
            log.info(
                "detector_worker.reap reaped %d zombie worker(s) total",
                total_reaped,
            )

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
                reid=self._reid,
                lk_api=self._lk_api,
                # When the RoomWorker has been sitting in an empty
                # patient_* room for the idle window with zero
                # cam_* publishers, it calls back here to ask for
                # removal. Common cause: startup reconciliation
                # joined a room whose camera was deleted (or never
                # paired with a wall phone) while we were down.
                # Without this hook the worker would sit there
                # forever burning a LiveKit identity slot.
                on_idle_timeout=self._evict_room,
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

    async def _evict_room(self, room_name: str) -> None:
        """Callback target for `RoomWorker._monitor_idle`. Pops the
        worker from `self._rooms` and awaits its `stop()`. Idempotent
        and safe to call concurrently with other paths (lock-guarded
        pop + bounded stop, mirrors `ensure_room_left_if_empty`).

        IMPORTANT: must be called from a task that is NOT inside the
        target room's supervisor task — `worker.stop()` cancels the
        supervisor and awaits it, which would deadlock if the
        supervisor itself called us. The idle monitor lives in a
        sibling task to the supervisor for exactly this reason.

        Never raises — bounded by `_STOP_TIMEOUT`. On timeout we
        drop the reference and trust LiveKit's session timeout to
        reap server-side; logged loudly for observability.
        """
        async with self._lock:
            worker = self._rooms.pop(room_name, None)
            if worker is None:
                # Already removed by participant_left or a concurrent
                # eviction. Idempotent no-op.
                return
        try:
            await asyncio.wait_for(worker.stop(), timeout=_STOP_TIMEOUT)
        except asyncio.TimeoutError:
            log.warning(
                "detector_worker.evict stop() timed out after %ss "
                "room=%s — dropping reference, LiveKit participant "
                "will time out server-side",
                _STOP_TIMEOUT, room_name,
            )
        except Exception:
            log.exception(
                "detector_worker.evict failed room=%s",
                room_name,
            )
        else:
            log.info(
                "detector_worker.evict room=%s — idle teardown complete",
                room_name,
            )

    async def stop_worker_for_camera(
        self, camera_id: UUID, *, timeout: float = 5.0,
    ) -> bool:
        """Find and stop the CameraIngest for `camera_id` across all
        currently-joined rooms. Called from the camera DELETE API
        handler so the in-process worker tears down at the same instant
        the DB row goes away.

        Never raises — all failures (timeout, missing ingest, raised
        exception from RoomWorker.remove_ingest) are caught, logged,
        and converted to a `False` return. The caller (API handler)
        treats this as best-effort; the CameraIngest self-destruct
        path (P1) catches any zombies within ~30s as a safety net.

        The DetectorWorker's `_rooms` dict is keyed by `room_name`,
        but the camera DELETE knows only `camera_id`. We scan all
        rooms looking for a CameraIngest with the matching ID — O(N×M)
        where N=rooms (≤ hundreds) and M=cameras per room (≤ ~3).
        Acceptable for production scale.

        Concurrency: holds `self._lock` only for the scan (one
        critical section). The actual `remove_ingest` call runs
        OUTSIDE the lock, mirroring the `ensure_room_joined` pattern
        — we don't want a slow LiveKit disconnect inside remove_ingest
        to block concurrent webhook calls behind the stop.

        Note: the most recent in-flight tick body may still complete
        AFTER this returns, but its analyses write will silently fail
        with an FK violation when the camera row is gone (CASCADE
        deleted everything that references it). Acceptable — no
        zombie state can survive past one tick window.

        Returns True iff an ingest was found AND its remove_ingest
        completed within `timeout`. False on not-found, timeout, or
        any exception.
        """
        async with self._lock:
            target: tuple[RoomWorker, str] | None = None
            for room_worker in self._rooms.values():
                for identity, ingest in room_worker._ingests_by_identity.items():
                    if ingest.camera_id == camera_id:
                        target = (room_worker, identity)
                        break
                if target is not None:
                    break
        if target is None:
            log.info(
                "detector_worker.stop_worker_for_camera camera=%s — "
                "no active ingest found (paired but offline, or already removed)",
                camera_id,
            )
            return False
        room_worker, identity = target
        try:
            await asyncio.wait_for(
                room_worker.remove_ingest(identity), timeout=timeout,
            )
        except asyncio.TimeoutError:
            log.warning(
                "detector_worker.stop_worker_for_camera camera=%s identity=%s "
                "— remove_ingest timed out after %.1fs; P1 self-destruct "
                "will reap the zombie",
                camera_id, identity, timeout,
            )
            return False
        except Exception:
            log.exception(
                "detector_worker.stop_worker_for_camera camera=%s identity=%s "
                "— remove_ingest raised; P1 self-destruct will reap the zombie",
                camera_id, identity,
            )
            return False
        log.info(
            "detector_worker.stop_worker_for_camera camera=%s identity=%s — stopped",
            camera_id, identity,
        )
        return True

    # ── diagnostics ────────────────────────────────────────────────

    def active_rooms(self) -> list[str]:
        """Snapshot of currently-joined room names. Used by the test
        suite and future admin endpoints.
        """
        return list(self._rooms.keys())
