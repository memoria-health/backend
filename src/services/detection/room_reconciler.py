"""Periodic REST-driven LiveKit participant reconciler.

Authoritative answer to "is this camera publishing to its LiveKit room
right now" comes from LiveKit Cloud's REST API
(`RoomService.list_participants`), not from the SDK's in-memory
event-sourced `room.remote_participants` dict. We ate one silent
failure session on 2026-05-29 ~02:29 PDT (`cam_b5670a9f` SDK-left
event arrived, pi_relay re-published and LiveKit Cloud accepted, but
the SDK never re-fired `participant_connected`; the `RoomWorker`
walked a permanently-stale dict for hours). REST is consulted
directly here instead.

Per `RoomWorker` we run a periodic task that:

  1. Calls `lkapi.room.list_participants(...)` every `poll_interval_s`.
  2. Computes the set of `cam_*` identities present.
  3. For each identity in REST not in the worker's
     `_ingests_by_identity`, attempts to spawn one. Two cases:
       (a) participant exists in the SDK dict but `track_subscribed`
           did not fire for one or more publications: call
           `RemoteTrackPublication.set_subscribed(True)` to re-issue
           the subscribe RPC + drive the existing spawn path under
           `RoomWorker._spawn_lock`.
       (b) participant is missing from the SDK dict entirely (true
           desync): trigger `room.disconnect()` so the supervisor
           reconnects and the connect-time backfill picks up the
           authoritative participant list.
  4. For each identity in `_ingests_by_identity` not in REST,
     increment an absent-streak. After `miss_threshold` consecutive
     polls of absence, remove the ingest and write `online=false`.
  5. For each present-and-spawned identity, evaluate the two-signal
     liveness rule:
        online = (identity in REST) AND
                 (time.monotonic() - ingest._last_frame_at < liveness_grace_s)
     and write `cameras.online` accordingly. This loop is the
     steady-state writer for a CONNECTED camera and the backstop that
     re-derives `online` from REST when an SDK event was lost. It is
     NOT the only writer: the `participant_left` / `room_finished`
     webhooks write `online=false` directly (the authoritative leave
     signal — see `camera_presence.py`), because this loop is torn down
     with its RoomWorker the instant a camera leaves and could never
     observe the absence itself. All writes — here and there — funnel
     through `camera_presence.py`, the single owner of the column.

SDK event handlers remain idempotent fast-path: in the common case a
new publisher arrives → `track_subscribed` fires → an ingest is
spawned via `_spawn_lock` → next reconciler tick sees the ingest
already present and writes `online=true`. The reconciler exists to
catch the rare case where the SDK event was lost.

A webhook fast-path hint is supported via `signal_tick()`: webhook
handlers call it to collapse the next periodic wait into a ~1s
debounced reconcile, giving sub-second join/leave reaction.

Failure tolerance: every REST call and every DB write is wrapped in
try/except. Transient failures leave per-identity state untouched, so
a brief LiveKit-Cloud blip or DB hiccup does NOT flap `cameras.online`.
The `miss_threshold` debounce on absence and `_last_online_written`
no-op gate on writes together guarantee that even adversarial REST
sequences (single-tick absence, single-tick presence) produce no
spurious mutations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import UUID

from livekit import api as lkapi

from src.services.detection import camera_presence

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.services.detection.room_worker import RoomWorker

log = logging.getLogger(__name__)


# REST poll cadence. 10s preserves the prior reconciler-loop interval
# and matches the deleted `StaleOnlineSweeper` sweep period, keeping
# the externally-observable recovery latency unchanged.
_DEFAULT_POLL_INTERVAL_S: float = 10.0


# How many consecutive successful polls must omit an identity before
# the reconciler removes its ingest and writes `online=false`. 2
# polls × 10s ≈ 20s grace before eviction — generous enough that a
# brief DNS flap or LiveKit Cloud rebalance does not tear down the
# ingest, tight enough that a true disconnect surfaces in the UI in
# under half a minute.
_DEFAULT_MISS_THRESHOLD: int = 2


# How fresh the most recent frame arrival (`CameraIngest._last_frame_at`)
# must be for the reconciler to write `online=true`. 30s tracks the
# prior `StaleOnlineSweeper` threshold so the user-visible online flip
# behavior on publisher death is unchanged.
_DEFAULT_LIVENESS_GRACE_S: float = 30.0


# Minimum interval between webhook-triggered immediate ticks. Webhook
# bursts (e.g. wall phone + Pi joining in the same 200ms window)
# collapse to one reconcile pass.
_DEFAULT_WEBHOOK_DEBOUNCE_S: float = 1.0


# Hard ceiling on the LiveKit REST `list_participants` call inside one
# reconcile pass. Without this, a hung TCP socket (network blip, dead
# keepalive) parks the await indefinitely while holding `self._lock`,
# silently freezing the periodic loop. We saw this in production
# 2026-05-30: the loop fired once at startup, then sat parked for ~70
# minutes (matching the OS TCP keepalive timeout) before recovering for
# one more pass and stalling again. `cameras.online` stayed True
# silently while publishers had already dropped, so the live grid
# showed stale tiles for every camera until the backend was restarted.
# 5s is conservative: LiveKit's REST p99 is sub-second; anything over
# 5s is a degenerate network condition we want to abort and retry next
# tick.
_DEFAULT_REST_TIMEOUT_S: float = 5.0


@dataclass
class ReconcileResult:
    """Output of one reconcile pass. Used by tests to assert decisions
    without inspecting internal `RoomWorker` state, and emitted as a
    structured log line when any work was performed.
    """

    rest_identities: set[str] = field(default_factory=set)
    sdk_identities: set[str] = field(default_factory=set)
    spawned: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    # Map of identity -> bool we wrote in this pass. Absence means no
    # write happened (either already in that state, or no DB-row
    # available).
    online_writes: dict[str, bool] = field(default_factory=dict)


# The online/offline SQL the reconciler issues now lives in
# `camera_presence.py` (the single owner of the `cameras.online` column).
# This loop calls `camera_presence.set_camera_online_*` /
# `set_camera_offline_by_id` and keeps the no-op gating + fresh-start
# decision here. See `camera_presence.py` for the full trigger-point map.


class LiveKitRoomReconciler:
    """Per-`RoomWorker` task that reconciles in-memory ingest state
    against LiveKit Cloud's REST view of the room.

    Reads `RoomWorker._ingests_by_identity` and `RoomWorker._room`
    (the SDK room object) directly. Writes `cameras.online` via the
    `sessionmaker` and routes spawn/teardown through
    `RoomWorker._backfill_participant_tracks` and
    `RoomWorker.remove_ingest`. Holds no other state worth surviving
    a restart.
    """

    def __init__(
        self,
        *,
        room_name: str,
        room_worker: "RoomWorker",
        lk_api: "lkapi.LiveKitAPI | None",
        sessionmaker: "async_sessionmaker",
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
        miss_threshold: int = _DEFAULT_MISS_THRESHOLD,
        liveness_grace_s: float = _DEFAULT_LIVENESS_GRACE_S,
        webhook_debounce_s: float = _DEFAULT_WEBHOOK_DEBOUNCE_S,
    ) -> None:
        self.room_name = room_name
        self._room_worker = room_worker
        self._lk_api = lk_api
        self._sessionmaker = sessionmaker
        self._poll_interval_s = poll_interval_s
        self._miss_threshold = miss_threshold
        self._liveness_grace_s = liveness_grace_s
        self._webhook_debounce_s = webhook_debounce_s

        self._present_streak: dict[str, int] = {}
        self._absent_streak: dict[str, int] = {}
        # Per-identity last `online` value we successfully wrote. The
        # no-op gates inside `_write_online_*` use this so we issue a
        # DB UPDATE only when state actually changed. Also drives the
        # fresh-start state-machine reset on the first false -> true
        # transition per camera session.
        self._last_online_written: dict[str, bool] = {}
        # Identities we have already logged "no DB row" for. Suppresses
        # per-poll log noise when LiveKit Cloud holds a phantom cam_*
        # participant whose `cameras` row was deleted.
        self._logged_orphan_identities: set[str] = set()

        self._tick_event: asyncio.Event = asyncio.Event()
        # Serializes `signal_tick` against the periodic poll; both
        # serialize on a single critical section so the same identity
        # cannot be processed twice concurrently.
        self._lock: asyncio.Lock = asyncio.Lock()
        self._last_signal_at: float = 0.0
        self._loop_task: asyncio.Task[None] | None = None
        self._stopped: bool = False

    async def start(self) -> None:
        """Begin the periodic reconcile loop. Idempotent; a second
        `start()` while the first is running is a no-op."""
        if self._loop_task is not None:
            return
        self._stopped = False
        self._loop_task = asyncio.create_task(
            self._loop(),
            name=f"room-reconciler-{self.room_name}",
        )

    async def stop(self) -> None:
        """Cancel the loop and await it. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        task = self._loop_task
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            self._loop_task = None

    async def signal_tick(self) -> None:
        """Fast-path hint from the webhook handler: do a reconcile
        pass soon, instead of waiting for the next periodic tick.

        Debounced by `_webhook_debounce_s` so a burst of webhooks for
        the same room (e.g. wall phone + Pi joining together) collapse
        to a single extra pass. Fire-and-forget — does not block on
        the resulting reconcile.
        """
        now = time.monotonic()
        if now - self._last_signal_at < self._webhook_debounce_s:
            return
        self._last_signal_at = now
        self._tick_event.set()

    async def reconcile_once(self) -> ReconcileResult:
        """Single reconcile pass. Public so tests can drive a pass
        without starting the loop. Held under `_lock` so a webhook
        signal racing the periodic tick serializes safely.
        """
        async with self._lock:
            return await self._reconcile_locked()

    # ── loop ───────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Periodic + webhook-signaled reconcile loop."""
        while not self._stopped:
            try:
                await asyncio.wait_for(
                    self._tick_event.wait(),
                    timeout=self._poll_interval_s,
                )
            except asyncio.TimeoutError:
                # Normal periodic tick.
                pass
            except asyncio.CancelledError:
                return
            self._tick_event.clear()
            if self._stopped:
                return
            try:
                await self.reconcile_once()
            except asyncio.CancelledError:
                return
            except Exception:
                log.exception(
                    "room_reconciler.loop_iteration_failed room=%s",
                    self.room_name,
                )

    # ── one pass ───────────────────────────────────────────────────

    async def _reconcile_locked(self) -> ReconcileResult:
        result = ReconcileResult()
        rest_identities = await self._fetch_rest_identities()
        if rest_identities is None:
            # REST call failed; leave per-identity state untouched.
            # No `cameras.online` write, no spawn, no eviction.
            return result
        result.rest_identities = rest_identities
        result.sdk_identities = set(self._room_worker._ingests_by_identity.keys())

        # 1. Process REST-present identities.
        for identity in rest_identities:
            try:
                await self._handle_present_identity(identity, result)
            except Exception:
                log.exception(
                    "room_reconciler.present_identity_failed room=%s identity=%s",
                    self.room_name, identity,
                )

        # 2. Process previously-tracked identities now absent from
        # REST. Increment streak; remove on threshold reached.
        for identity in list(self._room_worker._ingests_by_identity.keys()):
            if identity in rest_identities:
                # Present; debounce counter resets in the present-
                # handling branch above. Just clear any lingering
                # absent streak here as a belt-and-braces.
                self._absent_streak.pop(identity, None)
                continue
            self._absent_streak[identity] = self._absent_streak.get(identity, 0) + 1
            if self._absent_streak[identity] >= self._miss_threshold:
                try:
                    await self._handle_absent_identity(identity, result)
                except Exception:
                    log.exception(
                        "room_reconciler.absent_identity_failed room=%s identity=%s",
                        self.room_name, identity,
                    )

        # 3. Drop present-streak bookkeeping for identities that
        # vanished without ever spawning an ingest (rare; primarily
        # an orphan-identity case).
        for identity in list(self._present_streak.keys()):
            if identity not in rest_identities:
                self._present_streak.pop(identity, None)

        if result.spawned or result.removed or result.online_writes:
            log.info(
                "room_reconciler.pass room=%s rest=%d sdk=%d "
                "spawned=%s removed=%s online_writes=%s",
                self.room_name, len(result.rest_identities),
                len(result.sdk_identities), result.spawned,
                result.removed, result.online_writes,
            )
        return result

    async def _fetch_rest_identities(self) -> set[str] | None:
        """Call `list_participants`. Returns the set of `cam_*` identity
        strings on success, or None on any failure. None means "leave
        state untouched."
        """
        if self._lk_api is None:
            # Tests / cold-start with no credentials configured.
            log.debug(
                "room_reconciler.no_lkapi room=%s — skipping poll",
                self.room_name,
            )
            return None
        try:
            resp = await asyncio.wait_for(
                self._lk_api.room.list_participants(
                    lkapi.ListParticipantsRequest(room=self.room_name),
                ),
                timeout=_DEFAULT_REST_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            log.warning(
                "room_reconciler.list_participants_timeout room=%s after=%.1fs",
                self.room_name,
                _DEFAULT_REST_TIMEOUT_S,
            )
            return None
        except Exception:
            log.exception(
                "room_reconciler.list_participants_failed room=%s",
                self.room_name,
            )
            return None
        participants = getattr(resp, "participants", None)
        if participants is None:
            log.warning(
                "room_reconciler.list_participants_malformed room=%s",
                self.room_name,
            )
            return None
        result: set[str] = set()
        for participant in participants:
            identity = getattr(participant, "identity", None)
            if isinstance(identity, str) and identity.startswith("cam_"):
                result.add(identity)
        return result

    # ── per-identity handlers ──────────────────────────────────────

    async def _handle_present_identity(
        self, identity: str, result: ReconcileResult,
    ) -> None:
        """The identity is in LiveKit. Ensure we have an ingest, then
        write `online` based on the two-signal rule.
        """
        self._present_streak[identity] = self._present_streak.get(identity, 0) + 1
        # Reset any lingering absent streak (e.g. identity flapped).
        self._absent_streak.pop(identity, None)

        ingest = self._room_worker._ingests_by_identity.get(identity)
        if ingest is None:
            spawned = await self._try_spawn_ingest(identity)
            if spawned:
                result.spawned.append(identity)
            ingest = self._room_worker._ingests_by_identity.get(identity)
        if ingest is None:
            # Spawn was not possible this pass. Either the cameras row
            # is missing (orphan; log once), the SDK is mid-reconnect
            # (next reconnect's backfill will catch it), or
            # `set_subscribed` did not yet drive `_handle_track_subscribed`
            # (next poll will retry).
            if identity not in self._logged_orphan_identities:
                self._logged_orphan_identities.add(identity)
                log.warning(
                    "room_reconciler.identity_unspawnable room=%s identity=%s",
                    self.room_name, identity,
                )
            return

        camera_id = ingest.camera_id
        if self._frames_are_fresh(ingest):
            await self._write_online_true(identity, camera_id, result)
        else:
            await self._write_online_false(identity, camera_id, result)

    async def _try_spawn_ingest(self, identity: str) -> bool:
        """Best-effort spawn when REST sees an identity the SDK did
        not surface to us via `track_subscribed`.

        Returns True if an ingest is now present in the worker's dict.
        """
        room = self._room_worker._room
        if room is None:
            # Supervisor is between connect attempts. The next
            # reconnect's connect-time backfill will pick up the
            # identity. Skip this pass.
            log.debug(
                "room_reconciler.spawn_skip_no_room room=%s identity=%s",
                self.room_name, identity,
            )
            return False

        participant = room.remote_participants.get(identity)
        if participant is None:
            # True SDK desync: REST sees the participant, the SDK's
            # `remote_participants` dict does not. The only recovery
            # is to force the supervisor to reconnect — its connect-
            # time backfill pulls the fresh participant list from the
            # server and drives the spawn path. Heavy hammer; logged
            # at WARNING.
            log.warning(
                "room_reconciler.sdk_desync_forcing_reconnect "
                "room=%s identity=%s",
                self.room_name, identity,
            )
            try:
                await room.disconnect()
            except Exception:
                log.exception(
                    "room_reconciler.force_disconnect_failed room=%s",
                    self.room_name,
                )
            return False

        # Participant IS in the SDK dict, but `track_subscribed` did
        # not fire for one or more publications. Walk publications
        # via the same handler the SDK event would have called
        # (idempotent under `_spawn_lock`), then issue
        # `set_subscribed(True)` on every publication so the SDK re-
        # delivers tracks for any still-unsubscribed ones.
        publications = list(participant.track_publications.values())
        log.info(
            "room_reconciler.resubscribing room=%s identity=%s publications=%d",
            self.room_name, identity, len(publications),
        )
        try:
            await self._room_worker._backfill_participant_tracks(participant)
        except Exception:
            log.exception(
                "room_reconciler.backfill_failed room=%s identity=%s",
                self.room_name, identity,
            )
        for publication in publications:
            try:
                publication.set_subscribed(True)
            except Exception:
                log.exception(
                    "room_reconciler.set_subscribed_failed "
                    "room=%s identity=%s",
                    self.room_name, identity,
                )
        return identity in self._room_worker._ingests_by_identity

    async def _handle_absent_identity(
        self, identity: str, result: ReconcileResult,
    ) -> None:
        """Identity has been absent for `miss_threshold` polls. Stop
        the ingest and flip `online=false`.
        """
        ingest = self._room_worker._ingests_by_identity.get(identity)
        if ingest is None:
            # Already gone (raced with another teardown path). Clear
            # our bookkeeping and move on.
            self._absent_streak.pop(identity, None)
            self._present_streak.pop(identity, None)
            self._last_online_written.pop(identity, None)
            return
        camera_id = ingest.camera_id
        try:
            await self._room_worker.remove_ingest(identity)
        except Exception:
            log.exception(
                "room_reconciler.remove_ingest_failed room=%s identity=%s",
                self.room_name, identity,
            )
        result.removed.append(identity)
        self._absent_streak.pop(identity, None)
        self._present_streak.pop(identity, None)
        await self._write_online_false(identity, camera_id, result)
        # Drop our memory of having written `true` so a future re-spawn
        # of the same identity triggers the fresh-start state reset.
        self._last_online_written.pop(identity, None)

    def _frames_are_fresh(self, ingest) -> bool:  # noqa: ANN001
        """Two-signal liveness component: is the most recent frame
        within `_liveness_grace_s`? Reads `_last_frame_at` (set by
        `CameraIngest.record_frame_arrival` from the video consumer
        callback) directly — same process, no locking required."""
        # UNLESS WE STOPPED ASKING. A paused camera has its subscription
        # severed to stop paying LiveKit for video nobody will look at,
        # so no frames arrive BY OUR OWN CHOICE. The two-signal rule
        # exists to stop "joined but not yet publishing" reading as
        # online; it says nothing about a publisher we deliberately
        # stopped listening to, and applying it here would flip a
        # perfectly healthy camera to offline in the app about thirty
        # seconds after a caregiver pressed pause. REST presence — the
        # publisher IS in the room — is the whole truth in that case.
        if getattr(ingest, "subscription_severed", False):
            return True
        last_frame_at = getattr(ingest, "_last_frame_at", 0.0)
        if last_frame_at <= 0.0:
            # No frame has ever been pushed; cannot be considered live.
            return False
        return (time.monotonic() - last_frame_at) < self._liveness_grace_s

    # ── DB writes ──────────────────────────────────────────────────

    async def _write_online_true(
        self, identity: str, camera_id: UUID, result: ReconcileResult,
    ) -> None:
        # `prev` only chooses fresh-start (state reset) vs keep-state; it NO
        # LONGER gates whether we write. Both writes are DB-guarded (WHERE
        # online = false), so re-issuing every pass is a no-op once the row is
        # online (no Realtime echo) and SELF-HEALS a row that drifted false
        # under us — a gateway publisher never sends participant_left, so the
        # flag could stick false while frames flow. Trusting the in-memory
        # cache to skip the write was the "stuck offline" bug; the DB is the
        # authority now, and rowcount tells us whether it actually flipped.
        prev = self._last_online_written.get(identity)
        fresh_start = prev is None or prev is False
        try:
            now = datetime.now(timezone.utc)
            async with self._sessionmaker() as session:
                if fresh_start:
                    flipped = await camera_presence.set_camera_online_fresh_start(
                        session, camera_id, now,
                    )
                else:
                    flipped = await camera_presence.set_camera_online_keep_state(
                        session, camera_id,
                    )
                await session.commit()
        except Exception:
            log.exception(
                "room_reconciler.online_true_write_failed "
                "room=%s camera=%s",
                self.room_name, camera_id,
            )
            return
        self._last_online_written[identity] = True
        if flipped:
            result.online_writes[identity] = True

    async def _write_online_false(
        self, identity: str, camera_id: UUID, result: ReconcileResult,
    ) -> None:
        # Guarded (WHERE online = true) → a DB no-op when already offline (no
        # echo) and self-healing if the row drifted true. No in-memory
        # short-circuit, so a cache that thinks we're already offline can't
        # suppress a genuinely-needed write.
        try:
            async with self._sessionmaker() as session:
                flipped = await camera_presence.set_camera_offline_by_id(
                    session, camera_id,
                )
                await session.commit()
        except Exception:
            log.exception(
                "room_reconciler.online_false_write_failed "
                "room=%s camera=%s",
                self.room_name, camera_id,
            )
            return
        self._last_online_written[identity] = False
        if flipped:
            result.online_writes[identity] = False
