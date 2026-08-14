"""Dead-man watch over the GPU node fleet.

WHY THIS EXISTS
---------------
Every failure in the identity path is designed to be quiet. The backend
FAILS OPEN when a node is unreachable, so safety detectors keep firing and
the product looks alive; what actually stops is the patient-scoped routine
work — medication, drinking, TV, sleep and activity minutes — which "fail
quiet, skip with patient_elsewhere and do not retry".

That is the right runtime behaviour and a terrible operational one. The
2026-07-26 identity OOM went unnoticed for hours for exactly this reason,
and the roster does not change it: it makes recovery automatic when there
is somewhere to recover TO, and completely silent when there is not.

So this watches the two states that mean a home is actually blind, and
pages a human. It reads the same two tables the roster writes and holds no
state of its own beyond which keys it has already paged.

WHAT IT PAGES ON
----------------
1. UNASSIGNED HOMES (critical). A patient with no node is receiving no
   identity at all. This is the one that matters — it is the end state of
   every other failure, and the only one whose blast radius is a specific
   family. Given a grace period, because a home is briefly unassigned
   during every normal failover.

2. A NODE THAT STOPPED HEARTBEATING (warning, not critical). If there is
   spare capacity the roster re-places its homes within ~a poll and (1)
   never fires — the system did its job, and paging critical for a working
   failover is how a pager gets ignored. It is still worth telling someone,
   because capacity that vanished silently is capacity nobody replaces.

Both auto-resolve, so a failover that heals closes its own incident.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from sqlalchemy import text

from src.services import pager

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from src.settings import Settings

log = logging.getLogger(__name__)

# How long a home may sit unassigned before it is a page rather than a
# failover in progress. A node claims on its next poll (30s), so this is
# several polls of slack — long enough that a routine reshuffle is silent,
# short enough that a genuinely orphaned home is reported while the person
# who caused it is still at their desk.
UNASSIGNED_GRACE_S = 300

# Prefixes keep this subsystem's dedup keys out of the event gate's, which
# uses bare event uuids. A collision would collapse two incidents into one
# and settling either would hide the other.
_KEY_UNASSIGNED = "identity-fleet:unassigned-homes"
_KEY_NODE_DOWN = "identity-fleet:node-down:"
_KEY_LAG = "identity-fleet:lag:"

# Consecutive cycles a camera must exceed the tick length before the lag
# page fires. One bad cycle is a re-ground or a batch spike; three in a
# row (~1.5-3 min at the default watch interval) is a node past its
# rated envelope — the exact production tripwire the capacity rating
# leans on (docs/IDENTITY-GPU-ARCHITECTURE.md: SLA = lag under the tick,
# rated 4 occupied cams/4090, surge 6-8, gone by 10).
LAG_ALARM_CYCLES = 3


class FleetWatch:
    """Polls the roster tables and pages when homes are actually blind."""

    def __init__(
        self, *, settings: "Settings", sessionmaker: "async_sessionmaker",
    ) -> None:
        self._settings = settings
        self._sessionmaker = sessionmaker
        self._task: asyncio.Task[None] | None = None
        # Keys currently believed to be OPEN at PagerDuty. Purely an
        # optimisation to avoid re-posting an identical trigger every cycle;
        # PagerDuty dedups on the key anyway, so being wrong here costs a
        # redundant request, never a missed or duplicated page.
        self._open: set[str] = set()
        # Per-camera consecutive over-tick-lag cycle counts (lag alarm).
        self._lag_streak: dict[str, int] = {}
        # Set False after the first missing-table error so a not-yet-
        # applied occupancy delta costs one log line, not one per cycle.
        self._occupancy_ok = True
        self._http: "object | None" = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="identity-fleet-watch")
        log.info(
            "identity.fleet_watch_started interval=%ds paging=%s",
            self._settings.identity_fleet_watch_seconds,
            pager.is_configured(self._settings),
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

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._settings.identity_fleet_watch_seconds)
            try:
                await self.check()
            except Exception:  # noqa: BLE001 — the watchdog must never die
                log.exception("identity.fleet_watch_failed")

    async def check(self) -> dict[str, int]:
        """One pass. Returns counts, so a test can assert without paging."""
        stale_s = self._settings.identity_roster_stale_after_s
        async with self._sessionmaker() as session:
            # Homes with no node, or whose node has gone quiet. Both are
            # "nobody is watching this family"; the second is included
            # because the roster has not yet got round to releasing it and
            # the home is just as blind in the meantime.
            blind = (await session.execute(text(
                "SELECT count(*) FROM public.patients p "
                " WHERE p.identity_node_id IS NULL "
                "    OR NOT EXISTS ("
                "         SELECT 1 FROM public.identity_nodes n "
                "          WHERE n.node_id = p.identity_node_id "
                "            AND n.last_seen_at > now() - make_interval("
                "                  secs => :stale))"
            ), {"stale": stale_s})).scalar() or 0

            # ...and of those, the ones that have been blind long enough to
            # be a real problem rather than a failover mid-flight. Measured
            # from the node's last heartbeat where there is one, and from
            # the patient's creation where there is not — a home that has
            # NEVER been claimed is the same failure as one that lost its
            # node, and is otherwise invisible because it has no node row
            # to have aged.
            overdue = (await session.execute(text(
                "SELECT count(*) FROM public.patients p "
                " LEFT JOIN public.identity_nodes n "
                "        ON n.node_id = p.identity_node_id "
                " WHERE (p.identity_node_id IS NULL "
                "        OR n.last_seen_at <= now() - make_interval("
                "             secs => :stale)) "
                "   AND COALESCE(n.last_seen_at, p.created_at) "
                "       < now() - make_interval(secs => :grace)"
            ), {"stale": stale_s, "grace": UNASSIGNED_GRACE_S})).scalar() or 0

            # Nodes that were serving homes and have stopped polling.
            down = (await session.execute(text(
                "SELECT n.node_id, count(p.id), "
                "       EXTRACT(EPOCH FROM now() - n.last_seen_at)::int "
                "  FROM public.identity_nodes n "
                "  LEFT JOIN public.patients p ON p.identity_node_id = n.node_id "
                " WHERE n.last_seen_at <= now() - make_interval(secs => :stale) "
                " GROUP BY n.node_id, n.last_seen_at"
            ), {"stale": stale_s})).all()

            live_nodes = (await session.execute(text(
                "SELECT count(*) FROM public.identity_nodes "
                " WHERE last_seen_at > now() - make_interval(secs => :stale)"
            ), {"stale": stale_s})).scalar() or 0

        await self._page_unassigned(overdue, blind, live_nodes)
        await self._page_nodes_down(down)
        serving = await self._check_serving()
        return {
            "blind": blind, "overdue": overdue,
            "nodes_down": len(down), "nodes_live": live_nodes,
            **serving,
        }

    # ── serving quality: lag alarm + occupancy sampling ──────────────
    # The roster tables say a home HAS a node; only the node's own
    # /state says it is being SERVED — and how well. One poll per home
    # per cycle yields both the SLA tripwire (lag_s vs the tick length)
    # and the occupancy duty-cycle samples that turn homes-per-node
    # capacity planning into measurement (2026-08-14).

    async def _fetch_state(self, url: str) -> "dict | None":
        """GET one identity /state. Separate method so tests stub it."""
        import httpx

        if self._http is None:
            token = getattr(self._settings, "identity_state_token", "")
            self._http = httpx.AsyncClient(
                timeout=5.0,
                headers=(
                    {"Authorization": f"Bearer {token}"} if token else {}
                ),
            )
        try:
            resp = await self._http.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception:  # noqa: BLE001 — an unreachable node is the
            return None    # node-down page's job, not an exception here

    async def _check_serving(self) -> dict[str, int]:
        stale_s = self._settings.identity_roster_stale_after_s
        tick_s = float(self._settings.livekit_worker_tick_seconds)
        async with self._sessionmaker() as session:
            served = (await session.execute(text(
                "SELECT p.id::text, n.node_id, n.base_url, p.identity_port "
                "  FROM public.patients p "
                "  JOIN public.identity_nodes n "
                "    ON n.node_id = p.identity_node_id "
                " WHERE p.identity_port IS NOT NULL "
                "   AND n.last_seen_at > now() - make_interval(secs => :stale)"
            ), {"stale": stale_s})).all()

        lagging = 0
        per_node: dict[str, list[int]] = {}
        seen_cams: set[str] = set()
        for _pid, node_id, base_url, port in served:
            state = await self._fetch_state(f"{base_url}:{port}/state")
            if state is None:
                # Unreachable is the node-down page's territory; a zero
                # sample here would read as "empty node", which is a
                # different fact than "no data".
                continue
            cameras = state.get("cameras") or {}
            totals = per_node.setdefault(node_id, [0, 0])
            for cam_id, cam in cameras.items():
                totals[0] += 1
                if cam.get("tracks"):
                    totals[1] += 1
                seen_cams.add(cam_id)
                lag = cam.get("lag_s")
                over = isinstance(lag, (int, float)) and lag > tick_s
                # Only count lag against OCCUPIED cameras: an empty
                # room legitimately stops batching and its last lag_s
                # is a frozen snapshot, not a serving measurement
                # (2026-08-13 watchdog false-alarm lesson).
                if over and cam.get("tracks"):
                    self._lag_streak[cam_id] = (
                        self._lag_streak.get(cam_id, 0) + 1
                    )
                    if self._lag_streak[cam_id] == LAG_ALARM_CYCLES:
                        lagging += 1
                        log.error(
                            "identity.lag_over_tick cam=%s lag=%.1fs "
                            "tick=%.0fs node=%s — SLA breach sustained "
                            "%d cycles", cam_id, lag, tick_s, node_id,
                            LAG_ALARM_CYCLES,
                        )
                        await self._trigger(
                            f"{_KEY_LAG}{cam_id}",
                            summary=(
                                f"identity lag {lag:.1f}s > tick "
                                f"{tick_s:.0f}s on camera {cam_id} "
                                f"(node {node_id})"
                            ),
                            severity="warning",
                            details={
                                "camera": cam_id, "node": node_id,
                                "lag_s": lag, "tick_s": tick_s,
                                "remedy": (
                                    "node past its rated occupied-camera "
                                    "envelope — add a node or move homes "
                                    "(docs/IDENTITY-GPU-ARCHITECTURE.md)"
                                ),
                            },
                        )
                else:
                    if self._lag_streak.pop(cam_id, None):
                        await self._resolve(f"{_KEY_LAG}{cam_id}")
        # Cameras that vanished entirely drop their streaks + pages.
        for cam_id in [c for c in self._lag_streak if c not in seen_cams]:
            self._lag_streak.pop(cam_id, None)
            await self._resolve(f"{_KEY_LAG}{cam_id}")

        await self._record_occupancy(per_node)
        return {
            "cameras_polled": sum(v[0] for v in per_node.values()),
            "cameras_occupied": sum(v[1] for v in per_node.values()),
            "cameras_lagging": lagging,
        }

    async def _record_occupancy(
        self, per_node: "dict[str, list[int]]",
    ) -> None:
        if not self._occupancy_ok or not per_node:
            return
        try:
            async with self._sessionmaker() as session:
                for node_id, (total, occupied) in per_node.items():
                    await session.execute(text(
                        "INSERT INTO public.identity_occupancy_samples "
                        "  (node_id, total_cameras, occupied_cameras) "
                        "VALUES (:n, :t, :o)"
                    ), {"n": node_id, "t": total, "o": occupied})
                await session.commit()
        except Exception:  # noqa: BLE001
            # Most likely the delta is not applied yet (prepared-only in
            # prod by decision, 2026-08-14). Sampling is an observability
            # nicety — disable quietly rather than log-spam every cycle.
            self._occupancy_ok = False
            log.warning(
                "identity.occupancy_sampling_disabled — table missing? "
                "apply scripts/apply_identity_fleet_ops_delta.sql",
                exc_info=True,
            )

    async def _page_unassigned(
        self, overdue: int, blind: int, live_nodes: int,
    ) -> None:
        if overdue > 0:
            log.error(
                "identity.homes_blind overdue=%d blind=%d live_nodes=%d — "
                "these homes are receiving NO identity",
                overdue, blind, live_nodes,
            )
            await self._trigger(
                _KEY_UNASSIGNED,
                summary=(
                    f"{overdue} home(s) have no identity node "
                    f"({live_nodes} node(s) alive)"
                ),
                severity="critical",
                details={
                    "homes_overdue": overdue,
                    "homes_blind_now": blind,
                    "nodes_alive": live_nodes,
                    "grace_s": UNASSIGNED_GRACE_S,
                    "remedy": (
                        "no node has spare capacity — release another with "
                        "scripts/pod.sh release <tag>, or raise "
                        "NEMARA_NODE_CAPACITY on an existing one"
                    ),
                },
            )
        else:
            await self._resolve(_KEY_UNASSIGNED)

    async def _page_nodes_down(self, down: list) -> None:
        seen: set[str] = set()
        for node_id, homes, age_s in down:
            # A stale node holding NO homes is a corpse, not an incident
            # — replaced test/fleet nodes leave rows behind until the
            # reaper runs, and paging each one buried real alerts in
            # noise (2026-08-14: 11 corpse pages in an afternoon of
            # node replacement, node_id different every time). Not
            # added to `seen`, so any open page for it auto-resolves.
            if not homes:
                log.info(
                    "identity.node_gone node=%s (0 homes) — no page",
                    node_id,
                )
                continue
            key = f"{_KEY_NODE_DOWN}{node_id}"
            seen.add(key)
            log.warning(
                "identity.node_down node=%s homes_held=%d last_seen=%ss ago",
                node_id, homes, age_s,
            )
            await self._trigger(
                key,
                summary=(
                    f"GPU node {node_id} stopped heartbeating "
                    f"({age_s}s ago, {homes} home(s) held)"
                ),
                # Warning, not critical: with spare capacity the roster
                # re-places these homes and nobody needed to wake up. The
                # critical page is the one above, which fires only if that
                # re-placement does not happen.
                severity="warning",
                details={"node_id": node_id, "homes_held": homes,
                         "last_seen_s_ago": age_s},
            )
        # A node that came back resolves itself.
        for key in [k for k in self._open if k.startswith(_KEY_NODE_DOWN)]:
            if key not in seen:
                await self._resolve(key)

    async def _trigger(self, key: str, **kw) -> None:
        if key in self._open:
            return
        ok = await pager.trigger_keyed(
            settings=self._settings, dedup_key=key,
            source="memoria-identity-fleet", **kw,
        )
        if ok:
            self._open.add(key)

    async def _resolve(self, key: str) -> None:
        if key not in self._open:
            return
        await pager.resolve_keyed(settings=self._settings, dedup_key=key)
        self._open.discard(key)
