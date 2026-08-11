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
        return {
            "blind": blind, "overdue": overdue,
            "nodes_down": len(down), "nodes_live": live_nodes,
        }

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
