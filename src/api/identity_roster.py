"""GPU node roster: POST /v1/identity/roster.

A GPU node boots knowing only its own id and how many homes it can serve.
It polls here and is told which rooms to run. That single call is also its
heartbeat, so the same mechanism gives us four things:

  discovery   the backend joins patients -> identity_nodes.base_url
  scale up    start another node; it drains the unassigned queue on first poll
  failover    a node that stops polling has its homes released to whoever
              polls next
  rebalance   null out patients.identity_node_id and it is re-placed

Same shape as `GET /v1/gateway/roster`, which already does "fleet member
polls, backend lazily assigns, assignment lives in Postgres rather than in
config on N machines" in production for the gateway boxes.

POST rather than GET because this genuinely mutates — it claims homes and
stamps liveness. (The gateway roster is a GET that also mutates; that is a
wart worth not copying.)
"""

from __future__ import annotations

import logging
import secrets
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from src.api.deps import SessionDep, SettingsDep

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/identity", tags=["identity"])


class RosterRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=128)
    # Where the backend should reach this node's identity processes. In
    # production a tailnet MagicDNS name, precisely so it survives RunPod
    # reassigning the pod's direct-TCP endpoint on every restart.
    #
    # SCHEME AND HOST ONLY — no port, no path. Callers append `:<port>`
    # (IdentityRouter, and the /ready fleet probe), so a value carrying
    # either produces `http://host:8100:8012` or `http://host/x:8012`. Both
    # are silent: the row stores fine, and the breakage surfaces later as an
    # unreachable node whose homes are quietly identity-less.
    base_url: str = Field(min_length=1, max_length=512)
    # How many rooms this node will accept. Node-local because it depends on
    # NEMARA_PERCEPTION_CONCURRENCY and the GPU.
    capacity: int = Field(ge=1, le=64)

    @field_validator("base_url")
    @classmethod
    def _addressable(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("base_url must start with http:// or https://")
        if not parsed.hostname:
            raise ValueError("base_url must include a host")
        if parsed.path or parsed.query or parsed.fragment:
            raise ValueError("base_url must be scheme://host with no path")
        try:
            # Accessing .port is what raises on a malformed one; the check is
            # that it is ABSENT, since the per-room port is appended later.
            if parsed.port is not None:
                raise ValueError(
                    "base_url must not include a port — the roster assigns "
                    "one port per room and appends it"
                )
        except ValueError as exc:  # covers both the raise above and a bad port
            raise ValueError(str(exc)) from exc
        return v


class RosterRoom(BaseModel):
    # `patient_<uuid>` — the LiveKit room the identity process joins.
    room: str
    # Which port to listen on. Assigned by the backend, not chosen by the
    # node: the backend has to be able to REACH this process, and it cannot
    # do that if the node picks unilaterally.
    port: int


class RosterResponse(BaseModel):
    node_id: str
    # Rooms this node should be running. Authoritative: the node reconciles
    # to exactly this list, starting what is new and stopping what is gone.
    rooms: list[RosterRoom]
    capacity: int
    # How many homes this poll took off other (stale) nodes. Non-zero means a
    # failover just happened and is worth an alert.
    reclaimed: int


# One identity process per room, one port each. 100 slots is far above any
# node's plausible capacity (~4 at concurrency=1, ~8 at 2).
PORT_MIN = 8012
PORT_MAX = 8111

# A home is LIVE when it has at least one camera actually streaming.
# Slots follow live homes: claim order prefers them, and a dormant home
# yields its slot to one (2026-08-19 — prod had 7 of 8 slots held by
# May/June test homes with no cameras online while every home that was
# actually streaming sat unassigned and identity-blind).
_LIVE_CAM = (
    "EXISTS (SELECT 1 FROM public.cameras c "
    "         WHERE c.patient_id = p.id AND c.online AND NOT c.disabled)"
)


@router.post("/roster", response_model=RosterResponse)
async def post_roster(
    body: RosterRequest,
    session: SessionDep,
    settings: SettingsDep,
    x_internal_key: str = Header(default=""),
) -> RosterResponse:
    """Heartbeat, release stale nodes' homes, and claim up to capacity.

    Fails CLOSED exactly like the other /internal-style routes: an unset key
    outside local/test rejects every call rather than accepting every call.
    """
    expected = settings.internal_service_key
    if expected:
        if not secrets.compare_digest(x_internal_key, expected):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="invalid_internal_key",
            )
    elif settings.env not in ("local", "test"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="internal_key_not_configured",
        )

    stale_s = settings.identity_roster_stale_after_s

    # 1. Upsert + heartbeat. base_url and capacity are refreshed every poll so
    #    a node that moves (new tailnet name) or is resized (concurrency bump)
    #    is picked up without anyone editing a row by hand.
    await session.execute(
        text(
            "INSERT INTO public.identity_nodes "
            "  (node_id, base_url, capacity, last_seen_at) "
            "VALUES (:n, :u, :c, now()) "
            "ON CONFLICT (node_id) DO UPDATE SET "
            "  base_url = excluded.base_url, "
            "  capacity = excluded.capacity, "
            "  last_seen_at = now()"
        ),
        {"n": body.node_id, "u": body.base_url, "c": body.capacity},
    )

    # 2. Release homes held by nodes that have stopped heartbeating. Done on
    #    every poll by every node, so recovery needs no cron and no operator:
    #    the surviving nodes notice.
    #
    #    Only NULLs the assignment; the node row is left alone so a node that
    #    comes back (container restart, brief network partition) keeps its
    #    identity and its capacity, and simply re-claims.
    reclaimed = (
        await session.execute(
            text(
                # identity_port is cleared with the assignment — it belongs to
                # a node, not to the home, and must return to that node's pool.
                "UPDATE public.patients "
                "   SET identity_node_id = NULL, identity_port = NULL "
                "WHERE identity_node_id IS NOT NULL "
                "  AND identity_node_id <> :self "
                "  AND identity_node_id IN ("
                "    SELECT node_id FROM public.identity_nodes "
                "     WHERE last_seen_at < now() - make_interval(secs => :stale)"
                "  )"
            ),
            {"self": body.node_id, "stale": stale_s},
        )
    ).rowcount or 0

    # 2b. Reap corpse rows. A briefly-absent node keeps its row (see the
    #     comment above) — but a node dark for DAYS is a terminated pod,
    #     never a partition, and its row is registry clutter (21 of them
    #     had accumulated by 2026-08-17, purged by hand; this makes that
    #     hand-purge unnecessary forever). Holding-no-homes is implied:
    #     step 2 released any homes long before the corpse window ends.
    await session.execute(
        text(
            "DELETE FROM public.identity_nodes "
            " WHERE node_id <> :self "
            "   AND last_seen_at < now() - interval '7 days' "
            "   AND node_id NOT IN ("
            "     SELECT identity_node_id FROM public.patients "
            "      WHERE identity_node_id IS NOT NULL"
            "   )"
        ),
        {"self": body.node_id},
    )

    # 2c. Make room for LIVE homes by evicting DORMANT ones (2026-08-19).
    #     Claim priority alone cannot help a home that arrives after the
    #     slots are gone: nothing ever released an assignment on a live
    #     node, so June's test homes held every slot indefinitely.
    #
    #     NOT a timer. An idle home costs ~zero GPU (the empty-room
    #     economy), so evicting on a schedule would only buy churn — each
    #     re-claim costs a process start and a re-ground. A dormant home
    #     keeps its slot as warm standby until a LIVE home actually needs
    #     it, and never longer. A home with a live camera is never evicted.
    evicted = (
        await session.execute(
            text(
                "WITH need AS ("
                "  SELECT count(*) AS n FROM public.patients p "
                "   WHERE p.identity_node_id IS NULL AND " + _LIVE_CAM +
                "), spare AS ("
                "  SELECT :cap - count(*) AS n FROM public.patients "
                "   WHERE identity_node_id = :node "
                "), victims AS ("
                "  SELECT p.id FROM public.patients p "
                "   WHERE p.identity_node_id = :node AND NOT " + _LIVE_CAM +
                # Oldest first: the longest-settled dormant homes are the
                # likeliest to be permanently dormant.
                "   ORDER BY p.created_at "
                "   LIMIT GREATEST("
                "     (SELECT n FROM need) - GREATEST((SELECT n FROM spare), 0), 0) "
                "   FOR UPDATE SKIP LOCKED"
                ") UPDATE public.patients "
                "     SET identity_node_id = NULL, identity_port = NULL "
                "   WHERE id IN (SELECT id FROM victims)"
            ),
            {"node": body.node_id, "cap": body.capacity},
        )
    ).rowcount or 0
    if evicted:
        log.info(
            "identity.roster evicted=%d dormant home(s) on %s to seat live homes",
            evicted, body.node_id,
        )

    # 3. Claim up to remaining capacity from the unassigned pool.
    #
    #    FOR UPDATE SKIP LOCKED is what makes two nodes polling at the same
    #    instant safe: each takes a disjoint set instead of both reading the
    #    same rows and one overwriting the other.
    #
    #    Oldest-first is deliberate — a home that has been unassigned longest
    #    has been blind longest.
    #
    #    NOTE this only ever claims UNASSIGNED homes. It never takes a home
    #    off a live node to even out load, because a move costs the new node a
    #    re-ground and ~30-60s where patient-scoped routine detectors are
    #    silently skipped. Failure is worth that; tidiness is not.
    #    Each claimed home also gets a PORT — the lowest currently free one on
    #    this node. row_number() over the free-port series pairs the Nth
    #    claimed home with the Nth free port in a single statement, so two
    #    concurrent pollers cannot both be handed the same one (the partial
    #    unique index is the backstop if they somehow race).
    claimed = (
        await session.execute(
            text(
                "WITH avail AS ("
                "  SELECT :cap - count(*) AS n FROM public.patients "
                "   WHERE identity_node_id = :node"
                # The lock and the numbering must be separate CTEs: Postgres
                # rejects FOR UPDATE in a query that also has a window
                # function ("FOR UPDATE is not allowed with window
                # functions"). Lock first, number the locked set after.
                "), locked AS ("
                "  SELECT id, created_at FROM public.patients p "
                "   WHERE identity_node_id IS NULL "
                # LIVE homes first: a streaming home is blind RIGHT NOW,
                # while a dormant one has nothing to be blind about. Within
                # each group the oldest-waiting still wins.
                "   ORDER BY (" + _LIVE_CAM + ") DESC, created_at "
                "   LIMIT GREATEST((SELECT n FROM avail), 0) "
                "   FOR UPDATE SKIP LOCKED"
                "), pick AS ("
                "  SELECT id, row_number() OVER (ORDER BY created_at) AS rn "
                "    FROM locked"
                "), free_ports AS ("
                # CAST(), not `::int`: the params must be typed because
                # generate_series is overloaded (int/bigint/numeric/timestamp)
                # and asyncpg sends binds untyped -> "function is not unique".
                # `::` cannot be used here — SQLAlchemy's text() reads a colon
                # as the start of a bind parameter.
                "  SELECT gs AS port, row_number() OVER (ORDER BY gs) AS rn "
                "    FROM generate_series(CAST(:pmin AS int), CAST(:pmax AS int)) gs "
                "   WHERE gs NOT IN ("
                "     SELECT identity_port FROM public.patients "
                "      WHERE identity_node_id = :node AND identity_port IS NOT NULL"
                "   )"
                ") "
                "UPDATE public.patients p "
                "   SET identity_node_id = :node, identity_port = f.port "
                "  FROM pick JOIN free_ports f USING (rn) "
                " WHERE p.id = pick.id"
            ),
            {
                "node": body.node_id, "cap": body.capacity,
                "pmin": PORT_MIN, "pmax": PORT_MAX,
            },
        )
    ).rowcount or 0

    # LIMIT by capacity: lowering a node's capacity must actually shed load,
    # not just stop it claiming more. Oldest-held first, so the homes that
    # keep their place are the ones that have been settled longest — the
    # newest arrivals pay the re-ground.
    rows = (
        await session.execute(
            text(
                "SELECT id, identity_port FROM public.patients p "
                " WHERE identity_node_id = :node AND identity_port IS NOT NULL "
                # Shedding keeps the LIVE homes; settled-longest breaks ties.
                " ORDER BY (" + _LIVE_CAM + ") DESC, created_at LIMIT :cap"
            ),
            {"node": body.node_id, "cap": body.capacity},
        )
    ).all()

    await session.commit()

    rooms = [RosterRoom(room=f"patient_{pid}", port=port) for pid, port in rows]
    if reclaimed or claimed:
        # Worth a line at INFO: these are the only two events in the whole
        # scheme that change which machine is watching a home.
        log.info(
            "identity.roster node=%s rooms=%d claimed=%d reclaimed_from_stale=%d",
            body.node_id, len(rooms), claimed, reclaimed,
        )
    return RosterResponse(
        node_id=body.node_id,
        rooms=rooms,
        capacity=body.capacity,
        reclaimed=reclaimed,
    )
