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

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.api.deps import SessionDep, SettingsDep

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/identity", tags=["identity"])


class RosterRequest(BaseModel):
    node_id: str = Field(min_length=1, max_length=128)
    # Where the backend should reach this node's identity processes. In
    # production a tailnet MagicDNS name, precisely so it survives RunPod
    # reassigning the pod's direct-TCP endpoint on every restart.
    base_url: str = Field(min_length=1, max_length=512)
    # How many rooms this node will accept. Node-local because it depends on
    # NEMARA_PERCEPTION_CONCURRENCY and the GPU.
    capacity: int = Field(ge=1, le=64)


class RosterResponse(BaseModel):
    node_id: str
    # Rooms this node should be running, as `patient_<uuid>`. Authoritative:
    # the node reconciles to exactly this list, starting what is new and
    # stopping what is gone.
    rooms: list[str]
    capacity: int
    # How many homes this poll took off other (stale) nodes. Non-zero means a
    # failover just happened and is worth an alert.
    reclaimed: int


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
                "UPDATE public.patients SET identity_node_id = NULL "
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
    claimed = (
        await session.execute(
            text(
                "WITH room AS ("
                "  SELECT :cap - count(*) AS n FROM public.patients "
                "   WHERE identity_node_id = :node"
                "), pick AS ("
                "  SELECT id FROM public.patients "
                "   WHERE identity_node_id IS NULL "
                "   ORDER BY created_at "
                "   LIMIT GREATEST((SELECT n FROM room), 0) "
                "   FOR UPDATE SKIP LOCKED"
                ") "
                "UPDATE public.patients p SET identity_node_id = :node "
                "  FROM pick WHERE p.id = pick.id"
            ),
            {"node": body.node_id, "cap": body.capacity},
        )
    ).rowcount or 0

    rows = (
        await session.execute(
            text(
                "SELECT id FROM public.patients "
                " WHERE identity_node_id = :node ORDER BY created_at"
            ),
            {"node": body.node_id},
        )
    ).scalars().all()

    await session.commit()

    rooms = [f"patient_{pid}" for pid in rows]
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
