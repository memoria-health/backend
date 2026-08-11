"""Liveness + readiness probes.

`/health` is the cheap liveness check: did the process boot.
`/ready`  is the readiness check: can it reach Postgres.
"""

import logging
import secrets
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session

log = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]

# The dev-rig default for the identity sidecar. Only reachable from a
# container on the same Mac; on Railway it resolves to nothing, so a
# deploy that never set NEMARA_IDENTITY_STATE_URL would make /ready read
# "fail" forever and teach everyone to ignore it. Detected by name below
# and reported as not_configured instead — a different word, because the
# remedy is a deploy env var, not an incident.
_IDENTITY_DEV_DEFAULT = "host.docker.internal"

# Stamped by /v1/gateway/wait each time the gateway long-polls; /ready
# turns it into a "how long since the gateway checked in" age.
last_gateway_poll_mono: float = float("-inf")


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str


class ReadyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    checks: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness — is the process running."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    x_internal_key: Annotated[str, Header()] = "",
) -> ReadyResponse:
    """Readiness + dependency health (comprehensive, 2026-07-29).

    DB is the only HARD gate (503) — the backend genuinely cannot
    serve without it. Every other dependency is REPORTED, not gated:
    a healthcheck that kills/recycles the backend because the identity
    sidecar or LiveKit had a blip would turn one degraded dependency
    into a full outage. Overall status: ok | degraded.

    Caveat learned the hard way (both 2026-07-29 incidents): a
    dependency answering HTTP proves reachability, NOT data flow —
    LiveKit pinged fine while zero frames arrived, twice, for two
    different reasons. /health/freshness stays the product-truth
    probe; these checks are the diagnosis layer that says WHICH leg
    broke when freshness goes red.

    AUTH (added 2026-07-29): this handler makes outbound calls and
    names our dependencies, so outside local/test it needs the same
    `X-Internal-Key` as the /internal routes and fails closed when
    that key is unset. `/health` stays open — that is the one a
    platform healthcheck should point at, and Railway's config names
    no path at all today, so nothing breaks by locking this down.
    """
    import asyncio
    import time as _time

    import httpx

    from src.db.session import get_sessionmaker
    from src.settings import get_settings

    settings = get_settings()
    expected = settings.internal_service_key
    if expected:
        if not secrets.compare_digest(x_internal_key, expected):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="invalid_internal_key",
            )
    elif settings.env not in ("local", "test"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="internal_key_not_configured",
        )

    checks: dict[str, str] = {}
    # The DB connection is taken and RELEASED before any network call.
    # Injecting the request-scoped session instead would pin a pool
    # connection for the whole handler, including the outbound probes
    # below — a few requests per second of this probe would then hold
    # the entire pool, which is how the 2026-07-28 production login
    # outage started. (The auth gate above now also rejects before any
    # connection is taken, so an unauthorized caller costs nothing.)
    try:
        sm = get_sessionmaker()
        async with sm() as probe_session:
            (await probe_session.execute(text("SELECT 1"))).one()
        checks["db"] = "ok"
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "fail", "checks": {"db": "fail"}, "error": str(exc)},
        ) from exc

    async def _probe(url: str, headers: dict[str, str] | None = None) -> str:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                r = await client.get(url, headers=headers or {})
            # 401/403 is NOT reachability — it means our token is wrong
            # or missing, and `< 500` would have quietly called that
            # "ok". A check that passes when we cannot actually talk to
            # the dependency is worse than no check.
            if r.status_code in (401, 403):
                return "unauthorized"
            return "ok" if r.status_code < 500 else "fail"
        except httpx.HTTPError:
            return "fail"
        except httpx.InvalidURL:
            # NOT an HTTPError — it derives straight from Exception, so the
            # clause above misses it. Reachable because a node's base_url is
            # self-reported into identity_nodes; a malformed one would turn a
            # readiness probe into a 500. The roster validates on the way in,
            # this covers rows written before it did.
            return "bad_url"

    # Concurrent, and short — a readiness probe must not take longer
    # than the interval a monitor polls it on.
    lk_url = (
        settings.livekit_url.replace("wss://", "https://")
        if settings.livekit_url else ""
    )
    id_token = getattr(settings, "identity_state_token", "")
    id_headers = {"Authorization": f"Bearer {id_token}"} if id_token else None

    # Identity is a FLEET now, not one sidecar. Probing a single configured
    # URL was accurate when there was one process; with the roster it reports
    # on at most one node and would say "ok" with every other node dead —
    # exactly the silent-blindness failure the roster exists to prevent.
    #
    # So: ask the roster who is registered, then probe one room per node.
    # One per node, not one per room: this catches the failure that actually
    # takes a whole node out (the tailnet path, a dead container) at a cost
    # that stays flat as homes are added. A single wedged room is a
    # per-home concern and belongs to /health/freshness, which is the
    # product-truth probe.
    nodes = await _identity_node_targets(
        sm, getattr(settings, "identity_roster_stale_after_s", 90),
    )
    if nodes:
        # LiveKit rides along in the SAME gather. Every probe carries a 1.5s
        # timeout, so running the legs in sequence would make the worst case
        # additive — and a readiness endpoint that takes longer than its own
        # poll interval starts reporting on the past.
        lk_result, *node_results = await asyncio.gather(
            _probe(lk_url) if lk_url else _noop_ok(),
            *(_probe(f"{url}/state", id_headers) for _, url in nodes),
        )
        live = sum(1 for r in node_results if r == "ok")
        if live == len(nodes):
            checks["identity"] = "ok"
        else:
            # Name the count, not just "fail". "1 of 3 nodes" is a capacity
            # problem to page on; "0 of 3" is an outage; the two want very
            # different responses at 3am.
            checks["identity"] = f"degraded_{live}_of_{len(nodes)}_nodes"
        for (node_id, _), r in zip(nodes, node_results, strict=True):
            if r != "ok":
                checks[f"identity_node:{node_id}"] = r
    else:
        # No node has ever registered. Fall back to the single-URL probe so
        # local development — where identity runs on the laptop and there is
        # no roster at all — keeps working unchanged.
        #
        # A prod deploy that still carries the dev-rig identity URL is
        # MISCONFIGURED, not down — probing it would just time out on every
        # call and report "fail" forever. Say so by name and skip the probe.
        id_configured = not (
            settings.env not in ("local", "test")
            and _IDENTITY_DEV_DEFAULT in settings.identity_state_url
        )
        lk_result, id_result = await asyncio.gather(
            _probe(lk_url) if lk_url else _noop_ok(),
            _probe(f"{settings.identity_state_url}/state", id_headers)
            if id_configured else _noop_ok(),
        )
        if id_configured:
            checks["identity"] = id_result
        elif settings.env not in ("local", "test"):
            # Production with no registered node and no usable URL means
            # nothing is watching any home. That is not "not configured",
            # it is down, and it must not read as benign.
            checks["identity"] = "no_nodes_registered"
        else:
            checks["identity"] = "not_configured"

    if lk_url:
        checks["livekit"] = lk_result

    age = _time.monotonic() - last_gateway_poll_mono
    if last_gateway_poll_mono == float("-inf"):
        checks["gateway_poll"] = "never"       # fresh boot — not yet seen
    else:
        checks["gateway_poll"] = "ok" if age < 120 else f"stale_{int(age)}s"

    degraded = any(v not in ("ok", "never") for v in checks.values())
    return ReadyResponse(
        status="degraded" if degraded else "ok", checks=checks,
    )


class FreshnessResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: str
    ages_s: dict[str, int]
    stale: dict[str, int]
    # Cameras that are online but have never produced an analysis yet
    # (just paired, or just back from offline — `cameras.online` flips
    # as soon as frames arrive, which is BEFORE the first analysis
    # exists). Reported, never counted as stale: treating "no data
    # yet" as "data stopped" made every newly-online camera trigger a
    # restart, and the restart re-entered the same warm-up.
    warming: list[str] = []


async def _noop_ok() -> str:
    """Placeholder result for a dependency that isn't configured."""
    return "ok"


async def _identity_node_targets(sm: Any, stale_s: int) -> list[tuple[str, str]]:
    """One probeable `(node_id, base_url:port)` per registered GPU node.

    Only nodes that are BOTH heartbeating and actually serving a room are
    returned, for opposite reasons:

    * A node whose last poll has aged out is already being drained by the
      roster — its homes are being handed to a survivor. Reporting it as a
      dead dependency would page for a failover that is working.
    * A node with no assigned room has nothing listening on any identity
      port yet, so probing it would report "fail" for an idle-but-healthy
      node. This is the normal state of a freshly started node in the
      seconds before its first claim.

    Lowest port per node: an arbitrary but STABLE choice, so the check reads
    the same across polls rather than hopping between rooms.
    """
    try:
        async with sm() as s:
            rows = (
                await s.execute(
                    text(
                        "SELECT n.node_id, n.base_url, min(p.identity_port) "
                        "  FROM public.identity_nodes n "
                        "  JOIN public.patients p "
                        "    ON p.identity_node_id = n.node_id "
                        "   AND p.identity_port IS NOT NULL "
                        " WHERE n.last_seen_at > now() - make_interval("
                        "         secs => :stale) "
                        " GROUP BY n.node_id, n.base_url "
                        " ORDER BY n.node_id"
                    ),
                    # Passed in rather than read here: the handler already
                    # holds settings, and re-importing get_settings inside a
                    # helper would bind past the monkeypatch the tests use.
                    {"stale": stale_s},
                )
            ).all()
        return [(r[0], f"{r[1].rstrip('/')}:{r[2]}") for r in rows]
    except SQLAlchemyError:
        # The db check above already ran and passed, so this is unexpected —
        # most likely the roster delta has not been applied to this
        # environment. Degrade to the single-URL path rather than 500 a
        # readiness probe.
        log.exception("ready: could not read identity node roster")
        return []


_FRESHNESS_STALE_S = 300


@router.get("/health/freshness", response_model=FreshnessResponse)
async def freshness(session: SessionDep) -> FreshnessResponse:
    """Data-flowing probe — the check /health cannot make.

    2026-07-29 incident: the container's outbound LiveKit connections
    wedged for 108 minutes while /health answered 200 the whole time —
    process-alive is not product-working. This reports the age of each
    ONLINE camera's newest analysis row and goes 503 when any exceeds
    the stale threshold, so an external monitor (dev sentinel, prod
    Railway healthcheck / pinger) can key on data freshness directly.
    Offline/paused cameras are excluded — a camera that is not
    supposed to produce data must not fail the probe.
    """
    # LATERAL + LIMIT 1 walks the (camera_id, captured_at) index
    # backwards and touches one row per camera. The obvious
    # max()+GROUP BY version scans the whole analyses table on every
    # call (measured: 35ms over 162k rows and growing without bound,
    # vs 0.55ms here) — and a monitor polls this every 30s forever.
    rows = (await session.execute(text(
        "select substr(c.id::text, 1, 8) as cam,"
        " extract(epoch from now() - m.captured_at)::int as age"
        " from cameras c left join lateral ("
        "   select a.captured_at from analyses a"
        "   where a.camera_id = c.id order by a.captured_at desc limit 1"
        " ) m on true"
        " where c.online and not c.disabled and not c.ai_paused"
    ))).all()
    ages = {r.cam: r.age for r in rows if r.age is not None}
    warming = sorted(r.cam for r in rows if r.age is None)
    stale = {k: v for k, v in ages.items() if v > _FRESHNESS_STALE_S}
    if stale:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "stale", "ages_s": ages, "stale": stale,
                "warming": warming,
            },
        )
    return FreshnessResponse(
        status="ok", ages_s=ages, stale={}, warming=warming,
    )
