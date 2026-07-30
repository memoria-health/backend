"""Liveness + readiness probes.

`/health` is the cheap liveness check: did the process boot.
`/ready`  is the readiness check: can it reach Postgres.
"""

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session

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

    # Concurrent, and short — a readiness probe must not take longer
    # than the interval a monitor polls it on.
    lk_url = (
        settings.livekit_url.replace("wss://", "https://")
        if settings.livekit_url else ""
    )
    # A prod deploy that still carries the dev-rig identity URL is
    # MISCONFIGURED, not down — probing it would just time out on every
    # call and report "fail" forever. Say so by name and skip the probe.
    id_configured = not (
        settings.env not in ("local", "test")
        and _IDENTITY_DEV_DEFAULT in settings.identity_state_url
    )
    id_url = f"{settings.identity_state_url}/state"
    id_token = getattr(settings, "identity_state_token", "")
    id_headers = {"Authorization": f"Bearer {id_token}"} if id_token else None
    results = await asyncio.gather(
        _probe(lk_url) if lk_url else _noop_ok(),
        _probe(id_url, id_headers) if id_configured else _noop_ok(),
    )
    if lk_url:
        checks["livekit"] = results[0]
    checks["identity"] = results[1] if id_configured else "not_configured"

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
