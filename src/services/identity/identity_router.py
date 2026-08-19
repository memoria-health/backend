"""Routes each patient to the GPU node that actually serves them.

WHY THIS EXISTS
---------------
One identity process serves exactly ONE room, and a GPU node runs N of them.
Until now the backend held a single `IdentityClient` pointed at one global
`identity_state_url`, handed to every `RoomWorker`.

With more than one node that fails in the worst available way. The rooms on
the configured node prefix-match the payload and get real verdicts; every
other patient misses `_cam_entry`, gets `IdentityResult.empty()`, and is
therefore never bound to a track — no PATIENT label, no ledger bind, identity
suppression permanently bypassed. No error, no log, no verdict, forever.

This class resolves `camera -> identity_nodes.base_url` from the roster and
delegates to a per-node client.

WHY BY CAMERA AND NOT BY PATIENT (2026-08-20)
---------------------------------------------
It used to resolve by patient, which held only while every camera watched
exactly one person. A camera watching two residents publishes into ONE room
and so lives in ONE process; routing the second resident's ticks by their own
patient id sends them to a process that has never seen that camera, which
returns no verdict and fails open. The camera's stored `livekit_room` is the
address of the process that actually has the video, and it is used here as an
address only — nothing about who anyone is.

DROP-IN BY DESIGN
-----------------
`label_clip` already takes `camera_id`. So the router implements the same
surface `CameraIngest` already calls and nothing downstream changes:
`worker.py` constructs this instead of `IdentityClient`, and
`RoomWorker`/`CameraIngest` are untouched.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from uuid import UUID

from sqlalchemy import text

from src.services.identity.identity_client import IdentityClient
from src.services.identity.result import IdentityResult

log = logging.getLogger(__name__)

# How long a patient -> node mapping is trusted before re-reading the roster.
# Short because it bounds how long a failover takes to be noticed here: the
# roster releases a dead node's homes after ~90s, another node claims them on
# its next poll, and the backend follows within this window.
_ROUTE_CACHE_S = 15.0


class IdentityRouter:
    """One client per GPU node, chosen per patient.

    Mirrors `IdentityClient`'s constructor kwargs so it can be swapped in
    without touching the caller.
    """

    def __init__(
        self, *, sessionmaker: Any, settings: Any, body_gallery: Any = None,
    ) -> None:
        self._sessionmaker = sessionmaker
        self._settings = settings
        # base_url -> client. One httpx pool per node, reused across every
        # room on that node.
        self._clients: dict[str, IdentityClient] = {}
        # camera_id -> (expires_at, base_url | None). None is cached too:
        # a camera no process holds should not re-query the roster every
        # tick.
        self._routes: dict[str, tuple[float, str | None]] = {}
        # camera_id -> base_url, remembered from the last label_clip so that
        # `timing()` — which only receives a camera_id — can reach the same
        # node without a second lookup.
        self._camera_node: dict[str, str] = {}

    # ── plumbing ───────────────────────────────────────────────────────
    def _client_for(self, base_url: str) -> IdentityClient:
        c = self._clients.get(base_url)
        if c is None:
            # Each client reads settings for its token and trace config, then
            # we point it at this node. Constructing lazily means a node that
            # never serves a room here never opens a connection pool.
            c = IdentityClient(
                sessionmaker=self._sessionmaker,
                settings=self._settings,
                base_url=base_url,
            )
            self._clients[base_url] = c
            log.info("identity.router node_client_opened url=%s", base_url)
        return c

    async def _base_url_for(self, camera_id: UUID | str) -> str | None:
        """The node process that actually has this camera's video.

        Keyed on the CAMERA, not on the patient the tick is about. A
        camera publishes into exactly one LiveKit room and exactly one
        identity process subscribes to that room, so a camera watching
        two residents is decoded once, in one place. Asking "which
        process serves patient B" would send B's ticks to a process
        that has never seen the shared camera, which returns no verdict
        and fails open — silently, and forever, since a camera that is
        simply absent looks identical to a service that is down.

        `cameras.livekit_room` is that address, stored, and it is read
        here as an address only: no part of this resolves who anyone
        is. The room name embeds a patient uuid for historical reasons
        and nothing downstream is allowed to read meaning into that.
        """
        key = str(camera_id)
        hit = self._routes.get(key)
        now = time.monotonic()
        if hit is not None and hit[0] > now:
            return hit[1]

        url: str | None = None
        try:
            async with self._sessionmaker() as session:
                row = (
                    await session.execute(
                        text(
                            # base_url is the HOST; the port is per-room,
                            # because the node runs one process per room. A
                            # node-level URL alone reaches exactly one home
                            # and silently starves the rest.
                            "SELECT n.base_url, p.identity_port "
                            "  FROM public.cameras c "
                            "  JOIN public.patients p "
                            "    ON c.livekit_room = 'patient_' || p.id::text "
                            "  JOIN public.identity_nodes n "
                            "    ON n.node_id = p.identity_node_id "
                            " WHERE c.id = :cid "
                            "   AND p.identity_port IS NOT NULL"
                        ),
                        {"cid": key},
                    )
                ).first()
            if row is not None:
                url = f"{row[0].rstrip('/')}:{row[1]}"
        except Exception:
            # Fail OPEN, like every other identity path: a database blip must
            # not stop detection. Distinguished from "unassigned" below,
            # because they mean very different things operationally.
            log.exception("identity.router route_lookup_failed camera=%s", key)
            # Cache the miss briefly so a sustained outage does not add a
            # query per camera per tick.
            self._routes[key] = (now + _ROUTE_CACHE_S, None)
            return None

        if url is None:
            # No process holds this camera. Distinct from "the node is down",
            # which surfaces as state_unreachable from the client.
            log.warning("identity.router unassigned camera=%s", key)
        self._routes[key] = (now + _ROUTE_CACHE_S, url)
        return url

    # ── IdentityClient surface ─────────────────────────────────────────
    async def label_clip(self, **kwargs: Any) -> IdentityResult:
        camera_id = kwargs.get("camera_id")
        base_url = await self._base_url_for(camera_id) if camera_id else None
        if base_url is None:
            # No node serves this home. Fail open — the same shape the client
            # returns when the service is unreachable — so safety detectors
            # keep firing identity-agnostically.
            return IdentityResult.empty(faces_enrolled=True)
        if camera_id is not None:
            self._camera_node[str(camera_id)] = base_url
        return await self._client_for(base_url).label_clip(**kwargs)

    def timing(self, camera_id: Any) -> Any:
        """Delegate to whichever node last served this camera.

        `timing` reads the client's CACHED state and never fetches, so this
        is only ever asked right after a `label_clip` for the same camera.
        """
        base_url = self._camera_node.get(str(camera_id))
        if base_url is None:
            return None
        client = self._clients.get(base_url)
        return client.timing(camera_id) if client is not None else None

    async def aclose(self) -> None:
        for url, client in list(self._clients.items()):
            try:
                await client.aclose()
            except Exception:
                log.exception("identity.router close_failed url=%s", url)
        self._clients.clear()
