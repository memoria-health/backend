"""Push-mode ingest endpoint for Pi cameras.

POST /v1/cameras/me/frames

The Pi opens one long-lived HTTPS POST and streams a
`multipart/x-mixed-replace` body of JPEG frames. The endpoint parses
the body in-place (no buffering), feeds the JPEG bytes into a
`RelayPublisher`, which decodes them, runs motion detection, and
publishes as a LiveKit `cam_*` participant. From every downstream
component's perspective (RoomWorker, CameraIngest, clip_archival)
the Pi is indistinguishable from an iPhone-published camera.

Auth: `Authorization: Bearer <token>` where `<token>` is the
plaintext value the Pi received on its pi-announce response the
moment its discovery was claimed. Backend stores only the SHA-256
hash on `cameras.ingest_token_hash`; the push handler hashes the
presented token and looks the camera up by hash.

Concurrency: only one push connection per camera at a time. If a
new connection arrives while an old one is still active (e.g. the Pi
reconnected without cleanly closing the prior request), the new
connection wins and the old one is cancelled. This mirrors the
LiveKit "DuplicateIdentity" semantics on the iPhone path: most-
recent-wins is the correct behavior for a single physical device.

Failure model:
  - Per-frame decode failures handled inside `RelayPublisher`
    (consecutive-error cap raises and the response closes).
  - Auth failure -> 401 BEFORE any body is consumed.
  - Body parse failure -> the relay raises MjpegStreamError; the
    response returns 400 with a short reason. Pi retries.
  - LiveKit connect failure -> 502; Pi backs off and retries.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator

from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import select, text

from src.api.deps import SessionDep, SettingsDep
from src.db.camera import Camera
from src.services.ingest.pi.auth import hash_ingest_token
from src.services.ingest.pi.mjpeg_reader import parse_mjpeg
from src.services.ingest.pi.relay_publisher import RelayConfig, RelayPublisher

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["pi-ingest"])

# In-process registry of active push connections, keyed by camera_id.
# The value is the asyncio.Task wrapping the RelayPublisher run loop;
# a new connection cancels and replaces the entry so a stale prior
# connection cannot keep publishing after the Pi reconnected.
_active_relays: dict[str, asyncio.Task[None]] = {}

# Matches the boundary parameter from a multipart/x-mixed-replace
# Content-Type header. Same regex shape as mjpeg_reader.read_mjpeg
# uses for the pull-mode response Content-Type.
_BOUNDARY_RE = re.compile(rb'boundary="?([^";\s]+)"?')


def _extract_boundary(content_type: str | None) -> bytes:
    """Pull the boundary parameter out of the Content-Type header.

    The Pi sets `Content-Type: multipart/x-mixed-replace; boundary=frame`
    on its outbound POST. Anything else is a client bug; we 400 it.
    """
    if not content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="missing_content_type",
        )
    lower = content_type.lower()
    if "multipart/x-mixed-replace" not in lower:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="expected_multipart_mixed_replace",
        )
    match = _BOUNDARY_RE.search(content_type.encode("latin-1"))
    if match is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="missing_boundary_parameter",
        )
    return match.group(1)


@router.post(
    "/cameras/me/frames",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def push_frames(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    authorization: str = Header(default=""),
    content_type: str | None = Header(default=None),
) -> dict[str, str]:
    """Long-lived push endpoint. Returns after the Pi disconnects."""
    # ── auth ────────────────────────────────────────────────────
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)

    # Indexed equality on the hash column; constant-time comparison
    # at the application layer is unnecessary because we're matching
    # a SHA-256 digest against an indexed column, not bytewise-
    # comparing the presented token.
    cam_stmt = select(Camera).where(Camera.ingest_token_hash == digest).limit(1)
    cam = (await session.execute(cam_stmt)).scalar_one_or_none()
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_ingest_token",
        )

    # ── desired_state gate ──────────────────────────────────────
    # When the caregiver hits Disconnect for a Pi camera, this row's
    # desired_state flips to 'disconnected'. Returning 423 LOCKED
    # with {"paused": true} tells the Pi to stop pushing immediately
    # and switch to its low-rate paused-mode loop, which polls
    # `/v1/cameras/me/ingest-status` once every few seconds until the
    # state flips back. While paused the Pi sends zero frame bytes
    # upstream — no LiveKit egress, no ngrok request budget burn,
    # no detector ticks for a disconnected camera. The wall phone has
    # a symmetric path via the LiveKit publisher tear-down; this is
    # the Pi-side equivalent.
    if cam.desired_state == "disconnected":
        return JSONResponse(
            status_code=status.HTTP_423_LOCKED,
            content={"paused": True, "reason": "desired_state_disconnected"},
        )

    # ── body shape ──────────────────────────────────────────────
    boundary = _extract_boundary(content_type)

    # ── kick prior connection (most-recent-wins) ───────────────
    camera_key = str(cam.id)
    prior = _active_relays.pop(camera_key, None)
    if prior is not None and not prior.done():
        prior.cancel()
        try:
            await prior
        except (asyncio.CancelledError, Exception):
            # Prior connection's exception is no longer interesting;
            # the new connection supersedes it.
            pass

    # ── run the relay against the streaming request body ──────
    config = RelayConfig(
        camera_id=cam.id,
        camera_identity=cam.livekit_identity or f"cam_{cam.id.hex[:8]}",
        room_name=cam.livekit_room or f"patient_{cam.patient_id}",
        pi_stream_url="",  # unused in push mode; see RelayConfig docstring
    )
    publisher = RelayPublisher(config, settings)
    body_chunks: AsyncIterator[bytes] = request.stream()
    frames = parse_mjpeg(body_chunks, boundary=boundary)

    task = asyncio.create_task(
        publisher.run_from_iterator(frames),
        name=f"pi_push_{camera_key}",
    )
    _active_relays[camera_key] = task

    try:
        await task
    except asyncio.CancelledError:
        # The task we awaited above was cancelled (by a newer
        # connection arriving). Propagate cancellation up so FastAPI
        # closes the HTTP response cleanly.
        raise
    except Exception as exc:
        log.exception(
            "pi_push.relay_failed camera=%s err=%s", camera_key, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="relay_failed",
        ) from exc
    finally:
        # Only remove the entry if we're still the active task.
        # Newer connection may have replaced us already.
        if _active_relays.get(camera_key) is task:
            _active_relays.pop(camera_key, None)

    return {"status": "stream_closed"}


async def cancel_active_relay(camera_id: UUID) -> bool:
    """Cancel any in-flight push connection for `camera_id`. Called
    from the caregiver-facing camera PATCH handler the moment a
    desired_state transition to 'disconnected' commits, so the Pi
    drops its current upload stream within ~1 frame instead of
    finishing the current 30-min connection window first.

    Returns True iff a task was found and cancellation was requested.
    Idempotent: a second call for the same camera is a cheap no-op.
    The Pi will see EOF on its push, reconnect, hit the
    desired_state gate in `push_frames`, and switch to paused mode.
    """
    task = _active_relays.pop(str(camera_id), None)
    if task is None or task.done():
        return False
    task.cancel()
    return True


@router.get(
    "/cameras/me/ingest-status",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def get_ingest_status(
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict[str, str]:
    """Poll endpoint for the Pi's paused-mode loop.

    While paused (after a 423 from `/v1/cameras/me/frames`) the Pi
    cannot keep its long-lived push connection open — there's nothing
    upstream that would push a "resume" signal down to it. Instead it
    polls this cheap GET every few seconds and reads back the live
    desired_state. The moment it sees 'connected' it switches back
    into the frame-push loop.

    Auth: identical shape to the push endpoint — same ingest_token
    bearer, same `cameras.ingest_token_hash` lookup. Deliberately a
    separate endpoint from `/v1/cameras/me/desired-state` (which is
    api-key-authenticated for the wall phone); the Pi has only its
    ingest token, no api_key, so it can't authenticate against the
    wall-phone surface.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)
    cam_stmt = select(Camera).where(Camera.ingest_token_hash == digest).limit(1)
    cam = (await session.execute(cam_stmt)).scalar_one_or_none()
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_ingest_token",
        )
    return {"desired_state": cam.desired_state}


@router.post(
    "/cameras/me/heartbeat",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def post_heartbeat(
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict[str, bool]:
    """Liveness ping from a gateway (mini-PC pulling RTSP).

    The gateway posts this every few seconds per camera it manages. We stamp
    `cameras.last_heartbeat_at`. A stale stamp means the GATEWAY itself is down
    (crashed / power loss / lost internet) — which is DISTINCT from
    `desired_state='disconnected'` (a caregiver intentionally turned it off) and
    from `online=false` (not currently publishing). That three-way distinction
    is what lets the dashboard say "gateway offline" instead of mislabeling a
    dead mini-PC as an intentional disconnect.

    Same ingest-token bearer auth as the push + ingest-status endpoints, and a
    single raw UPDATE keyed by the token hash (no ORM model change needed).
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)
    result = await session.execute(
        text(
            "UPDATE public.cameras SET last_heartbeat_at = now() "
            "WHERE ingest_token_hash = :digest RETURNING id"
        ),
        {"digest": digest},
    )
    if result.first() is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_ingest_token",
        )
    await session.commit()
    return {"ok": True}


@router.get(
    "/gateway/roster",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def gateway_roster(
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict:
    """The gateway's source-of-truth roster.

    A mini-PC gateway holds ONE stable site token (not per-camera tokens) and
    polls this every few seconds. We return exactly the cameras it should be
    running, with their live desired_state — so a dashboard add/delete/connect/
    disconnect propagates on the next poll, and a deleted camera simply vanishes
    from the list (no orphaned per-camera token, no 401 flood). The act of
    polling also stamps liveness (gateway + each camera's last_heartbeat_at), so
    this single call replaces the separate per-camera desired-state + heartbeat
    polls. Same sha256 bearer-token shape as the ingest endpoints.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)
    gw = (await session.execute(
        text("SELECT id, patient_id FROM public.gateways WHERE token_hash = :d"),
        {"d": digest},
    )).first()
    if gw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_gateway_token",
        )
    gateway_id, patient_id = gw[0], gw[1]
    # Liveness: polling the roster IS the heartbeat (gateway + its cameras).
    await session.execute(
        text("UPDATE public.gateways SET last_seen_at = now() WHERE id = :id"),
        {"id": gateway_id},
    )
    await session.execute(
        text("UPDATE public.cameras SET last_heartbeat_at = now() "
             "WHERE patient_id = :p AND source_type = 'gateway'"),
        {"p": patient_id},
    )
    rows = (await session.execute(
        text("SELECT id, livekit_identity, livekit_room, desired_state "
             "FROM public.cameras WHERE patient_id = :p AND source_type = 'gateway' "
             "ORDER BY created_at"),
        {"p": patient_id},
    )).all()
    await session.commit()
    return {"cameras": [
        {"camera_id": str(r[0]), "livekit_identity": r[1],
         "livekit_room": r[2], "desired_state": r[3]}
        for r in rows
    ]}


@router.get(
    "/gateway/wait",
    status_code=status.HTTP_200_OK,
    response_model=None,
)
async def gateway_wait(
    session: SessionDep,
    since: str = "",
    authorization: str = Header(default=""),
) -> dict:
    """Long-poll: HOLD this request open until the gateway's roster (which
    cameras + their desired_state, fingerprinted) differs from `since`, then
    return the new roster. The gateway keeps ONE of these open and re-issues it,
    so a dashboard connect/disconnect reaches it in ~0.4s instead of waiting for
    the 3s poll. The plain `/gateway/roster` poll stays as the fallback if this
    drops. (Internal 0.4s DB check; a Postgres LISTEN/NOTIFY upgrade would make
    it truly instant + zero internal polling.)
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    presented = authorization[len("bearer "):].strip()
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    digest = hash_ingest_token(presented)
    gw = (await session.execute(
        text("SELECT id, patient_id FROM public.gateways WHERE token_hash = :d"),
        {"d": digest},
    )).first()
    if gw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_gateway_token",
        )
    patient_id = gw[1]

    async def snapshot():
        rows = (await session.execute(
            text("SELECT id, livekit_identity, livekit_room, desired_state "
                 "FROM public.cameras WHERE patient_id = :p AND source_type = 'gateway' "
                 "ORDER BY livekit_identity"),
            {"p": patient_id},
        )).all()
        await session.commit()   # release the connection between checks
        cams = [{"camera_id": str(r[0]), "livekit_identity": r[1],
                 "livekit_room": r[2], "desired_state": r[3]} for r in rows]
        fp = "|".join(f"{c['livekit_identity']}:{c['desired_state']}" for c in cams)
        return cams, fp

    elapsed = 0.0
    while elapsed < 25.0:
        cams, fp = await snapshot()
        if fp != since:
            return {"changed": True, "cameras": cams, "fingerprint": fp}
        await asyncio.sleep(0.4)
        elapsed += 0.4
    cams, fp = await snapshot()
    return {"changed": False, "fingerprint": fp}
