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
from sqlalchemy import select

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
