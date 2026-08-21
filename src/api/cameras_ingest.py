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
import time
from collections.abc import AsyncIterator

from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api import health as health_mod
from src.api.deps import SessionDep, SettingsDep
from src.db.camera import Camera
from src.repos import camera_detector_repo
from src.services.ingest.pi.auth import hash_ingest_token, mint_ingest_token
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
        text("SELECT id, caregiver_id FROM public.gateways WHERE token_hash = :d"),
        {"d": digest},
    )).first()
    if gw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_gateway_token",
        )
    gateway_id, caregiver_id = gw[0], gw[1]
    # Liveness (polling the roster IS the heartbeat) + reverse-SSH tunnel port
    # in one statement. The port is assigned lazily on the box's FIRST poll
    # (covers rows that predate the column too — no backfill needed); the box
    # writes it to .tunnel-port and its tunnel unit dials the relay on it,
    # which is what makes every deployed box reachable by ops. COALESCE
    # short-circuits (Postgres §9.18), so nextval() fires only when the port
    # is NULL — steady-state polls burn nothing — and the row lock serializes
    # concurrent first polls, so RETURNING always yields the final port.
    # nextval() never issues 22001 (reserved for the legacy box-01 unit); see
    # scripts/apply_gateway_tunnel_port_delta.sql for the per-env port ranges.
    tunnel_port = (await session.execute(
        text("UPDATE public.gateways SET last_seen_at = now(), "
             "tunnel_port = COALESCE(tunnel_port, "
             "  nextval('public.gateway_tunnel_port_seq')) "
             "WHERE id = :id RETURNING tunnel_port"),
        {"id": gateway_id},
    )).scalar()
    # A gateway box serves a CAREGIVER, but each gateway camera is OWNED by
    # a specific box (cameras.gateway_id). Return only THIS box's cameras,
    # plus any not-yet-assigned (gateway_id IS NULL) rows as a transitional
    # fallback until the ownership backfill/provision fills them in. This is
    # what makes liveness per-box: the heartbeat below only stamps the
    # cameras this box actually runs.
    # Ownership is the ACCOUNT (cameras.caregiver_id), not the patient — a
    # camera can be on the account before anyone assigns it. `patient_id IS
    # NOT NULL` is what keeps those unassigned ones OUT of the roster, so
    # they never publish until someone says who they watch.
    _cam_where = ("WHERE (caregiver_id = :c OR patient_id IN "
                  "      (SELECT id FROM public.patients WHERE caregiver_id = :c)) "
                  "AND patient_id IS NOT NULL "
                  "AND source_type = 'gateway' "
                  "AND (gateway_id = :gw OR gateway_id IS NULL)")
    await session.execute(
        text(f"UPDATE public.cameras SET last_heartbeat_at = now() {_cam_where}"),
        {"c": caregiver_id, "gw": gateway_id},
    )
    rows = (await session.execute(
        text("SELECT id, livekit_identity, livekit_room, desired_state, "
             "privacy_enabled, online, "
             # The STRICTEST covered patient wins. A shared camera
             # streams into the primary's room, so reading
             # cameras.patient_id read the primary ONLY and a secondary
             # patient's audio-off was silently ignored — the same
             # fall-back-to-primary we removed from identity, here in
             # the facilities audio kill, where being wrong means a
             # microphone open at a site that forbids recording.
             # COALESCE covers a camera with no coverage rows yet
             # (pre-backfill); patient_id still mirrors the primary.
             "COALESCE("
             " (SELECT bool_and(p.audio_capture_enabled)"
             "    FROM public.camera_patients cp"
             "    JOIN public.patients p ON p.id = cp.patient_id"
             "   WHERE cp.camera_id = cameras.id),"
             " (SELECT audio_capture_enabled FROM public.patients p "
             "   WHERE p.id = cameras.patient_id)"
             ") AS audio_capture_enabled "
             f"FROM public.cameras {_cam_where} ORDER BY created_at"),
        {"c": caregiver_id, "gw": gateway_id},
    )).all()
    await session.commit()
    # `privacy_enabled` (room-derived, 2026-07-27): when true the gateway
    # restricts this camera's track subscription permissions to
    # detector_worker_* identities (publisher-side sever) — same contract
    # as the wall phone's desired-state poll.
    # `audio_capture_enabled` (patient-level, 2026-07-28): when false the
    # gateway never publishes the audio track — the facilities audio kill,
    # severed at the source so nothing downstream can leak it.
    # `online` (2026-07-29): the SUBSCRIBER's view of whether this
    # camera's frames are actually landing here — stamped by the room
    # worker on real frame arrival and independent of detector activity
    # (an ai_paused camera still reads online). The gateway cannot see
    # this about itself: in the 2026-07-29 wifi-blip zombie its room,
    # publications and connection_state all said healthy while zero
    # media left the box. Handing it back closes that blind spot.
    # Deliberately NOT in /gateway/wait's fingerprint — `online` flaps
    # on ordinary blips, and fingerprinting it would wake the long poll
    # constantly for something the 3s roster poll delivers anyway.
    return {"tunnel_port": tunnel_port, "cameras": [
        {"camera_id": str(r[0]), "livekit_identity": r[1],
         "livekit_room": r[2], "desired_state": r[3],
         "privacy_enabled": bool(r[4]),
         "online": bool(r[5]),
         "audio_capture_enabled": bool(r[6]) if r[6] is not None else True}
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
    # Dependency-health stamp: /ready reports how long ago the gateway
    # last long-polled — a box whose service died or whose network
    # dropped stops polling, and that age is the cheapest backend-side
    # signal of it. Must stay BELOW the docstring: put above it, the
    # string stops being the docstring and the contract above vanishes
    # from /docs and from help().
    health_mod.last_gateway_poll_mono = time.monotonic()
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
        text("SELECT id, caregiver_id FROM public.gateways WHERE token_hash = :d"),
        {"d": digest},
    )).first()
    if gw is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_gateway_token",
        )
    gateway_id, caregiver_id = gw[0], gw[1]

    async def snapshot():
        rows = (await session.execute(
            text("SELECT id, livekit_identity, livekit_room, desired_state, "
                 "privacy_enabled, "
                 # Same strictest-covered-patient rule as the roster. It
                 # has to match exactly: this feeds the fingerprint the
                 # long poll wakes on, so a disagreement here would wake
                 # the box to be told the opposite of what the roster
                 # says, forever.
                 "COALESCE("
                 " (SELECT bool_and(p.audio_capture_enabled)"
                 "    FROM public.camera_patients cp"
                 "    JOIN public.patients p ON p.id = cp.patient_id"
                 "   WHERE cp.camera_id = cameras.id),"
                 " (SELECT audio_capture_enabled FROM public.patients p "
                 "   WHERE p.id = cameras.patient_id)"
                 ") AS audio_capture_enabled "
                 "FROM public.cameras WHERE patient_id IN "
                 "(SELECT id FROM public.patients WHERE caregiver_id = :c) "
                 "AND source_type = 'gateway' "
                 "AND (gateway_id = :gw OR gateway_id IS NULL) "
                 "ORDER BY livekit_identity"),
            {"c": caregiver_id, "gw": gateway_id},
        )).all()
        await session.commit()   # release the connection between checks
        cams = [{"camera_id": str(r[0]), "livekit_identity": r[1],
                 "livekit_room": r[2], "desired_state": r[3],
                 "privacy_enabled": bool(r[4]),
                 "audio_capture_enabled":
                     bool(r[5]) if r[5] is not None else True}
                for r in rows]
        # privacy_enabled + audio_capture_enabled ride in the fingerprint
        # so a classifier / reclassify / audio-kill flip wakes the
        # long-poll immediately — the gateway must react within ~0.4s,
        # not wait for the 3s roster fallback.
        fp = "|".join(
            f"{c['livekit_identity']}:{c['desired_state']}"
            f":{int(c['privacy_enabled'])}"
            f":{int(c['audio_capture_enabled'])}"
            for c in cams
        )
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


class GatewayEnrollRequest(BaseModel):
    """Body for the email-only enrolment of a service's bearer token."""
    email: str


class GatewayProvisionRequest(GatewayEnrollRequest):
    """Body for the email-only gateway provision (no caregiver password)."""
    room: str = "Camera"
    icon_name: str = "tv"


async def _enroll_gateway_token(
    session: AsyncSession, authorization: str, email: str, now: datetime,
) -> tuple[UUID, UUID, bool]:
    """Tie the presented bearer token to a caregiver account, idempotently.

    Returns `(gateway_id, caregiver_id, already_enrolled)`.

    This is the ONE place a `public.gateways` row is written. It used to
    live inside provision-camera, which meant provisioning a camera was
    the only way a token could ever become known — so a service with
    nothing to provision (a bridge watching no cameras, or any box that
    had just rotated its token) could not enrol at all, and 401d on the
    roster forever. Both callers share this so the two paths cannot drift
    into disagreeing about what "enrolled" means.
    """
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_bearer_token",
        )
    box_token = authorization[len("bearer "):].strip()
    if not box_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="empty_bearer_token",
        )
    cg = (await session.execute(
        text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
        {"e": email.strip()},
    )).first()
    if cg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="caregiver_not_found",
        )
    caregiver_id = cg[0]

    # Only the hash is ever stored, and it is what the roster looks up. The
    # SELECT-then-INSERT is what makes a re-enrol land on the same row
    # instead of adding a second one for the same token.
    token_hash = hash_ingest_token(box_token)
    existing = (await session.execute(
        text("SELECT id FROM public.gateways WHERE token_hash = :h"),
        {"h": token_hash},
    )).first()
    if existing is None:
        gateway_id = (await session.execute(
            text("INSERT INTO public.gateways (caregiver_id, token_hash, created_at) "
                 "VALUES (:c, :h, :n) RETURNING id"),
            {"c": caregiver_id, "h": token_hash, "n": now},
        )).scalar_one()
        return gateway_id, caregiver_id, False
    gateway_id = existing[0]
    await session.execute(
        text("UPDATE public.gateways SET caregiver_id = :c WHERE token_hash = :h"),
        {"c": caregiver_id, "h": token_hash},
    )
    return gateway_id, caregiver_id, True


@router.post(
    "/gateway/enroll",
    status_code=status.HTTP_200_OK,
)
async def gateway_enroll(
    body: GatewayEnrollRequest,
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict:
    """Make this service's bearer token known to the backend. No camera.

    Same handshake as provision-camera — Bearer <the service's own token>
    plus the caregiver's email — minus the camera. That was the missing
    half: a token only ever got its `gateways` row by provisioning
    something, so rotating a credential meant creating a junk camera row,
    and a bridge with zero cameras could not enrol at all.

    Idempotent, because a service re-POSTs this on every boot. Returns
    neither the token nor its hash: the caller already holds the
    plaintext, and echoing it only spreads it into logs.
    """
    _gw, _cg, already = await _enroll_gateway_token(
        session, authorization, body.email, datetime.now(timezone.utc),
    )
    await session.commit()
    return {"ok": True, "already_enrolled": already}


@router.post(
    "/gateway/provision-camera",
    status_code=status.HTTP_201_CREATED,
)
async def gateway_provision_camera(
    body: GatewayProvisionRequest,
    session: SessionDep,
    authorization: str = Header(default=""),
) -> dict:
    """Provision a gateway camera in the field with EMAIL ONLY — no password.

    Tenets: a generic box, set up in a new home, phone-only, no password.
      - Authorization: Bearer <box's UNIQUE token> (generated on first boot).
      - body.email identifies the CAREGIVER (account) this box belongs to.
    We tie this box's token -> that caregiver (public.gateways), then create the
    camera under EVERY one of the caregiver's patients. The roster returns them
    by caregiver. Nothing account-specific is baked into the image — the box
    learns its account HERE, at setup.
    """
    now = datetime.now(timezone.utc)
    # Tie THIS box's unique token to the caregiver, same idempotent upsert
    # POST /v1/gateway/enroll performs on its own — provisioning a camera is
    # now one of two ways to enrol, not the only one.
    gateway_id, caregiver_id, _already = await _enroll_gateway_token(
        session, authorization, body.email, now,
    )

    # No patient lookup any more: a camera belongs to the account, so it can
    # be provisioned before the caregiver has added anyone. The old
    # caregiver_has_no_patients rejection is gone with it.

    # ONE row per physical camera (2026-08-19). This used to loop over every
    # patient of the caregiver, so a single Trueview camera in a shared room
    # became N rows publishing N streams — N× ingest, N× model spend on
    # identical footage, duplicate events, and no way to add a patient later.
    # The camera now belongs to the ACCOUNT and starts unassigned: it is
    # absent from the roster, so it does not publish until someone says who
    # it watches (see POST /v1/cameras/{id}/assign).
    cameras = []
    for camera_id in (uuid4(),):
        livekit_identity = f"cam_{camera_id.hex[:8]}"
        # Room follows the primary patient and is set at assignment time.
        livekit_room = None
        next_display_id = (await session.execute(
            select(func.coalesce(func.max(Camera.display_id), 0)).where(
                Camera.caregiver_id == caregiver_id)
        )).scalar_one() + 1
        ingest_token_plaintext, ingest_token_hash = mint_ingest_token()
        cam = Camera(
            id=camera_id,
            caregiver_id=caregiver_id,
            patient_id=None,
            display_id=next_display_id,
            room=body.room,
            icon_name=body.icon_name,
            online=False,
            created_at=now,
            pairing_code=None,
            pairing_expires_at=None,
            api_key_hash=None,
            livekit_identity=livekit_identity,
            livekit_room=livekit_room,
            state="open",
            no_people_count=0,
            state_changed_at=now,
            close_threshold=6,
            battery_percent=None,
            desired_state="disconnected",
            source_type="gateway",
            # Own this camera to THIS box, so the roster + heartbeat scope
            # per-gateway (not fleet-wide across the caregiver's boxes).
            gateway_id=gateway_id,
            pi_stream_url=None,
            pi_audio_url=None,
            ingest_token_hash=ingest_token_hash,
            ingest_token_created_at=now,
            app_present=True,
        )
        session.add(cam)
        await session.flush()
        cameras.append({
            # Always unassigned: the caregiver names it and says who it
            # watches in the app. One path, whether the account has one
            # patient or twenty.
            "patient_id": None,
            "camera_id": str(camera_id),
            "livekit_identity": livekit_identity,
            "livekit_room": livekit_room,
            "ingest_token": ingest_token_plaintext,
        })
    await session.commit()
    return {"caregiver_email": body.email, "provisioned": len(cameras),
            "cameras": cameras}
