"""Run the identity service — continuous tracking + sparse identity, live.

The production form of the live-viewer experiment (2026-07-24 plan P1+P2):
per-camera continuous batches → pod EdgeTAM session → screen veto → track
state → three-rule identity (calibrated scoring / single seat / solo
harvest) → live MJPEG overlay + /state JSON.

Prereqs: pod in EdgeTAM mode via the SSH tunnel (scripts/pod.sh health),
LiveKit creds in env (NEMARA_LIVEKIT_URL/_API_KEY/_API_SECRET — or in
.env, which Settings loads), DB reachable (face enrollment gallery).
Stop the old live_tracker_view.py first — the identity service replaces it.

    uv run python tools/run_identity.py --room patient_<id>
    # open http://localhost:8012/       (all cameras + seat state)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from uuid import UUID

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))          # src.* (DB/repos/settings)
sys.path.insert(0, str(_repo / "gpu"))  # identity.* (this package)

import cv2
import livekit.api as lkapi
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from livekit import rtc

from src.db.session import get_sessionmaker
from src.repos import face_embedding_repo
from identity.gallery import CrossClipBodyGallery
from identity.matching import embedding_from_bytes
from identity.pod_client import SamServiceClient
from identity.journal import EventJournal
from identity.loop import CameraLoop, IdentityHome
from src.settings import get_settings

log = logging.getLogger("run_identity")

GALLERY_REFRESH_S = 60.0
_HOME: IdentityHome | None = None   # set in main; read by the HTTP layer

app = FastAPI()

# /state auth (contract.py): empty = no auth (local dev).
#
# ONE name across the seam: NEMARA_IDENTITY_STATE_TOKEN. The backend
# has always read it under that name (settings.identity_state_token +
# the NEMARA_ prefix); this side used to read NEMARA_IDENTITY_TOKEN,
# so setting either one alone left the OTHER side with an empty token
# — and an empty token here means auth_ok() waves everything through.
# Silent no-auth, no error anywhere. The legacy name is still accepted
# so an already-provisioned box keeps working, and says so loudly.
_STATE_TOKEN = os.environ.get("NEMARA_IDENTITY_STATE_TOKEN", "")
if not _STATE_TOKEN:
    _legacy = os.environ.get("NEMARA_IDENTITY_TOKEN", "")
    if _legacy:
        log.warning(
            "using deprecated NEMARA_IDENTITY_TOKEN; rename it to "
            "NEMARA_IDENTITY_STATE_TOKEN (the name the backend reads)"
        )
        _STATE_TOKEN = _legacy


@app.get("/")
def index() -> HTMLResponse:
    cams = sorted(_HOME.cameras) if _HOME else []
    imgs = "".join(
        f'<div style="margin:8px"><img src="/stream/{c}" '
        f'style="max-width:96vw"></div>' for c in cams
    )
    return HTMLResponse(
        "<html><head><title>identity</title></head>"
        '<body style="margin:0;background:#0b0b0d;color:#ddd;'
        f'font-family:monospace">{imgs or "no cameras yet"}</body></html>'
    )


@app.get("/state")
def state(request: Request) -> JSONResponse:
    from identity.contract import STATE_CONTRACT, auth_ok

    if not auth_ok(
        request.headers.get("authorization"), _STATE_TOKEN,
    ):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if _HOME is None:
        return JSONResponse({"contract": STATE_CONTRACT, "ready": False})
    seat = _HOME.seat
    return JSONResponse({
        "contract": STATE_CONTRACT,
        "ready": True,
        "seat": None if seat is None else {
            "camera": seat.camera_id,
            "camera_name": _HOME.name_of(seat.camera_id),
            "track": seat.track_id,
            "score": round(seat.score, 3), "anchor": seat.anchor,
            "face_confirmed": seat.face_confirmed,
            "since": seat.since, "confirmed_at": seat.confirmed_at,
        },
        "cameras": {
            cam_id: {
                "name": _HOME.name_of(cam_id),
                "tracks": {
                    tid: {"bbox": tv.bbox, "age_s": round(tv.age(now=time.time()), 1)}
                    for tid, tv in sp.state.tracks.items()
                },
                "solo": sp.state.solo,
                "lag_s": round(sp.lag_s, 2),
                "pod_ms": round(sp.pod_ms, 0),
                "stage_ms": sp.stage_ms,     # measured per-batch stages
                "probe_ms": sp.probe_ms,     # measured per-probe stages
            }
            for cam_id, sp in _HOME.cameras.items()
        },
    })


async def _mjpeg(cam_id: str):
    # ASYNC on purpose (2026-07-27): the sync version ran in FastAPI's
    # ~40-thread pool, one thread pinned per open stream — and when a
    # camera had no frame yet it never yielded, so a disconnected
    # client was never detected and the thread leaked forever. Browser
    # tabs re-opening streams across service restarts exhausted the
    # pool, wedging every sync endpoint (/, /state) while the async
    # loop ran fine. Async generators cost no threads and Starlette
    # cancels them on client disconnect even before the first yield.
    while True:
        sp = _HOME.cameras.get(cam_id) if _HOME else None
        jpg = sp.latest_jpeg if sp else None
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        await asyncio.sleep(1 / 15)


_TIMELINE: "deque" = None  # (ts, {cam: 'patient'|'people'|'empty'})


async def _timeline_sampler() -> None:
    """Every 10s, one status sample per camera — the identity-lane
    timeline (24h ring), the at-a-glance view the legacy dashboard
    lanes provided before the backend stopped."""
    global _TIMELINE
    from collections import deque as _dq

    _TIMELINE = _dq(maxlen=8640)             # 24h at 10s
    while True:
        await asyncio.sleep(10.0)
        try:
            if _HOME is None:
                continue
            seat = _HOME.seat
            row = {}
            for cam_id, sp in _HOME.cameras.items():
                if seat is not None and seat.camera_id == cam_id:
                    row[cam_id] = "patient"
                elif sp.state.tracks:
                    row[cam_id] = "people"
                else:
                    row[cam_id] = "empty"
            _TIMELINE.append((round(time.time()), row))
        except Exception:                     # noqa: BLE001
            log.exception("identity.timeline_sample_failed")


@app.get("/probes")
def probes(request: Request, n: int = 1600) -> JSONResponse:
    """Every identity probe's full scores (faces raw cos+px, body raw
    cos, fused candidates), newest-last per camera — the soak
    dashboard's tick-correlated analysis feed (Aryaman 2026-07-28).

    Authenticated like /state: this payload is per-track similarity to
    the patient's enrolled FACE AND BODY gallery — biometric-derived
    data, and more sensitive than /state's seat summary.
    """
    from identity.contract import auth_ok

    if not auth_ok(request.headers.get("authorization"), _STATE_TOKEN):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if _HOME is None:
        return JSONResponse({"probes": {}, "names": {}})
    return JSONResponse({
        "probes": {
            cam_id: list(sp.probe_history)[-max(1, min(n, 4000)):]
            for cam_id, sp in _HOME.cameras.items()
        },
        "names": dict(_HOME.cam_names),
    })


@app.get("/timeline")
def timeline(request: Request, n: int = 2160) -> JSONResponse:
    """Last n samples (default 6h at 10s) for the lane view."""
    from identity.contract import auth_ok

    if not auth_ok(request.headers.get("authorization"), _STATE_TOKEN):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if _TIMELINE is None:
        return JSONResponse({"samples": []})
    return JSONResponse({
        "samples": list(_TIMELINE)[-max(1, min(n, 8640)):],
        "names": dict(_HOME.cam_names) if _HOME else {},
    })


@app.get("/events")
def events(request: Request, n: int = 200) -> JSONResponse:
    """Unsupervised-soak journal tail + confirm-latency stats — the
    soak dashboard's Identity section reads this."""
    from identity.contract import auth_ok

    if not auth_ok(request.headers.get("authorization"), _STATE_TOKEN):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if _HOME is None or _HOME.journal is None:
        return JSONResponse({"events": [], "confirm_stats": []})
    lats = [round(v, 1) for _a, v in _HOME.confirm_stats]
    return JSONResponse({
        "names": dict(_HOME.cam_names),
        "events": _HOME.journal.tail(max(1, min(n, 1000))),
        "confirm_stats": [
            {"anchor": a, "latency_s": round(v, 1)}
            for a, v in _HOME.confirm_stats[-50:]
        ],
        "confirm_median_s": (
            round(sorted(lats)[len(lats) // 2], 1) if lats else None
        ),
        "confirm_max_s": (max(lats) if lats else None),
    })


@app.get("/snap/{name}")
def snap(name: str, request: Request):
    """Journal snapshot JPEGs — authenticated, because these are IMAGE
    data cropped from the camera, same exposure class as /stream. Every
    endpoint on this service is now behind the token; the split where
    only some were made the token look like a control it wasn't."""
    from fastapi.responses import Response

    from identity.contract import auth_ok

    if not auth_ok(request.headers.get("authorization"), _STATE_TOKEN):
        return Response(status_code=401)
    if _HOME is None or _HOME.journal is None or "/" in name or ".." in name:
        return Response(status_code=404)
    p = _HOME.journal.snaps / name
    if not p.is_file():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


@app.get("/stream/{cam_id}")
def stream(cam_id: str, request: Request):
    """Annotated live view — DEV/DEBUG ONLY.

    This is raw camera video with tracking boxes and carries NO room
    privacy gating: it would show a bathroom or bedroom exactly as the
    camera sees it. Authenticated like /state, and it must never be
    reachable in production — the prod build has no live-feed path
    that bypasses the privacy rules, and this endpoint must not become
    one (Aryaman 2026-07-29).
    """
    from identity.contract import auth_ok

    if not auth_ok(request.headers.get("authorization"), _STATE_TOKEN):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return StreamingResponse(
        _mjpeg(cam_id), media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── LiveKit intake (the proven viewer pattern) ───────────────────────
def _mint_token(key: str, secret: str, room: str) -> str:
    import datetime as _dt

    return (
        lkapi.AccessToken(key, secret)
        # LONG TTL (2026-07-25): the SDK re-presents the ORIGINAL token
        # on auto-reconnect. A default-TTL token expired overnight, a
        # network blip hit at 12:23, reconnect got 401 → "not
        # retrying" → the service ran blind for 6h on a frozen frame.
        .with_ttl(_dt.timedelta(days=14))
        .with_identity("identity-service")
        .with_name("Identity Service")
        .with_grants(lkapi.VideoGrants(
            room_join=True, room=room, can_subscribe=True, can_publish=False,
        ))
        .to_jwt()
    )


INTAKE_FPS = 15.0    # consume the stream fully (~66ms strides); the
                     # 3fps batch decimation downstream picks evenly
                     # from a dense ring — production value, soaked
INTAKE_MAX_W = 0     # native at intake (Aryaman 2026-07-26: resolution
                     # is not the cost lever; frame count is)
MOTION_SAMPLE_S = 0.5   # motion-latch sampling cadence (CPU, ~free)
MOTION_DIFF_THR = 3.0   # mean abs diff on a 64px gray thumbnail; static-
                        # camera noise floor is ~1-2, a person entering
                        # is well above — tune from motion_latched logs
STREAM_CAPACITY = 2  # SDK frame queue bound (0 = unbounded — smooth but
                     # OOM-risky; the 2026-07-23 stable run used unbounded)


async def _read_video(track: rtc.Track, cam_loop: CameraLoop) -> None:
    # Bounded queue (drop-oldest) by default: unbounded ballooned to 29k
    # queued raw frames when intake stalled 2026-07-24. The yesterday
    # preset restores unbounded for faithful replication (smoothness).
    stream = rtc.VideoStream(track, capacity=STREAM_CAPACITY)
    last_push = 0.0
    last_motion_sample = 0.0
    prev_thumb = None
    async for ev in stream:
        # One bad frame must never kill the reader — a dead reader
        # freezes the ring, which freezes tracks, which would let a
        # frozen track hold the patient seat (the ghost-seat hole; the
        # the loop's stale-intake watchdog is the second net).
        try:
            # Decimate BEFORE converting: at 15fps × 2 cameras the
            # RGBA→BGR + resize on the event loop is the intake cost;
            # the ring only needs ~INTAKE_FPS anyway.
            now = time.time()
            # Accumulator gate, not wall-clock gate: `now - last < 1/fps`
            # against a ~15fps camera drops every frame arriving a hair
            # early — effective intake fell to ~10fps and the 300-frame
            # ring silently held 30s instead of 20s (2026-07-24). The
            # accumulator admits true INTAKE_FPS; max() stops it lagging
            # into the past when frames pause.
            if now < last_push + 1.0 / INTAKE_FPS:
                continue
            frame = ev.frame
            if frame.type != rtc.VideoBufferType.RGBA:
                frame = frame.convert(rtc.VideoBufferType.RGBA)
            arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(
                frame.height, frame.width, 4,
            )
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            # ONE native-res frame kept per camera for the identity
            # probe — the camera is ~3400px wide and the 1280 ring
            # resize was crushing faces to ~22px (measured 2026-07-24:
            # probe found faces but they scored as untrustworthy).
            cam_loop.latest_native = (now, bgr)
            # Bound memory + pod upload (INTAKE_MAX_W=0 = ship native,
            # the yesterday-replica behavior).
            if INTAKE_MAX_W and bgr.shape[1] > INTAKE_MAX_W:
                s = INTAKE_MAX_W / bgr.shape[1]
                bgr = cv2.resize(bgr, (INTAKE_MAX_W, int(bgr.shape[0] * s)))
            # Frame indexing is owned by the cam_loop (survives re-subscribe).
            cam_loop.push_frame(now, bgr)
            last_push = max(last_push + 1.0 / INTAKE_FPS, now - 1.0 / INTAKE_FPS)
            # Motion latch for the empty-room economy: diff a tiny gray
            # thumbnail ~2x/s; any real change arms the camera's next
            # empty-room poll. Latched (never cleared here) so an entry
            # between polls can't be missed.
            if now - last_motion_sample >= MOTION_SAMPLE_S:
                thumb = cv2.cvtColor(
                    cv2.resize(bgr, (64, 36)), cv2.COLOR_BGR2GRAY,
                ).astype(np.int16)
                if prev_thumb is not None:
                    diff = float(np.abs(thumb - prev_thumb).mean())
                    if diff >= MOTION_DIFF_THR and not cam_loop.motion_pending:
                        cam_loop.motion_pending = True
                        log.info(
                            "identity.motion_latched cam=%s diff=%.1f",
                            cam_loop.camera_id, diff,
                        )
                prev_thumb = thumb
                last_motion_sample = now
        except Exception:                     # noqa: BLE001
            log.exception(
                "identity.frame_intake_failed cam=%s", cam_loop.camera_id,
            )


async def _run_livekit(
    url: str, token: str, home: IdentityHome, only_cam: str = "",
) -> None:
    room = rtc.Room()

    reader_tasks: dict[str, asyncio.Task] = {}
    loop_tasks: dict[str, asyncio.Task] = {}

    def _remove_camera(cam_id: str, why: str) -> None:
        """Tear a camera loop down when its publisher leaves. Without
        this, a deleted camera haunts home.cameras (and the dashboard +
        /state) until the next process restart (2026-07-28: ghost
        3d60d66b). Seat safety is free: removal drops the camera's
        tracks from the live set, so a seat it held dies via the normal
        seat.py not-in-live path."""
        cams = dict(home.cameras)
        cam_loop = cams.pop(cam_id, None)
        if cam_loop is None:
            return
        # Swap-not-mutate, same as the create path.
        home.cameras = cams
        for tasks in (reader_tasks, loop_tasks):
            t = tasks.pop(cam_id, None)
            if t is not None and not t.done():
                t.cancel()
        # The probe loop is owned by the CameraLoop itself (it starts
        # its own task), so popping the loop task does not stop it: it
        # kept running every 0.5s forever and held the camera's frame
        # ring alive with it. The ring holds native-resolution frames,
        # so each leaked camera costs hundreds of MB that is never
        # reclaimed — and camera churn (a flapping gateway) leaks one
        # per reconnect.
        probe = getattr(cam_loop, "_probe_task", None)
        if probe is not None and not probe.done():
            probe.cancel()
        cam_loop.ring.clear()
        log.info("identity.camera_removed cam=%s (%s)", cam_id, why)

    @room.on("participant_disconnected")
    def _on_participant_gone(participant):  # noqa: ANN001
        ident = participant.identity
        if ident.startswith("cam_"):
            _remove_camera(
                ident.removeprefix("cam_"), "participant_disconnected",
            )

    @room.on("disconnected")
    def _on_disconnected(*_a):  # noqa: ANN001
        # RESILIENCE (2026-07-27 wifi-glitch lesson): the SDK gives up
        # after ~10 reconnect attempts and leaves a roomless zombie —
        # the seat provider for the whole backend, silently dead. Die
        # instead: the supervisor restarts us in 3s, forever, and a
        # fresh boot rebuilds LiveKit + pod client + galleries from
        # scratch. The supervisor IS the infinite reconnection loop.
        log.error("identity.room_disconnected — exiting for supervisor restart")
        os._exit(1)

    @room.on("track_subscribed")
    def _on_sub(track, pub, participant):  # noqa: ANN001
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        ident = participant.identity          # e.g. cam_3c6570af
        if not ident.startswith("cam_"):
            return
        cam_id = ident.removeprefix("cam_")
        if only_cam and only_cam not in cam_id:
            log.info("identity.camera_skipped cam=%s (--only-cam)", cam_id)
            return
        cam_loop = home.cameras.get(cam_id)
        if cam_loop is None:
            cam_loop = CameraLoop(camera_id=cam_id, home=home)
            # Swap-not-mutate: live_keys()/HTTP handlers iterate this
            # dict from other threads; replace the ref atomically.
            cams = dict(home.cameras)
            cams[cam_id] = cam_loop
            home.cameras = cams
            loop_tasks[cam_id] = asyncio.ensure_future(cam_loop.run())
            log.info("identity.camera_started cam=%s", cam_id)
        # ONE reader per camera: a re-subscribe replaces the old reader
        # instead of racing it (double intake = double convert cost +
        # ring churn).
        old = reader_tasks.get(cam_id)
        if old is not None and not old.done():
            old.cancel()
        t = asyncio.ensure_future(_read_video(track, cam_loop))
        reader_tasks[cam_id] = t

        def _on_reader_done(task: asyncio.Task, cam: str = cam_id) -> None:
            if task.cancelled():
                return                        # replaced on re-subscribe
            exc = task.exception()
            log.warning(
                "identity.reader_exited cam=%s exc=%s — awaiting re-subscribe",
                cam, exc,
            )

        t.add_done_callback(_on_reader_done)

    await room.connect(url, token, rtc.RoomOptions(auto_subscribe=True))
    log.info(
        "identity.connected participants=%s",
        [p.identity for p in room.remote_participants.values()],
    )
    # Missed-subscription recovery walk (2026-07-29): auto_subscribe
    # covers the happy path, but a gateway restart can re-publish
    # tracks without a track_subscribed callback ever reaching us —
    # live today at 11:16: participant present, publications listed,
    # intake frozen (15s+ lag) until a manual service restart. Same
    # recovery the backend room_worker has ("walking for missed
    # track_subscribed"): every 10s, any cam_* video publication that
    # is unsubscribed — or subscribed while that camera's intake has
    # gone quiet — gets its subscription toggled.
    walk_every_s = 10.0
    intake_quiet_s = 15.0
    # Consecutive passes a camera has looked unhealthy. Re-subscribing
    # that keeps not working means the room object itself is wrong, so
    # escalate to the supervisor for a clean rebuild instead of warning
    # every 10s forever — the same doctrine as the pod watchdog and the
    # room-disconnect handler above.
    walk_strikes: dict[str, int] = {}
    walk_max_strikes = 6
    while True:
        await asyncio.sleep(walk_every_s)
        try:
            now = time.time()
            for p in list(room.remote_participants.values()):
                if not p.identity.startswith("cam_"):
                    continue
                cam_id = p.identity.removeprefix("cam_")
                if only_cam and only_cam not in cam_id:
                    continue
                sp = home.cameras.get(cam_id)
                last_ts = (
                    sp.ring[-1][1] if sp is not None and sp.ring else 0.0
                )
                quiet = now - last_ts > intake_quiet_s
                healthy = True
                for pub in p.track_publications.values():
                    # Per-publication: one failing subscribe must not
                    # skip the remaining cameras for the whole pass.
                    try:
                        if pub.kind != rtc.TrackKind.KIND_VIDEO:
                            continue
                        if not pub.subscribed:
                            healthy = False
                            log.warning(
                                "identity.walk_resubscribe cam=%s "
                                "(publication not subscribed)", cam_id,
                            )
                            pub.set_subscribed(True)
                        elif quiet:
                            healthy = False
                            # Re-issue subscribe ONLY — never
                            # unsubscribe first. Unsubscribing ends the
                            # video stream WITHOUT waking the reader
                            # task waiting on its queue, so the reader
                            # hangs and only a later track_subscribed
                            # callback can clean it up — the very
                            # callback this walk exists because it
                            # can't be trusted. Re-subscribing is
                            # harmless when already subscribed, and is
                            # exactly what the backend's room_worker
                            # does; the toggle version could create the
                            # outage it was meant to repair.
                            log.warning(
                                "identity.walk_resubscribe cam=%s "
                                "(subscribed but intake quiet %.0fs)",
                                cam_id, now - last_ts,
                            )
                            pub.set_subscribed(True)
                    except Exception:  # noqa: BLE001
                        healthy = False
                        log.exception(
                            "identity.walk_publication_failed cam=%s", cam_id,
                        )
                if healthy:
                    walk_strikes.pop(cam_id, None)
                    continue
                walk_strikes[cam_id] = walk_strikes.get(cam_id, 0) + 1
                if walk_strikes[cam_id] >= walk_max_strikes:
                    log.error(
                        "identity.walk_exhausted cam=%s — %d passes without "
                        "healthy intake; exiting for supervisor restart",
                        cam_id, walk_strikes[cam_id],
                    )
                    os._exit(5)
        except Exception:   # noqa: BLE001 — the walk must never kill the service
            log.exception("identity.walk_failed")


# ── Face gallery (persistent DB enrollment) ──────────────────────────
async def _load_face_matrix(patient_id: UUID) -> np.ndarray:
    sm = get_sessionmaker()
    async with sm() as session:
        rows = await face_embedding_repo.list_ready_for_patient_unscoped(
            session, patient_id,
        )
    embs = [embedding_from_bytes(r.embedding) for r in rows]
    if not embs:
        log.warning("identity.face_gallery_empty patient=%s", patient_id)
        return np.zeros((0, 512), dtype=np.float32)
    return np.stack(embs).astype(np.float32)


async def _refresh_gallery(home: IdentityHome) -> None:
    while True:
        await asyncio.sleep(GALLERY_REFRESH_S)
        try:
            home.face_matrix = await _load_face_matrix(home.patient_id)
        except Exception:                     # noqa: BLE001
            log.exception("identity.gallery_refresh_failed")


async def _amain(args: argparse.Namespace) -> None:
    global _HOME
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
    )
    settings = get_settings()
    url = os.environ.get("NEMARA_LIVEKIT_URL") or settings.livekit_url
    key = os.environ.get("NEMARA_LIVEKIT_API_KEY") or settings.livekit_api_key
    secret = (
        os.environ.get("NEMARA_LIVEKIT_API_SECRET")
        or settings.livekit_api_secret
    )
    patient_id = UUID(args.patient)

    sam = SamServiceClient(
        base_url=args.pod_url, timeout_s=25.0,
        expected_face_model="buffalo_l",
    )
    # DB may still be starting (post-reboot: Docker up before Supabase —
    # the 2026-07-24 outage). Start with an empty gallery rather than
    # dying; the 60s refresher heals it as soon as the DB answers.
    try:
        face_matrix = await _load_face_matrix(patient_id)
    except Exception:                         # noqa: BLE001
        log.exception(
            "identity.gallery_load_failed at startup — starting with an "
            "EMPTY face gallery; the %ds refresher will retry",
            int(GALLERY_REFRESH_S),
        )
        face_matrix = np.zeros((0, 512), dtype=np.float32)
    cam_names: dict = {}
    try:
        from sqlalchemy import text as _sql_text
        sm = get_sessionmaker()
        async with sm() as session:
            rows = (await session.execute(_sql_text(
                "select substr(id::text, 1, 8) as c8, room from cameras"
            ))).all()
            cam_names = {r.c8: r.room for r in rows}
    except Exception:                        # noqa: BLE001 — names are
        log.exception("identity.cam_names_failed")  # cosmetic; ids ok
    async def _pod_progress_watchdog(home_ref):
        """Link-health watchdog. `last_pod_ok` is stamped by every
        successful pod round-trip — real batches when the room is
        active, the heartbeat below when it is idle — so staleness has
        exactly ONE meaning: the pod path (tunnel, connection pool,
        pod process) is broken. Exit; the supervisor restart rebuilds
        the client. The heartbeat rides the SAME SamServiceClient the
        batches use on purpose: a poisoned pool (the 2026-07-27
        incident) must fail the heartbeat too, or it goes undetected.
        v1 gated on "frames fresh + no pod success", which conflated
        idle (motion-gated rooms send no batches) with wedged — quiet
        rooms crash-looped every ~2.5 min, wiping the body pool."""
        POD_WEDGE_S = 120.0
        HEARTBEAT_S = 30.0
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            try:
                await home_ref.sam.health()
                home_ref.last_pod_ok = time.time()
            except Exception as e:               # noqa: BLE001
                log.warning("identity.pod_heartbeat_failed: %s", e)
            now = time.time()
            last_ok = getattr(home_ref, "last_pod_ok", None)
            if last_ok is not None and now - last_ok > POD_WEDGE_S:
                log.error(
                    "identity.pod_wedged — no successful pod contact "
                    "(batch or heartbeat) for %.0fs; exiting for "
                    "supervisor restart",
                    now - last_ok,
                )
                os._exit(4)

    # Extra pod workers. The pod serialises everything behind a
    # process-global lock, so a second WORKER (own process, own lock,
    # own model copy) is what actually overlaps two cameras — measured
    # 1.5-1.9x with per-call compute unchanged. Unset the variable and
    # every camera goes back to `sam`; that is the whole revert.
    extra = [
        u.strip() for u in os.environ.get("NEMARA_POD_URLS", "").split(",")
        if u.strip() and u.strip().rstrip("/") != args.pod_url.rstrip("/")
    ]
    pods = [
        SamServiceClient(
            base_url=u, timeout_s=25.0, expected_face_model="buffalo_l",
        )
        for u in extra
    ]
    if pods:
        log.info(
            "identity.pod_pool workers=%d urls=%s",
            len(pods) + 1, [args.pod_url, *extra],
        )

    home = IdentityHome(
        patient_id=patient_id,
        sam=sam,
        pods=pods,
        body_gallery=CrossClipBodyGallery(settings=settings),
        face_matrix=face_matrix,
        journal=EventJournal(Path("out/identity_soak")),
        cam_names=cam_names,
    )
    home.last_pod_ok = time.time()   # watchdog baseline: boot counts
    _HOME = home
    log.info(
        "identity.up room=%s pod=%s gallery=%d refs — http://localhost:%d/",
        args.room, args.pod_url, len(home.face_matrix), args.port,
    )
    await asyncio.gather(
        _run_livekit(
            url, _mint_token(key, secret, args.room), home,
            only_cam=args.only_cam,
        ),
        _refresh_gallery(home),
        _timeline_sampler(),
        _pod_progress_watchdog(home),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", required=True, help="LiveKit room, patient_<id>")
    ap.add_argument(
        "--patient", default="75a30940-b5e2-4608-afa3-b31a1e1fcda6",
        help="patient UUID (face enrollment gallery)",
    )
    ap.add_argument("--pod-url", default="http://localhost:8000")
    ap.add_argument("--only-cam", default="",
                    help="substring filter: subscribe ONLY matching cameras "
                         "(e.g. 9271358c to pause all others)")
    ap.add_argument("--port", type=int, default=8012)
    args = ap.parse_args()
    # Production config is the ONLY config (2026-07-26): every
    # experiment flag (--preset/--faces/--track-width/--batch-fps/
    # --queue-cap) was deleted once its winning value was baked into
    # the constants (loop.py + the intake constants above). The
    # experiments and their measurements live in git history and
    # docs/plans/2026-07-26-identity-productionization.md.

    # 0.0.0.0 with no token = /state, /probes and the /stream debug feed
    # are open to anything that can reach the box. On the dev rig that is
    # the LAN; colocation puts it on a rented GPU host. Say it out loud
    # at every boot rather than leaving it to be discovered.
    if not _STATE_TOKEN:
        log.warning(
            "NEMARA_IDENTITY_STATE_TOKEN is unset — /state, /probes and "
            "/stream are UNAUTHENTICATED on 0.0.0.0:%d. Set it here and "
            "on the backend (same name) before any non-dev network.",
            args.port,
        )

    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=args.port, log_level="warning",
        ),
        daemon=True,
    ).start()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
