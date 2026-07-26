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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import livekit.api as lkapi
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from livekit import rtc

from src.db.session import get_sessionmaker
from src.repos import face_embedding_repo
from src.services.reid.cross_clip_body_gallery import CrossClipBodyGallery
from src.services.reid.embeddings import embedding_from_bytes
from src.services.reid.sam_client import SamServiceClient
from src.services.identity.journal import EventJournal
from src.services.identity.loop import CameraLoop, IdentityHome
from src.settings import get_settings

log = logging.getLogger("run_identity")

GALLERY_REFRESH_S = 60.0
_HOME: IdentityHome | None = None   # set in main; read by the HTTP layer

app = FastAPI()


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
def state() -> JSONResponse:
    if _HOME is None:
        return JSONResponse({"ready": False})
    seat = _HOME.seat
    return JSONResponse({
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


def _mjpeg(cam_id: str):
    while True:
        sp = _HOME.cameras.get(cam_id) if _HOME else None
        jpg = sp.latest_jpeg if sp else None
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1 / 15)


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


@app.get("/timeline")
def timeline(n: int = 2160) -> JSONResponse:
    """Last n samples (default 6h at 10s) for the lane view."""
    if _TIMELINE is None:
        return JSONResponse({"samples": []})
    return JSONResponse({
        "samples": list(_TIMELINE)[-max(1, min(n, 8640)):],
        "names": dict(_HOME.cam_names) if _HOME else {},
    })


@app.get("/events")
def events(n: int = 200) -> JSONResponse:
    """Unsupervised-soak journal tail + confirm-latency stats — the
    soak dashboard's Identity section reads this."""
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
def snap(name: str):
    from fastapi.responses import Response

    if _HOME is None or _HOME.journal is None or "/" in name or ".." in name:
        return Response(status_code=404)
    p = _HOME.journal.snaps / name
    if not p.is_file():
        return Response(status_code=404)
    return Response(content=p.read_bytes(), media_type="image/jpeg")


@app.get("/stream/{cam_id}")
def stream(cam_id: str) -> StreamingResponse:
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


INTAKE_FPS = 10.0    # convert/push at most this many frames per second
INTAKE_MAX_W = 1280  # resize cap at intake (0 = native, no resize)
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
            asyncio.ensure_future(cam_loop.run())
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
    while True:
        await asyncio.sleep(3600)


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
    home = IdentityHome(
        patient_id=patient_id,
        sam=sam,
        body_gallery=CrossClipBodyGallery(settings=settings),
        face_matrix=face_matrix,
        journal=EventJournal(Path("out/identity_soak")),
        cam_names=cam_names,
    )
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
    ap.add_argument("--preset", choices=["yesterday"], default=None,
                    help="'yesterday' = faithful replica of the 2026-07-23 "
                         "breakthrough run: 40-frame/12fps batches, ZERO "
                         "face/identity work, full-res frames. Use with the "
                         "pod at NEMARA_EDGE_TRACK_W=640. One-lever baseline.")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--faces", action="store_true",
                    help="ladder step 3a: keep the parity tracking knobs "
                         "but re-enable out-of-band identity probes "
                         "(face/body scoring, seat, harvest)")
    args = ap.parse_args()

    if args.preset == "yesterday":
        global INTAKE_FPS, INTAKE_MAX_W, STREAM_CAPACITY
        from src.services.identity import loop as loop_mod
        loop_mod.TARGET_FPS = 12.0
        loop_mod.MAX_BATCH = 40
        loop_mod.FACES_ENABLED = False       # zero identity work
        loop_mod.TRACK_SEND_W = 10**9        # never downscale (full res)
        loop_mod.ACTIVE_INTERVAL_S = 0.6
        INTAKE_FPS = 15.0                    # consume everything (83ms strides)
        INTAKE_MAX_W = 0                     # native resolution
        STREAM_CAPACITY = 0                  # unbounded queue: perfectly
        # smooth deliveries, accepted OOM risk FOR THIS TEST ONLY
        loop_mod.CameraLoop.RING_MAX = 300   # the viewer's 20s ring — the
        # 45-frame production ring overflowed once/session at 15fps
        # (each overflow silently breaks the carry chain)
        loop_mod.BATCH_MAX_SPAN_S = 1e9      # span guard OFF — the viewer
        # has none. At native res the pod runs ~165ms/frame, so steady
        # state is 40-frame/~7s batches; an 8s guard fires EVERY cycle
        # and tail-ships (the exact blind-gap bug). Contiguity > latency.
        if args.faces:
            loop_mod.FACES_ENABLED = True    # step 3a: identity rides the
            # proven tracking spine, out-of-band (/faces + local OSNet)
        print(f"[preset] YESTERDAY replica: 40f/12fps, "
              f"faces={'ON' if args.faces else 'off'}, native res, "
              f"unbounded queue, 15fps intake", flush=True)

    threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=args.port, log_level="warning",
        ),
        daemon=True,
    ).start()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
