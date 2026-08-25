"""Run the identity service — continuous tracking + sparse identity, live.

The production form of the live-viewer experiment (2026-07-24 plan P1+P2):
per-camera continuous batches → pod EdgeTAM session → screen veto → track
state → three-rule identity (calibrated scoring / single seat / solo
harvest) → live MJPEG overlay + /state JSON.

Prereqs: pod in EdgeTAM mode via the SSH tunnel (scripts/pod/pod.sh health),
LiveKit creds in env (NEMARA_LIVEKIT_URL/_API_KEY/_API_SECRET — or in
.env, which Settings loads), DB reachable (face enrollment gallery).
Stop the old live_tracker_view.py first — the identity service replaces it.

    uv run python tools/run_identity.py --room patient_<id>
    # open http://localhost:8012/       (all cameras + seat state)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
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


@app.post("/admin/exit")
def admin_exit(request: Request) -> JSONResponse:
    """Exit(0) shortly after replying; the entrypoint supervisor
    relaunches this room (fresh dev code first under NEMARA_DEV_CODE=1
    — experiment-node iteration without an image build). Same auth as
    /state."""
    from identity.contract import auth_ok

    if not auth_ok(
        request.headers.get("authorization"), _STATE_TOKEN,
    ):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    def _die() -> None:
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=_die, daemon=True).start()
    return JSONResponse({"status": "restarting"})


@app.get("/state")
def state(request: Request) -> JSONResponse:
    from identity.contract import STATE_CONTRACT, auth_ok

    if not auth_ok(
        request.headers.get("authorization"), _STATE_TOKEN,
    ):
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if _HOME is None:
        return JSONResponse({"contract": STATE_CONTRACT, "ready": False})
    return JSONResponse({
        "contract": STATE_CONTRACT,
        "ready": True,
        # One entry per LOCATED resident, keyed by patient uuid. A
        # resident who is absent from this map is not located — which
        # consumers must read as "unknown", never as another resident's
        # seat and never as a room default.
        "seats": {
            str(pid): {
                "camera": seat.camera_id,
                "camera_name": _HOME.name_of(seat.camera_id),
                "track": seat.track_id,
                "score": round(seat.score, 3), "anchor": seat.anchor,
                "face_confirmed": seat.face_confirmed,
                "since": seat.since, "confirmed_at": seat.confirmed_at,
            }
            for pid, seat in list(_HOME.seats.items())
        },
        "cameras": {
            cam_id: {
                "name": _HOME.name_of(cam_id),
                "tracks": {
                    tid: {
                        "bbox": tv.bbox,
                        "age_s": round(tv.age(now=time.time()), 1),
                        "seen_s": round(time.time() - tv.last_seen, 1),
                        "bbox_series": [
                            [round(ts, 3), list(bb)]
                            for ts, bb in tv.series[-120:]
                        ],
                    }
                    for tid, tv in sp.state.tracks.items()
                },
                "solo": sp.state.solo,
                "lag_s": round(sp.lag_s, 2),
                "latch": {
                    k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in getattr(sp, "latch_telemetry", {}).items()
                },
                "pod_ms": round(sp.pod_ms, 0),
                "stage_ms": sp.stage_ms,     # measured per-batch stages
                "probe_ms": sp.probe_ms,     # measured per-probe stages
                # Is this loop actually WORKING, as distinct from being
                # alive? A camera watching an empty room and a camera
                # whose every pod call throws both report zero tracks;
                # only these tell them apart. `since_ok_s` is the one to
                # alarm on — counters climb forever and say nothing
                # about now.
                "tick_health": {
                    "ok": sp.step_ok,
                    "failed": sp.step_fail,
                    "since_ok_s": (
                        round(time.time() - sp.last_step_ok_at, 1)
                        if sp.last_step_ok_at else None
                    ),
                    "last_error": sp.last_step_error or None,
                },
            }
            for cam_id, sp in _HOME.cameras.items()
        },
    })


_LINK_STATS_EVERY_S = float(
    os.environ.get("NEMARA_LINK_STATS_EVERY_S", "20"))
_LINK_STATS_TASK = None


async def _log_link_stats(room) -> None:
    """Why the SFU keeps handing us a smaller picture than we asked for.

    Measured 2026-08-25: we request HIGH on every layer request and are
    still given 320x180 — so the downshift is not self-inflicted, and the
    publisher's own encoder reports no limitation on any layer. That
    leaves the path to THIS subscriber, and nothing was measuring it.

    `available_incoming_bitrate` is the number that decides it: LiveKit
    downshifts a subscriber when its estimate of that falls below what
    the layer costs. If it sits comfortably above the stream's bitrate
    while we are being downshifted anyway, bandwidth is NOT the reason
    and the next place to look is the SFU's own policy.

    Read-only, every 20s, from the connection the service already holds
    — a second subscriber would compete for the very bandwidth being
    measured and change the answer.
    """
    while True:
        await asyncio.sleep(_LINK_STATS_EVERY_S)
        try:
            if not room.isconnected():
                continue
            stats = await room.get_rtc_stats()
            # We are the SUBSCRIBER. The stats are split by peer
            # connection and the publisher half describes what this
            # process sends, which is nothing.
            # The SDK splits stats by peer connection and 1.1.14 does not
            # always populate the half you expect, so take whichever
            # carries entries rather than assuming. Reported below so a
            # future empty result is legible instead of mysterious.
            _pub = list(getattr(stats, "publisher_stats", None) or [])
            _sub = list(getattr(stats, "subscriber_stats", None) or [])
            _src = _sub or _pub
            pair, video = None, []
            for s in _src:
                which = s.WhichOneof("stats")
                if which == "candidate_pair":
                    cp = s.candidate_pair.candidate_pair
                    # The nominated pair is the one actually carrying
                    # traffic; the others are stale ICE candidates whose
                    # numbers describe a path we are not using.
                    if cp.nominated or pair is None:
                        pair = cp
                elif which == "inbound_rtp":
                    if s.inbound_rtp.stream.kind == "video":
                        video.append(s.inbound_rtp)
            # ALWAYS say something. The first version logged only when it
            # found a candidate pair and a video stream, so an empty
            # stats list produced no line and no error — indistinguishable
            # from the task not running, which cost two deploys to work
            # out. Same failure this codebase already learned from
            # `screen_pass ok` meaning only "the call did not throw".
            if pair is None:
                kinds: dict = {}
                for s in _src:
                    k = s.WhichOneof("stats") or "?"
                    kinds[k] = kinds.get(k, 0) + 1
                log.warning(
                    "identity.link no candidate pair — pub=%d sub=%d "
                    "kinds=%s video=%d",
                    len(_pub), len(_sub), kinds, len(video),
                )
            else:
                log.warning(
                    "identity.link avail_in=%.0fkbps rtt=%.0fms bytes_in=%s "
                    "video_streams=%d",
                    pair.available_incoming_bitrate / 1000.0,
                    pair.current_round_trip_time * 1000.0,
                    pair.bytes_received, len(video),
                )
            for r in video[:4]:
                log.warning(
                    "identity.link_video %sx%s fps=%.1f decoded=%s "
                    "dropped=%s lost=%s jitter=%.0fms",
                    r.inbound.frame_width, r.inbound.frame_height,
                    r.inbound.frames_per_second, r.inbound.frames_decoded,
                    r.inbound.frames_dropped, r.received.packets_lost,
                    r.received.jitter * 1000.0,
                )
        except Exception:  # noqa: BLE001 — a probe must never kill the service
            log.exception("identity.link_stats_failed")
            return


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
            row = {}
            for cam_id, sp in _HOME.cameras.items():
                if _HOME.seat_on(cam_id) is not None:
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
def _mint_token(key: str, secret: str, room: str, shard: str = "") -> str:
    import datetime as _dt

    # UNIQUE PER SHARD (2026-08-24). LiveKit allows one participant per
    # identity per room: a second join with the same identity DISCONNECTS
    # the first. Under `patient_<id>` rooms there was exactly one identity
    # service per room, so a constant "identity-service" was safe. Account
    # rooms put every one of a caregiver's cameras in ONE room while the
    # fleet still shards identity PER CAMERA, so two processes joined the
    # same room under the same name and evicted each other on a ~60s
    # cycle — each ran ~66s, was kicked when the other restarted, exited
    # rc=1, and the backend read `reid_unavailable_llm` in the gaps.
    #
    # `--only-cam` is the shard key and is globally unique (it is the
    # camera's LiveKit identity), so it stays distinct across NODES too,
    # which a port could not. Empty means one service owns the whole
    # room, where the bare name is still correct.
    #
    # The suffix is safe for consumers: the gateway and the bridge
    # allow-list AI subscribers by PREFIX (`identity-service`), not by
    # equality, so a suffixed name still gets track permission.
    ident = f"identity-service-{shard}" if shard else "identity-service"
    return (
        lkapi.AccessToken(key, secret)
        # LONG TTL (2026-07-25): the SDK re-presents the ORIGINAL token
        # on auto-reconnect. A default-TTL token expired overnight, a
        # network blip hit at 12:23, reconnect got 401 → "not
        # retrying" → the service ran blind for 6h on a frozen frame.
        .with_ttl(_dt.timedelta(days=14))
        .with_identity(ident)
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
MOTION_DIFF_THR = 3.0   # mean abs diff threshold; static-camera noise
                        # floor is ~1-2, a person entering is well above
                        # — tune from motion_latched logs
# Latch metric (2026-08-18): MAX over 4x4-px blocks of the 64x36 gray
# thumbnail, same threshold — not the whole-frame mean. The global mean
# is blind to small-area REAL motion and generous to light changes:
# a hand waved vigorously 20ft away is ~4 thumbnail px x ~20 levels
# ~= 0.05 global (50x under threshold, never latched), while a phone
# screen lighting up re-exposes the whole IR frame and sailed over it
# (Aryaman's bedroom, tonight). Per-block, the wave concentrates to
# ~5.0 in its block and latches; IR grain is spatially uncorrelated,
# so a 16-px block still averages it to ~1 sigma-ish — and a rare
# grain latch costs one 4-frame existence poll, never a VLM call.
# NEMARA_MOTION_LATCH=global restores the old whole-frame mean.
MOTION_LATCH_MODE = os.environ.get("NEMARA_MOTION_LATCH", "block").strip().lower()


MOTION_BLOCK_THR = float(os.environ.get("NEMARA_MOTION_BLOCK_THR", "3.5"))
                        # block-MAX rides above the global-mean noise
                        # floor (max over 144 blocks picks up the noise
                        # tail), so it gets its own slightly higher bar.
                        # A 20ft hand-wave measures ~10 in native units
                        # — margin is on the signal side.


def _latch_diff(thumb: "np.ndarray", prev: "np.ndarray") -> tuple[float, float]:
    """(motion figure, threshold) between two 36x64 int16 thumbnails.

    Movement-vs-light disambiguation (Aryaman 2026-08-18): a light
    shift (phone glow, IR auto-gain re-exposure) is mostly
    MULTIPLICATIVE and smooth — it scales brightness without moving
    edges — so the previous thumbnail is GAIN-MATCHED to the current
    one's mean before diffing (matched, never normalized to a fixed
    target: fixed-target scaling amplified dark-scene grain ~2x in
    calibration). Motion RELOCATES structure, which survives the gain
    match and concentrates in one 4x4-px block; the whole-frame mean
    diluted a 20ft hand-wave ~50x under threshold while a phone glow
    sailed over it — exactly backwards for a bedroom.
    """
    if MOTION_LATCH_MODE == "global":
        return float(np.abs(thumb - prev).mean()), MOTION_DIFF_THR
    gain = float(thumb.mean()) / max(float(prev.mean()), 1.0)
    d = np.abs(thumb.astype(np.float32) - prev.astype(np.float32) * gain)
    # 4x4-px blocks: (36,64) -> (9,4,16,4) -> per-block means, take max.
    return float(d.reshape(9, 4, 16, 4).mean(axis=(1, 3)).max()), MOTION_BLOCK_THR
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
            # What size is ACTUALLY arriving. The overlay for one camera
            # rendered 640 wide while another rendered 1920, and nothing
            # in the pipeline resizes (TRACK_SEND_W is uncapped), so the
            # difference had to be at intake — but the node's inbound
            # bandwidth said both were on the top rung. Two measurements
            # that cannot both be right, and no way to tell which was
            # lying without this line. Once per size change, not per
            # frame.
            if bgr.shape[:2] != getattr(cam_loop, "_last_shape", None):
                cam_loop._last_shape = bgr.shape[:2]
                log.warning(
                    "identity.intake_frame_size cam=%s %dx%d",
                    cam_loop.camera_id, bgr.shape[1], bgr.shape[0],
                )
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
                    diff, thr = _latch_diff(thumb, prev_thumb)
                    # Live calibration surface (2026-08-18): the latch
                    # was tuned on synthetic thumbnails; these fields
                    # ride /state so a real hand-wave's magnitude can be
                    # read off the dashboard instead of guessed.
                    _lt = getattr(cam_loop, "latch_telemetry", None)
                    if _lt is None or now - _lt["peak_at"] > 15.0:
                        cam_loop.latch_telemetry = {
                            "last": diff, "peak15": diff, "peak_at": now,
                            "thr": thr,
                        }
                    else:
                        _lt["last"] = diff
                        if diff >= _lt["peak15"]:
                            _lt["peak15"] = diff
                            _lt["peak_at"] = now
                    if diff >= thr and not cam_loop.motion_pending:
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


# ── Subscribed layer: pinned, and stepped down on an empty room ──────
#
# TWO jobs, and the first one is not optional.
#
# PIN IT. With no explicit request the SFU picks the rung, and it
# re-picks whenever its congestion estimate twitches — measured
# backend-side as sharp<->blurry flapping that blinds YOLO and face
# matching (soak #2: blur<->identity r=0.52). That flapping is why the
# gateway ships `.simulcast-off` at all. Every subscriber that pins its
# layer takes the choice away from the SFU, which is what makes putting
# the ladder back a safe thing to do.
#
# THEN STEP IT DOWN WHEN THE ROOM IS EMPTY. LiveKit bills the
# subscription and does not care that we are doing nothing with the
# frames. An empty room delivered 5 Mbps to this process all night for
# no reason — and this is the SECOND copy, because the backend holds
# its own subscription to the same camera.
#
# The step-down must never cost a detection. Two reasons it does not:
# the motion latch that ends the idle period runs on a 64x36 gray
# thumbnail and cannot tell the layers apart, and unlike the backend
# this process has no buffered window to judge in retrospect — it reads
# frames continuously, so the tick after the step-up is already native.
# gpu/identity/loop.py's 2026-07-26 finding — that shipping 640px costs
# "the small-distant-person detection risk" — is about the layer we
# work AT, not the layer we idle at.
_LAYER_ACTIVE = "high"
# LOW, not medium. LiveKit bills each subscriber for what IT pulls, and
# an empty room is most of the day: measured 2026-08-24, the TV Room had
# a person in 40% of ticks and the bedroom 74%. Stepping to the bottom
# rung is roughly 150-300 kbps against ~4 Mbps, and the room stays
# visible — so nothing has to wake us, unlike unsubscribing.
#
# LOW here even though the BACKEND deliberately uses MEDIUM, and the
# difference is worth stating because it looks like an inconsistency.
#
# The backend steps to 640x360 because the tick that wakes it analyses
# the window ALREADY IN ITS BUFFER — so whatever rung was arriving
# during the quiet period is what a fall gets judged on, and 320x180 is
# not something to show a VLM a fall in. Identity has no such
# obligation: it keeps no window for later judgement, and the moment it
# wakes it asks for `_LAYER_ACTIVE` and the next batch is full
# resolution.
#
# And its wake authority does not depend on the rung at all.
# `motion_pending` is set by a THUMBNAIL DIFF on the incoming frames
# (see the latch above), not by person detection — the same reason the
# backend notes that block motion "cannot tell the difference either
# way". A person walking in still moves enough pixels at 320x180 to
# latch it.
_LAYER_IDLE = os.environ.get("NEMARA_IDENTITY_IDLE_QUALITY", "low").strip().lower()
# Which cameras this process has stopped pulling video for because the
# room is empty. Read by the subscription walk so its repair does not
# undo the economy.
_idle_video: dict[str, bool] = {}
# Cameras we have EVER received a motion_state packet from.
#
# The sever is only safe for a camera that can wake us, and not every
# camera can. The gateway publishes motion_state; the Warner bridge
# publishes NONE — verified 2026-08-25 by grepping the whole bridge for
# the topic. Severing a camera that cannot wake us means it goes dark
# and stays dark, which is the worst failure this service has: silent,
# and indistinguishable from a quiet room.
#
# So a camera earns the sever by proving it can wake us. Until then it
# gets the old step-down, which is cheaper but never blind.
_motion_capable: set[str] = set()
# Fall back to the old step-down-a-rung behaviour. The escape hatch if
# severing on empty turns out to miss a wake — set to 1 and the room
# merely gets cheaper instead of silent.
_PAUSED_SEVER_ONLY = os.environ.get(
    "NEMARA_IDENTITY_IDLE_STEP_ONLY", "").strip() in ("1", "true", "yes")
_QUALITY = {
    "low": rtc.VideoQuality.VIDEO_QUALITY_LOW,
    "medium": rtc.VideoQuality.VIDEO_QUALITY_MEDIUM,
    "high": rtc.VideoQuality.VIDEO_QUALITY_HIGH,
}
# Publication + last-requested layer per camera. Transitions only: a
# layer request is a signalling round-trip and this is consulted on
# every intake decision.
_pubs: dict[str, Any] = {}
_layer: dict[str, str] = {}


_PIN_LAYER = os.environ.get("NEMARA_IDENTITY_PIN_LAYER", "1").strip() not in (
    "0", "false", "no")


def request_layer(cam_id: str, quality: str) -> None:
    """Ask the SFU for one rung. Best-effort — on failure we keep
    whatever we were getting, which is the pre-2026-08-24 behaviour.

    KILL-SWITCH, because this call is itself a suspect. Identity never
    made it before 2026-08-24, and the camera that oscillates
    2304 -> 320 -> 2304 does so on almost exactly the period at which we
    send it — with the publisher healthy (all three layers active,
    quality_limitation_reason=0), 312 Mbps of headroom on the node, and
    the same behaviour when that camera is the only subscription on the
    box. Every other explanation has been measured and eliminated, which
    leaves the request. `NEMARA_IDENTITY_PIN_LAYER=0` removes it so that
    can be tested rather than argued about."""
    if not _PIN_LAYER:
        return
    if _layer.get(cam_id) == quality:
        return
    pub = _pubs.get(cam_id)
    if pub is None:
        return
    _layer[cam_id] = quality
    try:
        pub.set_video_quality(_QUALITY.get(quality, _QUALITY["high"]))
        log.info("identity.layer_request cam=%s quality=%s", cam_id, quality.upper())
    except Exception:  # noqa: BLE001 — an economy must not break intake
        log.warning("identity.layer_request_failed cam=%s", cam_id, exc_info=True)


def set_idle_video(cam_id: str, idle: bool) -> None:
    """Stop pulling video for an empty room; resume on the first motion.

    NOT a step down to a cheaper rung. LiveKit bills each subscriber for
    what it pulls, and the bottom rung is still a stream — an empty room
    is most of the day (measured 2026-08-24: the TV Room had a person in
    40% of ticks, the bedroom 74%), so the correct amount to download
    for it is none.

    Unsubscribing normally means going blind, because identity's own
    wake signal is a thumbnail diff over the frames it is no longer
    receiving. It does not here: the camera publishes a `motion_state`
    DATA PACKET, and LiveKit delivers data to every participant in the
    room whether or not they subscribe to any track. So identity keeps
    the cheap channel and drops the expensive one, and needs nothing
    from the backend to come back.

    `paused_cameras()` already severs on ai_paused and is a different
    question — that one is the caregiver switching the AI off, this one
    is the room being empty. Both end in `set_subscribed(False)`; the
    subscription walk re-asserts whichever is currently wanted.
    """
    if _PAUSED_SEVER_ONLY or cam_id not in _motion_capable:
        # No proof this camera can wake us — step the rung down instead
        # of going blind. Logged once per transition so the reason a
        # camera is merely cheap rather than free is visible.
        if idle and cam_id not in _motion_capable:
            log.info(
                "identity.idle_step_only cam=%s — no motion_state packet "
                "seen from this camera, stepping the layer down instead of "
                "unsubscribing (a camera that cannot wake us must not sleep)",
                cam_id,
            )
        request_layer(cam_id, _LAYER_IDLE if idle else _LAYER_ACTIVE)
        return
    _idle_video[cam_id] = idle
    pub = _pubs.get(cam_id)
    if pub is None:
        return
    try:
        pub.set_subscribed(not idle)
        if not idle:
            # A re-subscribe resets the SFU's idea of what we asked for,
            # so the pin has to go out again with it.
            _layer.pop(cam_id, None)
            request_layer(cam_id, _LAYER_ACTIVE)
        log.info("identity.idle_video cam=%s subscribed=%s", cam_id, not idle)
    except Exception:  # noqa: BLE001 — an economy must never break intake
        log.warning("identity.idle_video_failed cam=%s", cam_id, exc_info=True)


_paused_cache: dict = {"t": 0.0, "v": set()}
_PAUSED_TTL_S = 8.0


async def paused_cameras() -> set[str]:
    """Cam-id prefixes whose analysis the user has switched off.

    THE SAME RULE AS THE BACKEND. Pausing AI stops the detector worker
    pulling video; this process holds a SECOND subscription to the same
    camera, so leaving it running means pausing halves the bill instead
    of ending it. Nothing here will be looked at while the flag is set,
    and the flag is the resume signal, so there is nothing to wait for.

    Polled rather than passed as a flag: `--only-cam` is fixed at launch
    and pause state is not, and the alternative — dropping paused
    cameras from the node roster so the supervisor stops the process —
    costs a full model reload on resume. An unsubscribed process is
    already ~zero GPU (the empty-room economy), so keeping it alive and
    idle buys a fast resume for nothing.

    Fails OPEN: an unreachable database must never blind a camera whose
    user has NOT paused it.
    """
    now = time.time()
    if now - _paused_cache["t"] < _PAUSED_TTL_S:
        return _paused_cache["v"]
    try:
        sm = get_sessionmaker()
        async with sm() as session:
            from sqlalchemy import text as _sql_text
            rows = (await session.execute(_sql_text(
                "SELECT substr(id::text, 1, 8) FROM public.cameras "
                " WHERE ai_paused OR disabled"
            ))).all()
        _paused_cache.update(t=now, v={r[0] for r in rows})
    except Exception:  # noqa: BLE001 — never blind an unpaused camera
        log.warning("identity.paused_lookup_failed", exc_info=True)
        _paused_cache["t"] = now
    return _paused_cache["v"]


def cam_selected(only_cam: str, ident: str) -> bool:
    """Does `--only-cam` select this LiveKit participant?

    Matched against the FULL identity (`cam_650ca94d`), never the
    stripped camera id, so both shapes work: the roster passes
    `camera_identity` verbatim and a human passes the bare `650ca94d`.
    Comparing against the stripped id accepted only the human shape, so
    every roster-started process skipped the one camera it existed for
    and served an empty camera map while looking healthy (2026-08-23).
    """
    return not only_cam or only_cam in ident


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

    @room.on("data_received")
    def _on_data(data) -> None:
        """Wake on the camera's own motion packet.

        This is what makes unsubscribing safe. LiveKit delivers data to
        every participant in the room regardless of track subscriptions,
        so a camera we are pulling ZERO video from can still tell us
        somebody moved. Same topic and payload the backend already
        consumes (`room_worker._handle_data_packet`).

        Never raises into the SDK's event loop: a malformed packet must
        cost one missed wake, not the reader.
        """
        try:
            if getattr(data, "topic", "") != "motion_state":
                return
            participant = getattr(data, "participant", None)
            ident = getattr(participant, "identity", "") or ""
            if not ident.startswith("cam_"):
                return
            cam_id = ident.removeprefix("cam_")
            # Any motion packet at all proves this camera can wake us,
            # which is what makes the sever safe for it.
            if cam_id not in _motion_capable:
                _motion_capable.add(cam_id)
                log.info("identity.motion_capable cam=%s", cam_id)
            payload = json.loads(bytes(data.data).decode("utf-8"))
            if not bool(payload.get("in_motion")):
                return
            if not _idle_video.get(cam_id):
                return                      # already awake
            log.info("identity.motion_wake cam=%s", cam_id)
            cam_loop = home.cameras.get(cam_id)
            if cam_loop is not None:
                cam_loop.motion_pending = True
            set_idle_video(cam_id, False)
        except Exception:  # noqa: BLE001 — a bad packet is not a fault
            log.warning("identity.motion_packet_failed", exc_info=True)

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
        if not cam_selected(only_cam, ident):
            # UNSUBSCRIBE, do not merely ignore. `auto_subscribe=True`
            # pulls every publisher in the room, and returning here only
            # skipped the PROCESSING — the SFU kept delivering this
            # camera's full stream to a process that dropped every frame
            # on the floor.
            #
            # It is quadratic. One process per SENSOR, all in one account
            # room, each subscribed to all N cameras = N^2 streams for N
            # cameras. Measured 2026-08-24 with N=2: the node pulled
            # 2,259 KB/s (~18 Mbps) where the two top layers total
            # 9 Mbps. At N=8, the node's port capacity, that is 64
            # streams to do the work of 8.
            try:
                pub.set_subscribed(False)
                log.info(
                    "identity.camera_unsubscribed cam=%s (--only-cam)", cam_id)
            except Exception:  # noqa: BLE001 — never break the callback
                log.warning(
                    "identity.unsubscribe_failed cam=%s — still paying for "
                    "this stream", cam_id, exc_info=True)
            return
        cam_loop = home.cameras.get(cam_id)
        if cam_loop is None:
            cam_loop = CameraLoop(camera_id=cam_id, home=home)
            # Swap-not-mutate: live_keys()/HTTP handlers iterate this
            # dict from other threads; replace the ref atomically.
            cams = dict(home.cameras)
            cams[cam_id] = cam_loop
            home.cameras = cams
            cam_loop.on_idle = (
                lambda idle, c=cam_id: set_idle_video(c, idle)
            )
            loop_tasks[cam_id] = asyncio.ensure_future(cam_loop.run())
            log.info("identity.camera_started cam=%s", cam_id)
        # PIN the layer on every (re)subscribe, before any frame arrives.
        # Unpinned, the SFU chooses and re-chooses, and that flapping is
        # what blinds face matching. A re-subscribe also resets the SFU's
        # idea of what we asked for, so the remembered value must go with
        # it or the next request looks redundant and gets skipped.
        _pubs[cam_id] = pub
        _layer.pop(cam_id, None)
        request_layer(cam_id, _LAYER_ACTIVE)
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
    # Held in a module global on purpose. asyncio keeps only a WEAK
    # reference to a running task, so a bare create_task() whose result
    # nobody stores can be garbage-collected before it ever runs — which
    # is exactly what happened here: the code shipped, the process
    # restarted, and not one stats line was ever emitted.
    global _LINK_STATS_TASK
    _LINK_STATS_TASK = asyncio.create_task(_log_link_stats(room))
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
    # Last subscription state we asked for, per camera. Transitions only.
    _subscribed: dict[str, bool] = {}
    while True:
        await asyncio.sleep(walk_every_s)
        try:
            now = time.time()
            paused = await paused_cameras()
            for p in list(room.remote_participants.values()):
                if not p.identity.startswith("cam_"):
                    continue
                cam_id = p.identity.removeprefix("cam_")
                if not cam_selected(only_cam, p.identity):
                    continue
                # Pause follows the user, not the launch flags, so it is
                # polled here rather than passed in. The walk already
                # runs every 10s over exactly these publications, so the
                # check costs one cached query and resume is ~10s.
                # Two independent reasons not to pull: the caregiver
                # paused the AI, and the room is empty. The walk repairs
                # toward whichever is currently true, or its own repair
                # would re-subscribe an empty room every 10 seconds and
                # quietly undo the economy.
                want = (cam_id[:8] not in paused) and not _idle_video.get(cam_id)
                resubscribed = False
                if _subscribed.get(cam_id) is not want:
                    _subscribed[cam_id] = want
                    resubscribed = True
                    for pub in p.track_publications.values():
                        try:
                            pub.set_subscribed(want)
                        except Exception:  # noqa: BLE001 — never break the walk
                            log.warning(
                                "identity.pause_subscribe_failed cam=%s want=%s",
                                cam_id, want, exc_info=True)
                    log.info(
                        "identity.subscription %s cam=%s",
                        "restored" if want else "SEVERED (ai_paused)", cam_id)
                if want:
                    # ENFORCE the layer every pass, not once at
                    # subscribe. `set_subscribed(True)` resets the track
                    # settings and the SDK default is a width-0 request
                    # the SFU reads as the SMALLEST rung, so the walk's
                    # own repair shrank the picture every time it ran —
                    # and `request_layer` would not undo it, because it
                    # acts only on a change and still believed HIGH was
                    # in force.
                    #
                    # Asserting it right after the subscribe was not
                    # enough: the settings reset lands at the SFU AFTER
                    # our request and overwrites it. The collapse then
                    # lasted exactly one walk period, 10s, every time —
                    # which is what says it is a race and not congestion.
                    #
                    # So this is a reconcile — but ONLY on the pass that
                    # actually re-subscribed. Forgetting the remembered
                    # value every pass sent one UpdateTrackSettings per
                    # camera per 10 seconds forever, on a camera nothing
                    # had touched, and `request_layer`'s own docstring
                    # names that call as the prime suspect for the
                    # 2304 -> 320 -> 2304 oscillation. It was right:
                    # measured 2026-08-25 on the TV Room, the SFU
                    # downshifted between our requests and the intake
                    # walked 2304x1296 -> 640x360 -> 320x180 -> 2304x1296
                    # on the request period.
                    #
                    # There is no race to beat on a pass where we did not
                    # call `set_subscribed`, because nothing reset the
                    # track settings. Re-asserting there is not
                    # enforcement, it is churn — the same lesson the
                    # bridge's permission re-send taught on the publisher
                    # side the day before.
                    if resubscribed:
                        _layer.pop(cam_id, None)
                    request_layer(cam_id, _LAYER_ACTIVE)
                else:
                    # Nothing to police on a camera we are not pulling —
                    # and the intake-quiet strike counter below would
                    # otherwise read our own silence as a fault and
                    # exit(5) into a supervisor restart loop.
                    walk_strikes.pop(cam_id, None)
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


# ── Face galleries + coverage (persistent DB enrollment) ─────────────
async def _load_room_identity(
    room: str,
) -> tuple[dict[UUID, np.ndarray], dict[str, list[UUID]]]:
    """Who this room's cameras watch, and each of their face galleries.

    A camera publishes into exactly ONE room, so `cameras.livekit_room`
    is the whole membership question — a shared lounge covering two
    residents is one camera in one room, and BOTH residents' galleries
    have to be loaded here or the second one can never be recognized.

    Coverage is read per camera rather than per room because the two
    differ: a room can hold a private bedroom camera (one resident) and
    a shared lounge camera (two), and asking the bedroom camera about
    the second resident would only add a losing comparison.

    Returns ({patient_id: (N,512) gallery}, {camera_id8: [patient_id]}).
    A resident with no ready enrollment is kept in coverage but gets no
    gallery — `IdentityHome.covered` drops them from the question.
    """
    sm = get_sessionmaker()
    coverage: dict[str, list[UUID]] = {}
    galleries: dict[UUID, np.ndarray] = {}
    async with sm() as session:
        from sqlalchemy import text as _sql_text

        rows = (await session.execute(
            _sql_text(
                "SELECT substr(c.id::text, 1, 8) AS c8, cp.patient_id "
                "  FROM public.cameras c "
                "  JOIN public.camera_patients cp ON cp.camera_id = c.id "
                " WHERE c.livekit_room = :room"
            ),
            {"room": room},
        )).all()
        for c8, pid in rows:
            coverage.setdefault(c8, []).append(pid)
        for pid in {pid for _c8, pid in rows}:
            embs = [
                embedding_from_bytes(r.embedding)
                for r in await face_embedding_repo
                .list_ready_for_patient_unscoped(session, pid)
            ]
            if embs:
                galleries[pid] = np.stack(embs).astype(np.float32)
            else:
                log.warning("identity.face_gallery_empty patient=%s", pid)
    return galleries, coverage


async def _refresh_gallery(home: IdentityHome, room: str) -> None:
    """Re-read enrollment AND coverage on a timer.

    This is also how a camera newly shared with a second resident takes
    effect: the next refresh sees the extra `camera_patients` row and
    the node starts asking about that person. No restart, no roster
    round-trip.
    """
    while True:
        await asyncio.sleep(GALLERY_REFRESH_S)
        try:
            home.face_galleries, home.coverage = (
                await _load_room_identity(room)
            )
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
    sam = SamServiceClient(
        base_url=args.pod_url, timeout_s=25.0,
        expected_face_model="buffalo_l",
    )
    # DB may still be starting (post-reboot: Docker up before Supabase —
    # the 2026-07-24 outage). Start with no galleries rather than dying;
    # the 60s refresher heals it as soon as the DB answers.
    try:
        face_galleries, coverage = await _load_room_identity(args.room)
    except Exception:                         # noqa: BLE001
        log.exception(
            "identity.gallery_load_failed at startup — starting with NO "
            "face galleries; the %ds refresher will retry",
            int(GALLERY_REFRESH_S),
        )
        face_galleries, coverage = {}, {}
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
                # ALL workers, not just the base pod (2026-08-15 review
                # finding 2 on the v3.0 worker pool): a HUNG extra
                # worker never exits — the entrypoint's crash-restart
                # can't see it — so its cameras would fail silently
                # forever while worker 0's heartbeat kept this watchdog
                # happy. Any worker failing keeps last_pod_ok stale and
                # the wedge exit below gets a fresh start loudly.
                await home_ref.sam.health()
                for _extra in home_ref.pods:
                    await _extra.health()
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
        sam=sam,
        pods=pods,
        body_gallery=CrossClipBodyGallery(settings=settings),
        face_galleries=face_galleries,
        coverage=coverage,
        journal=EventJournal(Path("out/identity_soak")),
        cam_names=cam_names,
    )
    home.last_pod_ok = time.time()   # watchdog baseline: boot counts
    _HOME = home
    log.info(
        "identity.up room=%s pod=%s residents=%d galleries=%s — "
        "http://localhost:%d/",
        args.room, args.pod_url, len(home.face_galleries),
        {str(p)[:8]: len(g) for p, g in home.face_galleries.items()},
        args.port,
    )
    await asyncio.gather(
        _run_livekit(
            url, _mint_token(key, secret, args.room, shard=args.only_cam),
            home, only_cam=args.only_cam,
        ),
        _refresh_gallery(home, args.room),
        _timeline_sampler(),
        _pod_progress_watchdog(home),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room", required=True, help="LiveKit room, patient_<id>")
    # NO --patient (removed 2026-08-20). The room's cameras name the
    # residents to load, via `cameras.livekit_room` -> `camera_patients`,
    # because a camera can watch more than one person and a single
    # command-line patient could only ever describe one of them.
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
