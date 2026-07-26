"""Live EdgeTAM tracker view — RESURRECTED 2026-07-24, byte-faithful to
the 2026-07-23 ~9:30pm PDT run that tracked beautifully.

This is the exact viewer that ran that night (reconstructed from the
session record of its creation + both edits), untouched by everything
built since: no identity, no probes, no presets, no resize, unbounded
SDK queue, 300-frame ring, adjacent batches (no overlap seed), pod
session key "livingroom_view" — the same key as that night.

    uv run python tools/live_tracker_view.py --room patient_<id> \
        --cam-participant 3c6570af
    # open http://localhost:8011/
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
import time
from collections import deque

import cv2
import httpx
import livekit.api as lkapi
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from livekit import rtc

# ── shared state (single writer per field; GIL-atomic swaps) ──────────
_frames: deque = deque(maxlen=300)          # (global_idx, monotonic_ts, bgr)
_frame_idx = 0
_fps_in = 0.0
_latest_jpeg: bytes | None = None
_hud = "waiting for frames…"


def _color(tid: int) -> tuple[int, int, int]:
    """Stable vivid BGR color per track id."""
    r = (int(tid) * 2654435761) & 0xFFFFFFFF
    return (50 + r % 180, 50 + (r >> 8) % 180, 50 + (r >> 16) % 180)


def _containment(inner, outer) -> float:
    """area(inner ∩ outer) / area(inner) — the display_veto test."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    a = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (iw * ih) / a


# ── LiveKit ingest ────────────────────────────────────────────────────
def _mint_token(api_key: str, api_secret: str, room: str) -> str:
    return (
        lkapi.AccessToken(api_key, api_secret)
        .with_identity("edgetam-live-view")
        .with_name("EdgeTAM Live View")
        .with_grants(
            lkapi.VideoGrants(
                room_join=True, room=room,
                can_subscribe=True, can_publish=False,
            )
        )
        .to_jwt()
    )


async def _read_video(track: rtc.Track) -> None:
    """Pull every frame off the track into the rolling ring, as BGR."""
    global _frame_idx, _fps_in
    stream = rtc.VideoStream(track)
    n, t0 = 0, time.monotonic()
    async for ev in stream:
        frame = ev.frame
        if frame.type != rtc.VideoBufferType.RGBA:
            frame = frame.convert(rtc.VideoBufferType.RGBA)
        arr = np.frombuffer(frame.data, dtype=np.uint8).reshape(
            frame.height, frame.width, 4,
        )
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        _frame_idx += 1
        _frames.append((_frame_idx, time.monotonic(), bgr))
        n += 1
        if n >= 30:
            now = time.monotonic()
            _fps_in = n / max(1e-3, now - t0)
            n, t0 = 0, now


async def _run_livekit(url: str, token: str, cam_filter: str = "") -> None:
    room = rtc.Room()

    @room.on("track_subscribed")
    def _on_sub(track, pub, participant):  # noqa: ANN001
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        # Both cameras share one patient_<id> room; pick the one we want.
        if cam_filter and cam_filter not in participant.identity:
            print(f"[livekit] skip video from {participant.identity!r} "
                  f"(want substring {cam_filter!r})")
            return
        print(f"[livekit] READING video from {participant.identity!r}")
        asyncio.ensure_future(_read_video(track))

    await room.connect(url, token, rtc.RoomOptions(auto_subscribe=True))
    print(f"[livekit] connected; remote participants: "
          f"{[p.identity for p in room.remote_participants.values()]}")
    while True:
        await asyncio.sleep(3600)


# ── pod driver (contiguous windows → EdgeTAM continuous session) ──────
async def _pod_observe(client, pod_url, cam_id, batch, detect_screens, thr):
    files = [("frames", (f"{gi:06d}.jpg", jpg, "image/jpeg")) for gi, jpg in batch]
    data = {
        "camera_id": cam_id, "person_thr": str(thr),
        "face_interval": "999", "face_cap": "0",         # no face work — pure tracking
        "detect_screens": "1" if detect_screens else "0",
    }
    r = await client.post(f"{pod_url}/observe", data=data, files=files, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _boxes_for_frame(payload: dict, gi: int) -> dict[int, tuple]:
    """Per-track box for the newest frame in the batch (or its last row)."""
    out: dict[int, tuple] = {}
    for t in payload.get("tracks", []):
        tid = int(t["track_id"])
        chosen = None
        for row in t.get("rows", []):
            chosen = row                       # last row = newest in batch order
            if int(row[0]) == gi:
                break
        if chosen is not None:
            out[tid] = (chosen[1], chosen[2], chosen[3], chosen[4], chosen[5])
    return out


def _overlay(bgr, boxes, screens, vetoed, hud) -> bytes:
    img = bgr.copy()
    h = img.shape[0]
    for x1, y1, x2, y2, p in screens:
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 210), 2)
        cv2.putText(img, f"SCREEN {p:.2f}", (int(x1), int(y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 210), 1)
    for tid, (x1, y1, x2, y2, prob) in boxes.items():
        v = tid in vetoed
        col = (110, 110, 110) if v else _color(tid)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 1 if v else 3)
        tag = f"T{tid} {prob:.2f}" + ("  VETO:screen" if v else "")
        cv2.putText(img, tag, (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.rectangle(img, (0, h - 26), (len(hud) * 9 + 16, h), (0, 0, 0), -1)
    cv2.putText(img, hud, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1)
    ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return out.tobytes() if ok else b""


async def _driver(pod_url, cam_id, thr, window_s, target_fps) -> None:
    global _latest_jpeg, _hud
    last_idx, screens_at, last_log = -1, 0.0, 0.0
    screens: list = []
    async with httpx.AsyncClient() as client:
        while True:
            await asyncio.sleep(window_s)
            new = [f for f in list(_frames) if f[0] > last_idx]
            if not new:
                continue
            # CONTIGUOUS decimation — the fix for the renumbering. Keep
            # frames evenly across the WHOLE since-last interval at
            # ~target_fps; NEVER keep only the tail. `new[-N:]` (the old
            # bug) drops the frames that pile up while a slow pod catches
            # up → a blind gap between windows → renumbering. On EdgeTAM
            # (~36ms/frame) the pod keeps up so batches stay small and
            # real-time; on SAM3 it can't, so this lags but stays
            # contiguous (bounded per batch so it can't death-spiral).
            kept, next_t = [], 0.0
            for f in new:
                if f[1] >= next_t:
                    kept.append(f)
                    next_t = f[1] + 1.0 / target_fps
            if len(kept) > 40:                           # coarsen, still even
                step = len(kept) / 40.0
                kept = [kept[int(i * step)] for i in range(40)]
            new = kept
            enc, newest = [], None
            for gi, _ts, bgr in new:
                ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    enc.append((gi, jpg.tobytes()))
                    newest = (gi, bgr)
            if not enc:
                continue
            last_idx = enc[-1][0]
            now = time.monotonic()
            do_screens = (now - screens_at) > 120.0      # screens are furniture
            t0 = time.time()
            try:
                payload = await _pod_observe(client, pod_url, cam_id, enc,
                                             do_screens, thr)
            except Exception as e:                        # noqa: BLE001
                _hud = f"pod error: {str(e)[:60]}"
                continue
            pod_ms = (time.time() - t0) * 1000.0
            if do_screens:
                screens_at = now
                screens = [tuple(float(v) for v in s)
                           for s in payload.get("screen_regions", [])]
            gi_new, bgr_new = newest
            boxes = _boxes_for_frame(payload, gi_new)
            vetoed = {
                tid for tid, b in boxes.items()
                if any(_containment(b[:4], s[:4]) >= 0.8 for s in screens)
            }
            _hud = (f"EdgeTAM live | tracks={len(boxes)} veto={len(vetoed)} "
                    f"screens={len(screens)} | pod={pod_ms:.0f}ms "
                    f"batch={len(enc)}f | in={_fps_in:.0f}fps")
            _latest_jpeg = _overlay(bgr_new, boxes, screens, vetoed, _hud)
            if now - last_log > 3.0:
                print(f"[track] {_hud} | ids={sorted(boxes)}", flush=True)
                last_log = now


# ── MJPEG server ──────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(
        '<html><head><title>EdgeTAM live</title></head>'
        '<body style="margin:0;background:#0b0b0d;display:flex;'
        'align-items:center;justify-content:center;height:100vh">'
        '<img src="/stream" style="max-width:100vw;max-height:100vh">'
        '</body></html>'
    )


def _mjpeg():
    while True:
        jpg = _latest_jpeg
        if jpg:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        time.sleep(1 / 15)


@app.get("/stream")
def stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _amain(args) -> None:
    url = os.environ["NEMARA_LIVEKIT_URL"]
    key = os.environ["NEMARA_LIVEKIT_API_KEY"]
    secret = os.environ["NEMARA_LIVEKIT_API_SECRET"]
    token = _mint_token(key, secret, args.room)
    print(f"[view] serving http://localhost:{args.port}/  room={args.room} "
          f"pod={args.pod_url} cam_id={args.cam_id}")
    await asyncio.gather(
        _run_livekit(url, token, args.cam_participant),
        _driver(args.pod_url, args.cam_id, args.person_thr,
                args.window_s, args.target_fps),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", required=True, help="LiveKit room, e.g. patient_<id>")
    ap.add_argument("--cam-participant", default="",
                    help="substring of the camera participant identity to view "
                         "(e.g. the living-room camera id); the log prints every "
                         "identity it sees, so run once to read them off")
    ap.add_argument("--pod-url", default="http://localhost:8000")
    ap.add_argument("--cam-id", default="livingroom_view",
                    help="pod session key — keep distinct from the soak's camera")
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--person-thr", type=float, default=0.4)
    ap.add_argument("--window-s", type=float, default=0.6,
                    help="how often to ship a contiguous batch to the pod")
    ap.add_argument("--target-fps", type=float, default=12.0,
                    help="decimate each contiguous window to ~this fps, evenly "
                         "across the interval — never front-drop (the fix)")
    args = ap.parse_args()

    threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=args.port,
                                   log_level="warning"),
        daemon=True,
    ).start()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
