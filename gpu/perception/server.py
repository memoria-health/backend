"""GPU-pod SAM perception microservice v2 (FastAPI) — 2026-07-20.

v2 contract shift: the pod is PERCEPTION ONLY. It detects + tracks
people (SAM 3.1 concept propagation), cuts mask-clean crops, and
face-detects + embeds them (SCRFD + ArcFace in an isolated GPU worker
process) — and returns RAW observations. It makes ZERO identity
decisions: no labels, no gallery, no thresholds. All matching,
locking, carrying, pooling, and harvesting run on the backend brain
(`RemoteReplayPipeline` replays these observations through the same
`BodyReIDPipeline` code the legacy arm uses).

Consequences vs v1: the enrolled gallery never leaves the backend
(the /gallery endpoint is gone), and the response is per-track raw
material keyed by ORIGINAL frame index (frames arrive named by that
index, so no remapping layer exists anywhere).

Contract:
  GET  /health    models + gpu + face-provider status
  POST /observe   multipart `frames` JPEGs named "<origidx>.jpg"
                  + form: person_thr, face_interval, face_cap
                  -> raw observations JSON (schema in _payload docs)

Run on the pod (self-contained — src/ NOT needed):
  NEMARA_SAM_CKPT=/workspace/nemara/models/sam3.1_multiplex.pt \
  PYTHONPATH=/workspace/sam3:/workspace/nemara \
  uvicorn perception.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import logging
import os
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse

from perception.face_worker import FaceWorker
from perception import continuous, rawwire, sam_track

log = logging.getLogger("perception")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)

app = FastAPI(title="nemara-sam-perception-v2")

# ── Bearer auth ────────────────────────────────────────────────────────
# /observe takes patient frames and /faces returns face embeddings, and
# until 2026-08-11 neither had ANY authentication. That was survivable only
# because the server binds 127.0.0.1 and the sole route in was an ssh
# tunnel — i.e. the security property lived in a deployment detail, not in
# the service. Split the services, run anything else on the box, or set
# PERCEPTION_HOST=0.0.0.0 for one debug session, and it was wide open with
# nothing complaining.
#
# Deliberately NOT imported from identity.contract: perception is the lower
# layer and must not depend on the brain. Eight lines of hmac is a better
# trade than that coupling. Same semantics as contract.auth_ok, including
# "empty token disables auth" — see the boot warning below.
_AUTH_TOKEN = os.environ.get("NEMARA_PERCEPTION_TOKEN", "").strip()


def _auth_ok(header_value: str | None) -> bool:
    """True iff the caller may use the perception API.

    Constant-time compare (hmac.compare_digest) — no timing oracle on the
    token.
    """
    if not _AUTH_TOKEN:
        return True
    if not header_value or not header_value.startswith("Bearer "):
        return False
    return hmac.compare_digest(
        header_value.removeprefix("Bearer "), _AUTH_TOKEN,
    )


def _require_auth(request: Request) -> None:
    """Raise 401 unless the request carries the shared bearer.

    /health is deliberately left open: it carries no patient data and is
    what a supervisor and the container healthcheck poll.
    """
    if not _auth_ok(request.headers.get("authorization")):
        raise HTTPException(status_code=401, detail="unauthorized")


if not _AUTH_TOKEN:
    log.warning(
        "perception.auth_disabled — NEMARA_PERCEPTION_TOKEN is unset, so "
        "/observe and /faces accept EVERY caller. Safe only while bound to "
        "loopback (host=%s). Set the token before exposing this port.",
        os.environ.get("PERCEPTION_HOST", "127.0.0.1"),
    )

_CKPT = os.environ.get(
    "NEMARA_SAM_CKPT", "/workspace/nemara/models/sam3.1_multiplex.pt",
)
_FACE_MODEL = os.environ.get("NEMARA_FACE_MODEL", "buffalo_l")
_face = FaceWorker(
    model_name=_FACE_MODEL,
    providers=os.environ.get(
        "NEMARA_FACE_PROVIDERS",
        "CUDAExecutionProvider,CPUExecutionProvider",
    ),
)
_ready = False
# Deep-health state: repr of the last SAM load failure, None when the
# tracker is known-good. Set at startup warm and on every observe's
# get_sam; /health returns 503 while set. Shallow health (process up,
# models unchecked) hid a 5-hour total outage on 2026-08-13.
_sam_error: "str | None" = None

# Single-flight admission control (reviews E1/S1/C7): SAM perception is
# ~10-20s of GPU work per request and the predictor is a process
# singleton — a second in-flight request would only queue, breach the
# caller's timeout, and burn GPU on an abandoned response. Busy →
# instant 429 so the backend fails open in milliseconds. The
# threading.Lock makes the SAM/face single-flight invariant explicit
# for the to_thread hop (the semaphore alone already guarantees it).
_busy = asyncio.Semaphore(
    max(1, int(os.environ.get("NEMARA_PERCEPTION_CONCURRENCY", "1")))
)
# How long a 2nd camera waits for the single GPU slot before 429. MUST
# exceed one observe's service time or a concurrent camera always 429s
# (2026-07-23: a ~5s living-room observe > the old 4s window starved the
# bedroom — 183 429s). Env-tunable; the durable fix is keeping observe
# service < this via the frame budget (fewer SAM frames).
_ACQUIRE_TIMEOUT_S = float(os.environ.get("NEMARA_PERCEPTION_ACQUIRE_S", "10.0"))
class _Gate:
    """N-way admission to the perception pass, with `.locked()`.

    Was a plain `threading.Lock()`, justified (2026-07-21) by "SAM
    perception is ~10-20s of GPU work per request and the predictor is
    a process singleton". The first half of that stopped being true on
    2026-08-01: a pass costs 0.6-1.3s now, so the lock defends against
    a cost that no longer exists while the A40 sits at 17% SM.

    The second half was never a reason to serialise. Model weights are
    read-only and a torch forward has no mutable module state; the
    per-camera state lives in `continuous._SESSIONS`, keyed by camera.
    What DID need serialising is the face worker's multiprocessing
    pipe, and that now guards itself inside `FaceWorker.embed` — which
    is where it belongs, since a global gate protecting it by accident
    is a coincidence, not a design.

    Defaults to 1, i.e. exactly the old behaviour. Raise
    NEMARA_PERCEPTION_CONCURRENCY to admit more; that env var is the
    whole revert.
    """

    def __init__(self, n: int) -> None:
        self._sem = threading.BoundedSemaphore(max(1, n))
        self._active = 0
        self._mu = threading.Lock()

    def __enter__(self):
        self._sem.acquire()
        with self._mu:
            self._active += 1
        return self

    def __exit__(self, *exc) -> bool:
        with self._mu:
            self._active -= 1
        self._sem.release()
        return False

    def locked(self) -> bool:
        """True if any pass is in flight — the self-recycle guard reads
        this to avoid exiting mid-inference."""
        with self._mu:
            return self._active > 0


_PERCEPTION_CONCURRENCY = max(
    1, int(os.environ.get("NEMARA_PERCEPTION_CONCURRENCY", "1"))
)
_perception_lock = _Gate(_PERCEPTION_CONCURRENCY)

# Upload sanity caps (review S4): reject absurd requests before they
# reach decode; skip any decoded frame beyond a sane camera size (a
# crafted small JPEG can decode to gigabytes).
_MAX_FRAMES = 128
_MAX_FILE_BYTES = 2_500_000
_MAX_SIDE_PX = 2560


# ── raw frame wire (2026-08-13) ────────────────────────────────────────
# Format and rationale live in perception/rawwire.py — one module owns
# both directions so identity's pack and this parse cannot drift.
# Same auth, same caps. The JPEG routes stay for any off-box caller.
#
# Body ceilings (the multipart path's per-file cap, translated): a real
# observe batch is ≤40 native frames ≈ 250MB; a faces frame is one
# native image ≈ 6MB. Checked against Content-Length BEFORE
# request.body() buffers anything — the raw routes must not be the one
# door with no size gate on an authenticated-but-hostile caller.
_RAW_MAX_OBSERVE_BODY = 512 * 1024 * 1024
_RAW_MAX_FACES_BODY = 64 * 1024 * 1024


async def _read_raw_body(request: Request, cap: int) -> bytes:
    cl = request.headers.get("content-length")
    try:
        if cl is not None and int(cl) > cap:
            raise HTTPException(413, f"raw body over {cap} bytes")
    except ValueError:
        raise HTTPException(400, "bad content-length")
    data = await request.body()
    if len(data) > cap:
        raise HTTPException(413, f"raw body over {cap} bytes")
    return data


def _parse_raw_frames(
    data: bytes,
) -> tuple[dict, list[tuple[int, np.ndarray]]]:
    """rawwire.parse_frames with this server's caps; malformed → 400."""
    try:
        return rawwire.parse_frames(
            data, max_frames=_MAX_FRAMES, max_side_px=_MAX_SIDE_PX,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

# Mirrors _TrackState.last_face_check_frame's sentinel so the pod's
# face schedule replays the brain's arithmetic exactly: first attempt
# on the first frame a track appears, then every `face_interval`
# frames (orig-index space), capped at `face_cap` attempts.
_NEVER = -10_000

# VRAM-hygiene self-recycle (2026-07-21 leak incident): ~9h of session
# create/destroy fragmented the CUDA caching allocator to 25.5GB
# resident and stretched session-create from <1s to 9.5s. The server
# exits cleanly once it is BOTH old (uptime > SAM_RECYCLE_UPTIME_S)
# AND idle (no request for SAM_RECYCLE_IDLE_S) — the start_sam.sh
# supervisor loop restarts it, so the recycle is invisible to the
# backend. 0 disables.
_RECYCLE_UPTIME_S = float(os.environ.get("SAM_RECYCLE_UPTIME_S", "21600"))
_RECYCLE_IDLE_S = float(os.environ.get("SAM_RECYCLE_IDLE_S", "300"))
_start_ts = time.time()
_last_request_ts = time.time()


def _recycle_watchdog() -> None:
    if _RECYCLE_UPTIME_S <= 0:
        return
    while True:
        time.sleep(60)
        up = time.time() - _start_ts
        idle = time.time() - _last_request_ts
        if (
            up >= _RECYCLE_UPTIME_S
            and idle >= _RECYCLE_IDLE_S
            and not _perception_lock.locked()
        ):
            log.info(
                "perception.self_recycle uptime=%.0fs idle=%.0fs — "
                "exiting for supervisor restart (VRAM hygiene)",
                up, idle,
            )
            os._exit(0)


# Refuse to serve with faces on CPU when a GPU is present (default on).
# ORT does not raise when the CUDA EP fails to load — it warns and falls
# back, and that silence cost us a node-wide CPU wall before anyone
# noticed (2026-08-13: face on CPUExecutionProvider starved every other
# stage at 96-100% CPU while the 4090 sat at 0-13%). Exiting here keeps
# the supervisor restarting a loudly-broken node instead of the fleet
# quietly running at a tenth of its capacity. Set to 0 only for a
# deliberate CPU-faces session.
_REQUIRE_GPU_FACES = os.environ.get(
    "NEMARA_FACE_REQUIRE_GPU", "1",
).strip() not in ("0", "false", "")


@app.on_event("startup")
def _startup() -> None:
    global _ready
    t0 = time.time()
    provider = "worker-failed"
    try:
        provider = _face.start()
    except Exception:  # noqa: BLE001
        log.exception("perception.face_worker_start_failed (CPU-less?)")
        # A start-timeout can leave a child alive with a pending
        # 'ready' message — kill it or the pipe desyncs and one crop's
        # faces attach to the next crop (review C1).
        _face.stop()
    import torch

    if (
        _REQUIRE_GPU_FACES
        and torch.cuda.is_available()
        and not provider.startswith("CUDA")
    ):
        log.critical(
            "perception.face_on_cpu_with_gpu_present provider=%s — "
            "refusing to serve (NEMARA_FACE_REQUIRE_GPU=0 to override). "
            "Fix the image: the onnxruntime module lacks or cannot load "
            "its CUDA provider (CPU-wheel shadowing or missing cuDNN).",
            provider,
        )
        # Reap the face child FIRST: os._exit skips multiprocessing
        # cleanup, and on node E (2026-08-13) ~120 supervisor restarts
        # of this exact path orphaned ~90 CUDA processes until the GPU
        # itself OOMed. A loud refusal must not leak.
        _face.stop()
        os._exit(1)
    global _sam_error
    try:
        pred = sam_track.get_sam(_CKPT)
        # Warm the full path (CUDA kernel compile + temp-clip round
        # trip) so the first real /observe isn't a cold 30s.
        for _ in sam_track.track_stream(
            pred, [np.zeros((360, 640, 3), np.uint8)],
            concept="person", thr=0.4,
        ):
            pass
        _sam_error = None
    except Exception as exc:  # noqa: BLE001
        _sam_error = repr(exc)
        log.exception(
            "perception.sam_warm_failed — /health reports 503 until a "
            "load succeeds (a tracker-less perception must look sick)",
        )
    threading.Thread(target=_recycle_watchdog, daemon=True).start()
    _ready = True
    log.info(
        "perception.ready v2 warm=%.1fs ckpt=%s face_provider=%s",
        time.time() - t0, _CKPT, provider,
    )


@app.get("/health")
def health():
    import torch

    body = {
        "status": (
            "degraded" if _sam_error else ("ok" if _ready else "warming")
        ),
        "contract": "observe-v2",
        "cuda": torch.cuda.is_available(),
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "face_provider": _face.provider,
        # Face-model parity: the backend refuses to consume
        # observations embedded by a different model than its enrolled
        # gallery's (review G4) — expose the name so it can check.
        "face_model": _FACE_MODEL,
    }
    if _sam_error:
        # 503, not 200-with-a-sad-body: every existing consumer (boot
        # wait, gates, dashboards) checks with `curl -f` or status
        # string — both must fail when the tracker cannot load.
        body["sam_error"] = _sam_error
        return JSONResponse(body, status_code=503)
    return body


# Display-veto concept (2026-07-21, screen-face incident): measured on
# the incident frames — finds displays reliably; phantom containment
# 1.0 vs patient 0.13. Compound prompts win in SAM3 grounding (same
# lesson as the face countersign).
_SCREEN_CONCEPT = "television, laptop, or phone screen"


def _observe_sync(
    named_frames: list[tuple[int, "bytes | np.ndarray"]],
    person_thr: float,
    face_interval: int,
    face_cap: int,
    detect_screens: bool = False,
    camera_id: str = "default",
    client_stamp: str = "UNSTAMPED",
) -> dict:
    """The whole perception pass, run in a worker thread under
    `_perception_lock` so /health stays responsive while SAM grinds
    and the shared predictor + face-worker pipe stay single-flight.

    Each frame is either JPEG bytes (the multipart /observe path) or an
    already-decoded BGR ndarray (the /observe_raw path — identity lives
    on the same box, so shipping JPEG across loopback bought nothing
    and cost an encode+decode on every batch; measured 2026-08-13 it
    was a top CPU stage under multi-camera load)."""
    with _perception_lock:
        t_start = time.time()

        # ── decode, preserving original extracted-frame indices ────
        t0 = time.time()
        decoded: list[tuple[int, np.ndarray]] = []
        oversize = 0
        for orig_idx, raw in named_frames:
            if isinstance(raw, np.ndarray):
                img = raw
            else:
                buf = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is None:
                continue
            if max(img.shape[:2]) > _MAX_SIDE_PX:
                oversize += 1
                continue
            decoded.append((orig_idx, img))
        decoded.sort(key=lambda p: p[0])
        # Uniform-dims filter (reviews C2/E4): the SFU can flap
        # resolution mid-clip, and the mp4 writer would silently drop
        # non-matching frames — shifting SAM positions off pos_to_orig.
        # Dropped frames simply have no rows (Path A on the backend).
        dim_dropped = 0
        if decoded:
            ref = decoded[0][1].shape[:2]
            kept = [(i, im) for i, im in decoded if im.shape[:2] == ref]
            dim_dropped = len(decoded) - len(kept)
            decoded = kept
        if dim_dropped or oversize:
            log.warning(
                "perception.frames_dropped dims=%d oversize=%d",
                dim_dropped, oversize,
            )
        decode_ms = (time.time() - t0) * 1000.0
        if not decoded:
            return {"_error": (400, "no decodable frames")}

        imgs: list[np.ndarray | None] = [img for _, img in decoded]
        pos_to_orig = [orig for orig, _ in decoded]
        # Screen pass source — captured NOW because the propagation
        # loop releases frame pixels as it goes (imgs[pos] = None).
        screen_src = imgs[-1].copy() if detect_screens and imgs else None

        # ── SAM propagation with inline face attempts ──────────────
        # Masks live only inside each yielded frame, and each frame's
        # pixels are released after processing (`imgs[pos] = None`) —
        # memory ramps DOWN through the request instead of holding
        # ~240MB of frames for its whole life (review E3).
        rows_by_track: dict[int, list[list[float]]] = {}
        face_by_track: dict[int, list[list]] = {}
        face_sched: dict[int, dict[str, int]] = {}
        det_count = 0
        max_prob = 0.0
        prob_sum = 0.0
        face_ms_total = 0.0
        face_attempts = 0
        faces_found = 0

        global _sam_error
        try:
            pred = sam_track.get_sam(_CKPT)
        except Exception as exc:  # noqa: BLE001
            # Surfaced on /health as 503 (see below): a perception that
            # cannot load its tracker must look SICK, not "ok" — after
            # the 2026-08-13 L5 OOM this exact state served 500s for ~5
            # hours behind a green health check.
            _sam_error = repr(exc)
            log.exception("perception.sam_unavailable")
            return {"_error": (503, "tracker unavailable")}
        _sam_error = None
        t_loop = time.time()
        # CONTINUOUS SESSION (EdgeTAM-continuous arch): one persistent
        # per-camera session yields STABLE track ids across ticks (SAM3
        # periodic re-ground + EdgeTAM carry-propagate). Replaces the
        # stateless per-clip track_stream whose fresh ids renumbered the
        # subject across ticks and stranded the backend binding into a
        # ghost. Masks are held in `session_out` for this tick; frame
        # pixels (`imgs[pos]`) are still released after the face pass.
        try:
            session = continuous.get_session(camera_id)
        except continuous.SessionCapacityError as exc:
            log.warning("perception.session_capacity %s", exc)
            return {"_error": (429, str(exc))}
        session_out = session.observe(
            imgs, person_thr=person_thr, sam_pred=pred,
        )
        for pos in sorted(session_out):
            if pos >= len(pos_to_orig):
                continue
            orig = pos_to_orig[pos]
            img = imgs[pos]
            for tid, (bbox, mask) in session_out[pos].items():
                prob = 1.0
                det_count += 1
                prob_sum += prob
                if prob > max_prob:
                    max_prob = prob
                rows_by_track.setdefault(tid, []).append(
                    [orig, *[float(v) for v in bbox], float(prob)],
                )
                # Face schedule — the brain's arithmetic, verbatim.
                # face_cap=0 = attribution ineligible this tick (no
                # enrolled gallery): skip ALL face work (review G1).
                if face_cap <= 0 or img is None:
                    continue
                sched = face_sched.setdefault(
                    tid, {"last": _NEVER, "attempts": 0},
                )
                if (
                    sched["attempts"] < face_cap
                    and orig - sched["last"] >= face_interval
                ):
                    sched["last"] = orig
                    sched["attempts"] += 1
                    face_attempts += 1
                    crop = sam_track.mask_crop(img, bbox, mask)
                    if crop is None or crop.size == 0:
                        continue
                    hits, w_ms = _face.embed(crop)
                    face_ms_total += w_ms
                    x0, y0 = float(bbox[0]), float(bbox[1])
                    for emb, det, (fx1, fy1, fx2, fy2) in hits:
                        faces_found += 1
                        face_by_track.setdefault(tid, []).append([
                            orig, float(det),
                            fx1 + x0, fy1 + y0, fx2 + x0, fy2 + y0,
                            base64.b64encode(
                                emb.astype(np.float32).tobytes(),
                            ).decode("ascii"),
                        ])
            imgs[pos] = None
        loop_ms = (time.time() - t_loop) * 1000.0
        sam_ms = loop_ms - face_ms_total

        # ── SAM-face COUNTERSIGN (2026-07-21, B3 incident) ─────────
        # SCRFD confidently claims faces on the backs of heads; those
        # claims scored ~0.07 vs the gallery and testified the patient
        # OTHER. Before any claim ships, the face CONCEPT session
        # (SAM's semantic grounding) validates that a face exists
        # overlapping the claim. Unconfirmed claims are annotated
        # oracle_face=false — the brain treats them as silence, so the
        # original det bar keeps its guest-branding recall while
        # counterfeit testimony dies at the source. Runs only when
        # claims exist, only over the attempt frames (~5-10): ~1.5-2.5s
        # on face-active ticks, zero otherwise.
        confirm_ms = 0.0
        if face_by_track:
            t_fc = time.time()
            try:
                claim_frames = sorted({
                    int(fo[0]) for obs in face_by_track.values() for fo in obs
                })
                frame_by_idx = dict(named_frames)
                fc_imgs = []
                fc_orig = []
                for orig in claim_frames:
                    raw = frame_by_idx.get(orig)
                    if raw is None:
                        continue
                    im = (
                        raw if isinstance(raw, np.ndarray)
                        else cv2.imdecode(
                            np.frombuffer(raw, dtype=np.uint8),
                            cv2.IMREAD_COLOR,
                        )
                    )
                    if im is not None:
                        fc_imgs.append(im)
                        fc_orig.append(orig)
                face_masks: dict[int, list] = {}
                if fc_imgs:
                    # Prompt measured 2026-07-21 on live labeled frames
                    # (Aryaman's variant): "face" confirmed backs of
                    # heads 4/4; "face with visible eyes and nose"
                    # separated PERFECTLY — 0/4 on backs, 5/5 on real
                    # faces. The qualifier does carry weight in SAM3's
                    # concept grounding.
                    for pos, dets in sam_track.track_stream(
                        pred, fc_imgs,
                        concept="face with visible eyes and nose",
                        thr=0.3,
                    ):
                        if pos < len(fc_orig):
                            face_masks[fc_orig[pos]] = [
                                bbox for _, bbox, _, _ in dets
                            ]
                for obs in face_by_track.values():
                    for fo in obs:
                        orig = int(fo[0])
                        cx1, cy1, cx2, cy2 = fo[2], fo[3], fo[4], fo[5]
                        confirmed = any(
                            not (bx2 <= cx1 or bx1 >= cx2
                                 or by2 <= cy1 or by1 >= cy2)
                            for bx1, by1, bx2, by2 in face_masks.get(orig, [])
                        )
                        fo.append(bool(confirmed))
            except Exception:  # noqa: BLE001
                log.exception("perception.face_confirm_failed — claims "
                              "ship unconfirmed (treated as true)")
            confirm_ms = (time.time() - t_fc) * 1000.0

        # ── display-veto screen pass (on request only, ~0.5s) ──────
        screen_regions: list[list[float]] = []
        screens_ms = 0.0
        if screen_src is not None:
            t_sc = time.time()
            try:
                for _pos, dets in sam_track.track_stream(
                    pred, [screen_src], concept=_SCREEN_CONCEPT, thr=0.3,
                ):
                    for _tid, bbox, _m, p in dets:
                        screen_regions.append(
                            [*(float(v) for v in bbox), float(p)],
                        )
            except Exception:  # noqa: BLE001
                log.exception("perception.screen_pass_failed")
            screens_ms = (time.time() - t_sc) * 1000.0
        total_ms = (time.time() - t_start) * 1000.0

        payload = {
            "contract": "observe-v2",
        "epoch": session.epoch,
            "frames_seen": len(decoded),
            "frame_indices": pos_to_orig,
            "tracks": [
                {
                    "track_id": tid,
                    "rows": rows,
                    "face_obs": face_by_track.get(tid, []),
                }
                for tid, rows in sorted(rows_by_track.items())
            ],
            "stats": {
                "det_count": det_count,
                "max_prob": round(max_prob, 4),
                "mean_prob": (
                    round(prob_sum / det_count, 4) if det_count else 0.0
                ),
                "person_thr": person_thr,
                "face_attempts": face_attempts,
                "faces_found": faces_found,
                "face_provider": _face.provider,
                "face_model": _FACE_MODEL,
                "frames_dropped_dims": dim_dropped,
            },
            "timings_ms": {
                "decode": round(decode_ms, 1),
                "sam": round(sam_ms, 1),
                "face": round(face_ms_total, 1),
                "face_confirm": round(confirm_ms, 1),
                "screens": round(screens_ms, 1),
                "total": round(total_ms, 1),
            },
            "screen_regions": screen_regions,
        }
        log.info(
            "perception.observe client=%s frames=%d tracks=%d dets=%d max_prob=%.2f "
            "face_attempts=%d faces=%d | decode=%.0fms sam=%.0fms "
            "face=%.0fms total=%.0fms",
            client_stamp,
            len(decoded), len(rows_by_track), det_count, max_prob,
            face_attempts, faces_found, decode_ms, sam_ms, face_ms_total,
            total_ms,
        )
        # Return cached blocks to the driver after every pass (leak
        # incident 2026-07-21): costs ~0.1-0.3s of re-allocation on the
        # next tick; 19GB of allocator drift cost the whole service.
        import torch

        torch.cuda.empty_cache()
        return payload


@app.post("/faces")
async def faces(
    request: Request,
    frame: UploadFile = File(...),
    boxes: str = Form(...),
    full: str = Form("0"),
) -> JSONResponse:
    """Out-of-band face embedding on KNOWN person boxes (2026-07-24
    decoupling): no SAM, no session, no single-flight semaphore — the
    tracking cadence never pays for identity. `boxes` is JSON
    [[track_id, x1, y1, x2, y2], ...] in the sent frame's pixel space.
    ~25ms/crop on GPU. No mask-clean crop and no countersign here: a
    plain padded bbox crop is fine for SCORING (garbage crops score
    below the calibrated ramps); harvesting stays gated backend-side.

    `full=1` ADDITIONALLY detects faces on the WHOLE frame, returned
    with track_id=-1 (caller assigns). Why: the caller's boxes ride the
    tracking pipeline, which lags the world by 5-13s — during a close
    walk-past the box points at where the person WAS and the crop can
    miss the face entirely (live 2026-07-24). Full-frame detection is
    lag-immune for CLOSE faces (the detector's internal resize makes
    distant faces invisible full-frame — that's what the crops cover).
    Every hit carries cx/cy (face center, frame space) for assignment.
    """
    import json as _json

    _require_auth(request)
    if not _ready:
        raise HTTPException(503, "service warming up")
    raw = await frame.read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "undecodable frame")
    return _faces_impl(img, _json.loads(boxes), full == "1")


@app.post("/faces_raw")
async def faces_raw(request: Request) -> JSONResponse:
    """/faces over the raw frame wire: one frame, boxes and full ride
    the JSON header. Same reason /observe_raw exists — the probe loop
    was JPEG-encoding NATIVE-resolution frames every 1.5s while
    searching, for a loopback hop."""
    _require_auth(request)
    if not _ready:
        raise HTTPException(503, "service warming up")
    header, frames = _parse_raw_frames(
        await _read_raw_body(request, _RAW_MAX_FACES_BODY),
    )
    if len(frames) != 1:
        raise HTTPException(400, "faces_raw wants exactly one frame")
    return _faces_impl(
        frames[0][1],
        header.get("boxes") or [],
        bool(int(header.get("full", 0))),
    )


def _faces_impl(
    img: np.ndarray, boxes: list, full: bool,
) -> JSONResponse:
    t0 = time.time()
    out = []
    h, w = img.shape[:2]
    for row in boxes[:16]:
        tid, x1, y1, x2, y2 = row[:5]
        px, py = (x2 - x1) * 0.1, (y2 - y1) * 0.1
        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(w, int(x2 + px)), min(h, int(y2 + py))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        hits, _ms = _face.embed(crop)
        # STRICT attribution (2026-07-24): a padded crop of adjacent
        # people can catch the NEIGHBOR's head at its edge, and every
        # face found here used to be credited to this crop's track —
        # a wrong-person identity donation. Keep only faces whose
        # center lies inside the UNPADDED person box, and only the
        # largest such face (one person box = one face, at most).
        best = None
        for emb, det, (fx1, fy1, fx2, fy2) in hits:
            fcx, fcy = cx1 + (fx1 + fx2) * 0.5, cy1 + (fy1 + fy2) * 0.5
            if not (x1 <= fcx <= x2 and y1 <= fcy <= y2):
                continue
            if best is None or (fy2 - fy1) > best[2]:
                best = (emb, det, fy2 - fy1, fcx, fcy)
        if best is not None:
            emb, det, fpx, fcx, fcy = best
            out.append({
                "track_id": int(tid),
                "det": float(det),
                "face_px": float(fpx),
                "cx": float(fcx),
                "cy": float(fcy),
                "emb": base64.b64encode(
                    emb.astype(np.float32).tobytes(),
                ).decode("ascii"),
            })
    if full:
        hits, _ms = _face.embed(img)
        for emb, det, (fx1, fy1, fx2, fy2) in hits:
            out.append({
                "track_id": -1,
                "det": float(det),
                "face_px": float(fy2 - fy1),
                "cx": float((fx1 + fx2) * 0.5),
                "cy": float((fy1 + fy2) * 0.5),
                "emb": base64.b64encode(
                    emb.astype(np.float32).tobytes(),
                ).decode("ascii"),
            })
    return JSONResponse({
        "contract": "faces-v1",
        "faces": out,
        "ms": round((time.time() - t0) * 1000.0, 1),
    })


@app.post("/observe")
async def observe(
    request: Request,
    frames: list[UploadFile] = File(...),
    person_thr: float = Form(0.4),
    face_interval: int = Form(5),
    face_cap: int = Form(8),
    detect_screens: int = Form(0),
    camera_id: str = Form("default"),
) -> JSONResponse:
    """Run SAM perception on one tick's frames and return raw
    observations. Frame files MUST be named "<origidx>.jpg" — all
    outputs are keyed by that original index. Busy → instant 429 (the
    backend fails open). Admission is NEMARA_PERCEPTION_CONCURRENCY
    passes at a time (default 1); calls for the SAME camera serialise on
    that camera's session lock regardless, since a session's tick
    counter, id allocator and carried masks are read-modify-write."""
    # Before the frame reads: an unauthorized caller should not be able to
    # make us decode its uploads.
    _require_auth(request)
    if not _ready:
        raise HTTPException(503, "service warming up")
    if len(frames) > _MAX_FRAMES:
        raise HTTPException(400, f"too many frames (>{_MAX_FRAMES})")

    try:
        # SMALL admission queue (soak finding, 2026-07-21, 2nd
        # calibration): absorbs cadence JITTER only. A long (20s) wait
        # cascaded — when service time exceeds the tick period,
        # utilization >1 and queueing just converts 429s into client
        # read-timeouts while the pod burns GPU on abandoned requests.
        # The capacity fix lives on the backend (frame budget keeps
        # per-tick service below the tick period); this timeout sheds
        # genuine overload fast, preserving fail-open.
        await asyncio.wait_for(_busy.acquire(), timeout=_ACQUIRE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(429, "perception busy — retry next tick")
    try:
        named_jpegs: list[tuple[int, bytes]] = []
        for f in frames:
            try:
                orig_idx = int(
                    os.path.splitext(os.path.basename(f.filename))[0],
                )
            except (TypeError, ValueError):
                raise HTTPException(
                    400, f"frame filename not an index: {f.filename}",
                )
            raw = await f.read()
            if len(raw) > _MAX_FILE_BYTES:
                log.warning(
                    "perception.frame_too_big idx=%d bytes=%d — skipped",
                    orig_idx, len(raw),
                )
                continue
            named_jpegs.append((orig_idx, raw))

        client_stamp = request.headers.get("x-nemara-client", "UNSTAMPED")
        payload = await asyncio.to_thread(
            _observe_sync, named_jpegs, person_thr, face_interval, face_cap,
            bool(detect_screens), camera_id, client_stamp,
        )
    finally:
        global _last_request_ts
        _last_request_ts = time.time()
        _busy.release()
    if "_error" in payload:
        code, msg = payload["_error"]
        raise HTTPException(code, msg)
    return JSONResponse(payload)


@app.post("/observe_raw")
async def observe_raw(request: Request) -> JSONResponse:
    """/observe over the raw frame wire (format at _parse_raw_frames):
    identical perception pass, zero image-codec work on either end.
    Request params ride the JSON header. A separate route rather than a
    content-type switch because the multipart route's File(...) params
    make FastAPI reject a non-form body before our code runs."""
    _require_auth(request)
    if not _ready:
        raise HTTPException(503, "service warming up")
    data = await _read_raw_body(request, _RAW_MAX_OBSERVE_BODY)
    try:
        await asyncio.wait_for(_busy.acquire(), timeout=_ACQUIRE_TIMEOUT_S)
    except asyncio.TimeoutError:
        raise HTTPException(429, "perception busy — retry next tick")
    try:
        header, named_frames = _parse_raw_frames(data)
        client_stamp = request.headers.get("x-nemara-client", "UNSTAMPED")
        payload = await asyncio.to_thread(
            _observe_sync, named_frames,
            float(header.get("person_thr", 0.4)),
            int(header.get("face_interval", 5)),
            int(header.get("face_cap", 8)),
            bool(int(header.get("detect_screens", 0))),
            str(header.get("camera_id", "default")),
            client_stamp,
        )
    finally:
        global _last_request_ts
        _last_request_ts = time.time()
        _busy.release()
    if "_error" in payload:
        code, msg = payload["_error"]
        raise HTTPException(code, msg)
    return JSONResponse(payload)
