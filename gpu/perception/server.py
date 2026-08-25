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
import contextlib
import hmac
import logging
import os
import pathlib
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
def _load_auth_token() -> str:
    """The shared bearer, from the environment or from a file.

    The file path exists because the node's env is fixed at container
    start and the entrypoint sources nothing from the volume — so setting
    this via RunPod means recycling the pod, while the dev-code overlay
    can ship code in seconds. A file on the shared volume can be written
    once and picked up by the next service restart, which is what makes
    turning auth ON not require taking the node down.
    """
    env = os.environ.get("NEMARA_PERCEPTION_TOKEN", "").strip()
    if env:
        return env
    path = os.environ.get(
        "NEMARA_PERCEPTION_TOKEN_FILE", "/persist/secrets/perception_token")
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


_AUTH_TOKEN = _load_auth_token()


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


def _require_admin_auth(request: Request) -> None:
    """Same, but an UNSET token refuses instead of allowing.

    An empty token disabling auth is a defensible default for /observe:
    the pod binds loopback and the tunnel is the boundary. It is not
    defensible for a route that kills the process. With the pod on the
    tailnet and the token unset, anything that could reach the port could
    restart it, and the boot warning only ever mentioned /observe and
    /faces so nobody would have known.

    Fails CLOSED: no token configured means the admin surface does not
    exist, rather than existing for everyone.
    """
    if not _AUTH_TOKEN:
        raise HTTPException(
            status_code=404,
            detail="admin surface disabled (NEMARA_PERCEPTION_TOKEN unset)",
        )
    _require_auth(request)


if not _AUTH_TOKEN:
    log.warning(
        "perception.auth_disabled — NEMARA_PERCEPTION_TOKEN is unset, so "
        "/observe and /faces accept EVERY caller and the /admin routes are "
        "refused outright. Safe only while bound to loopback (host=%s). "
        "Set the token before exposing this port.",
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


def _int_set(raw) -> "set[int]":
    """Track ids off the wire. Anything malformed is simply not in the
    set — this only ever suppresses a COUNT, so a bad value costs one
    unnecessary re-look, never a lost person."""
    out: "set[int]" = set()
    if not isinstance(raw, (list, tuple)):
        return out
    for v in raw[:64]:
        try:
            out.add(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _lever_on(name: str, default: str = "1") -> bool:
    """Levers default ON since 2026-08-24, and here is why.

    These were `NEMARA_EXP_*` experiment flags in the 2026-08-16
    concurrency work, proved there, applied by patching a pod in place,
    and never promoted. So every pod recreate silently dropped them —
    including one today — and the fleet quietly ran a configuration that
    had never been measured. Nothing reported the difference: capacity
    just looked worse than the accepted result and nobody could say why.

    Opt-OUT now (`NEMARA_SDP_PIN=0`), so the validated configuration is
    what a fresh pod boots with and turning it off is a deliberate act.

    The plan doc still names these `NEMARA_EXP_SDP_PIN` /
    `NEMARA_EXP_SAM3_NO_GC`, which nothing has ever read — set those and
    you get no lever and no error. Both spellings are accepted here so
    following the older runbook cannot silently do nothing.
    """
    for env_name in (name, name.replace("NEMARA_", "NEMARA_EXP_")):
        raw = os.environ.get(env_name, "").strip().lower()
        if raw:
            return raw in ("1", "true", "yes", "on")
    return default in ("1", "true")


@app.on_event("startup")
def _startup() -> None:
    global _ready
    # ── Speed levers (2026-08-16, survivors of the corruption hunt).
    # Neither fixes the in-process overlap corruption (falsified) —
    # they cut per-pass overhead, and are safe because each worker
    # process is single-flight.
    #
    # No-gc lever: SAM3's close_session runs gc.collect() after EVERY
    # ground plus a device-wide torch.cuda.empty_cache() under VRAM
    # pressure — a whole-GPU stall per ground. Stub both process-wide.
    # Needs a memory-growth soak before becoming fleet default.
    if _lever_on("NEMARA_SAM3_NO_GC"):
        import gc as _gc

        import torch as _torch
        _gc.collect = lambda *a, **k: 0
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache = lambda *a, **k: None
        log.warning(
            "lever: gc.collect + cuda.empty_cache stubbed process-wide",
        )
    # SDP-pin lever: both models write the process-global SDPA backend
    # enables on the hot attention path (measured as a large cadence
    # tax). Pin all three ON once, no-op the setters; EdgeTAM's
    # flag-flipping context is neutralized in continuous._get_edge
    # (same env).
    if _lever_on("NEMARA_SDP_PIN"):
        import torch as _torch
        _torch.backends.cuda.enable_flash_sdp(True)
        _torch.backends.cuda.enable_math_sdp(True)
        _torch.backends.cuda.enable_mem_efficient_sdp(True)
        _torch.backends.cuda.enable_flash_sdp = lambda *a, **k: None
        _torch.backends.cuda.enable_math_sdp = lambda *a, **k: None
        _torch.backends.cuda.enable_mem_efficient_sdp = lambda *a, **k: None
        # The setters are NOT the only door, and pinning them was
        # therefore never the pin it claimed to be.
        #
        # EdgeTAM wraps every attention op in
        # `torch.backends.cuda.sdp_kernel(enable_flash=..., enable_math=...)`
        # (sam2/modeling/sam/transformer.py:32). On torch 2.4.1 that is
        # deprecated and forwards to `torch.nn.attention.sdpa_kernel`,
        # which installs a backend LIST through a different path and
        # never touches the functions patched above. So EdgeTAM could
        # still turn math SDP off process-wide, and a SAM3 attention
        # running in another camera's thread at that moment found no
        # enabled backend it could use: "No available kernel."
        #
        # Measured 2026-08-25: three perception restarts gave 126, 27
        # and 0 of those in the first minute — the signature of a race,
        # not of a config. `continuous._get_edge` already neutralises
        # EdgeTAM's context, but only from the first EdgeTAM load, which
        # is exactly the window the burst lives in.
        #
        # Both spellings, because the deprecated one imports the new one
        # at call time and would otherwise walk straight past this.
        def _nullctx(*_a, **_k):
            return contextlib.nullcontext()
        _torch.backends.cuda.sdp_kernel = _nullctx
        try:
            import torch.nn.attention as _tna
            _tna.sdpa_kernel = _nullctx
        except ImportError:      # older torch: the new spelling is absent
            pass
        log.warning(
            "lever: SDPA backend flags pinned all-on, setters AND the "
            "sdp_kernel/sdpa_kernel context managers no-oped",
        )
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


def _gpu_op_ok() -> "str | None":
    """Run one trivial op on the GPU. None on success, the error otherwise.

    `torch.cuda.is_available()` asks the driver whether a device exists.
    It keeps saying yes after the CUDA context has been poisoned — an OOM,
    an Xid, a device-side assert — which is the failure mode that actually
    takes a pod out. So health said `cuda: true` while every observe
    raised, and the pod stayed in the roster being handed work it could
    not do.

    An add on two 1-element tensors plus a synchronize is enough: the
    sync is the part that surfaces a poisoned context, because launches
    are async and the error is only reported when we wait for it.
    """
    import torch

    if not torch.cuda.is_available():
        return "cuda_unavailable"
    try:
        t = torch.ones(1, device="cuda")
        _ = (t + t).item()          # .item() syncs
        torch.cuda.synchronize()
    except Exception as exc:        # noqa: BLE001 — any failure is a failure
        return f"{type(exc).__name__}: {exc}"
    return None


@app.get("/health")
def health():
    import torch

    gpu_error = _gpu_op_ok()
    body = {
        "status": (
            "degraded" if (_sam_error or gpu_error)
            else ("ok" if _ready else "warming")
        ),
        "contract": "observe-v2",
        # Proven by running something, not by asking the driver.
        "cuda": gpu_error is None,
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else None
        ),
        "face_provider": _face.provider,
        # Face-model parity: the backend refuses to consume
        # observations embedded by a different model than its enrolled
        # gallery's (review G4) — expose the name so it can check.
        "face_model": _FACE_MODEL,
        # The display veto's health. `ok=0` with `fail>0` means every
        # screen pass is throwing — tracking looks fine, /health said
        # "ok", and meanwhile every TV and every framed face in the
        # fleet is minting phantom people. That state ran undetected
        # until 2026-08-24; it is on the surface now so a monitor can
        # fail on it.
        "screen_pass": {
            "ok": _SCREEN_FAIL["ok"], "failed": _SCREEN_FAIL["n"],
            "last_error": _SCREEN_FAIL["last"] or None,
            "last_found": _SCREEN_FAIL["found"],
            # The LIVE values, not the constants. This reads the same
            # override files the pass reads, because reporting the
            # defaults meant /health said thr=0.3 for a day while every
            # pass ran at 0.05 from /persist/state/screen_thr. The one
            # number the veto depends on was unobservable from outside
            # the box, and the disagreement was only found by reading
            # both by hand (2026-08-25).
            "concepts": list(_screen_config()[0]),
            "thr": _screen_config()[1],
        },
        # Which levers this process actually booted with. The 2026-08-16
        # capacity result was measured WITH these; a pod recreate used
        # to drop them silently and nobody could tell by looking.
        "levers": {
            "sdp_pin": _lever_on("NEMARA_SDP_PIN"),
            "sam3_no_gc": _lever_on("NEMARA_SAM3_NO_GC"),
            "concurrency": _PERCEPTION_CONCURRENCY,
        },
    }
    if _sam_error or gpu_error:
        # 503, not 200-with-a-sad-body: every existing consumer (boot
        # wait, gates, dashboards) checks with `curl -f` or status
        # string — both must fail when the tracker cannot load, and the
        # same has to be true when the GPU itself is gone.
        if _sam_error:
            body["sam_error"] = _sam_error
        if gpu_error:
            body["gpu_error"] = gpu_error
        return JSONResponse(body, status_code=503)
    return body


# ── Experiment ergonomics (2026-08-16, Aryaman: kill the 30-min
# debug loop). /admin/exit restarts this worker under its entrypoint
# supervisor — which re-copies /persist/dev-code first when
# NEMARA_DEV_CODE=1 — so a code iteration is rsync + curl, no image
# build, no SSH. /logs tails the shared log directory over HTTP so
# scoring needs no log-reader pod. Both behind the same bearer auth
# as /observe.
_LOGDIR = os.environ.get("NEMARA_LOGDIR", "/persist/state/logs")


@app.post("/admin/exit")
async def admin_exit(request: Request) -> JSONResponse:
    """Exit(0) shortly after replying; the entrypoint supervisor
    relaunches (fresh dev code first when NEMARA_DEV_CODE=1)."""
    _require_admin_auth(request)

    def _die() -> None:
        time.sleep(0.3)
        os._exit(0)

    threading.Thread(target=_die, daemon=True).start()
    return JSONResponse({"status": "restarting"})


@app.get("/logs")
async def logs(request: Request, file: str = "", tail: int = 400):
    """List log files (no `file`) or tail one. `file` is a bare name
    inside the log dir — path components are rejected."""
    _require_auth(request)
    import pathlib
    root = pathlib.Path(_LOGDIR)
    if not file:
        try:
            entries = sorted(
                root.iterdir(), key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:50]
        except OSError:
            raise HTTPException(503, "log dir unavailable")
        return {"files": [
            {"name": p.name, "bytes": p.stat().st_size,
             "mtime": int(p.stat().st_mtime)}
            for p in entries if p.is_file()
        ]}
    if "/" in file or "\\" in file or file.startswith("."):
        raise HTTPException(400, "bare filenames only")
    path = root / file
    if not path.is_file():
        raise HTTPException(404, "no such log")
    tail = max(1, min(int(tail), 20000))
    from fastapi.responses import PlainTextResponse

    def _tail() -> str:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            # ~200 bytes/line generous estimate, capped at 32MB
            take = min(size, min(tail * 200, 32_000_000))
            fh.seek(size - take)
            data = fh.read()
        lines = data.decode(errors="replace").splitlines()
        return "\n".join(lines[-tail:]) + "\n"

    return PlainTextResponse(await asyncio.to_thread(_tail))


# Display-veto concept (2026-07-21, screen-face incident): measured on
# the incident frames — finds displays reliably; phantom containment
# 1.0 vs patient 0.13. Compound prompts win in SAM3 grounding (same
# lesson as the face countersign).
#
# WIDENED 2026-08-24 to rendered people that are not on a screen. The
# TV Room's wall of ~12 painted faces minted up to SEVEN phantom person
# tracks at once (`covered=1` on every probe, so exactly one was real),
# each flicker birthing a fresh track and paying a fresh ground —
# measured at ~1.3s against ~130ms to propagate an existing one. That
# is what made a second camera look like a capacity problem.
#
# The veto machinery does not care what kind of surface it is: a
# rendered person is pinned inside the region on EVERY observed frame
# while a real one crossing in front dips out within a clip. A painting
# is more furniture than a television is, so the 120s refresh is if
# anything too eager for it.
#
# Deliberately NOT "picture" or "image" alone — both ground far too
# broadly and would swallow the room. Framed wall art is the concept.
_SCREEN_FAIL = {"n": 0, "ok": 0, "last": "", "found": -1}

# TUNABLE, because it has to be. The pass now runs and finds NOTHING —
# `found=0` on a bedroom frame with a large television in it — and there
# is no baseline for the original wording on this build, since it threw
# on every call until today. Deciding between "my longer wording diluted
# the grounding" and "grounding was never finding these anyway" needs
# both tried against the same live frame, and a container rebuild per
# attempt makes that a day's work instead of ten minutes.
#
# The threshold is exposed for the same reason: 0.3 was chosen against a
# laptop screen filling much of the frame, and a framed picture on a far
# wall is a much weaker match.
# ONE CONCEPT PER PASS, ROTATED — not one compound string.
#
# Measured 2026-08-24 on the live TV Room frame: "television" finds the
# bedroom television; "television screen or framed picture hanging on a
# wall" finds NOTHING, on either camera. Adding words destroyed a
# detection that worked. The old comment here claimed compound prompts
# win in SAM3 grounding; on this model and these frames the opposite is
# true, so the list is the fix and each entry stays short.
#
# Rotated rather than run together because the pass happens INSIDE the
# analysis call, holding the GPU slot, and costs ~0.5s. Three at once is
# a 1.5s stall every couple of minutes on a rig whose normal step is
# 100-500ms. One per pass costs exactly what today costs and simply
# re-confirms each object more slowly — which is free, because these
# things are furniture. ScreenMap's TTL is 900s against a 360s full
# cycle at three concepts, so a region never expires mid-rotation.
#
# WHAT IS ACTUALLY IN THESE ROOMS (looked at the frame, 2026-08-24):
# the TV Room's phantom faces come from ONE LARGE UNFRAMED CANVAS of
# painted faces, not framed pictures — which is why every wording built
# around "framed" was describing the wrong object. There are also pink
# figurative statues on a console table, the next most likely thing to
# be read as a person.
# MEASURED against the live rooms, 2026-08-24. Short beats descriptive,
# every time:
#
#   "television"                                        -> 1
#   "television or computer screen"                     -> 2
#   "painting"                                          -> 11
#   "television screen or framed picture hanging on a wall" -> 0
#   "painted artwork on a wall" / "mural" / "art" / "poster" -> 0
#
# The last two lines are why the earlier attempts failed: adding words
# destroyed detections that a single word found, and "framed" described
# the wrong object entirely — the TV Room's phantom faces come from ONE
# LARGE UNFRAMED CANVAS, not framed pictures.
#
# "painting" also caught the one genuinely framed picture by the door,
# which is the behaviour we want and did not have to ask for.
#
# "statue" was in this rotation and is gone (Aryaman 2026-08-25). It
# never returned a region on the room's pink figurative statues in any
# wording tried, so it spent a third of every rotation asking a question
# with no answer and slowed the two concepts that do work. Statues going
# unvetoed is the accepted cost.
_SCREEN_CONCEPTS_DEFAULT = (
    "television or computer screen",
    "painting",
)
# 0.05, not 0.3. Measured on the TV Room wall 2026-08-24: at 0.3 the
# grounding pass returned nothing and the veto never engaged, so the
# painted faces kept generating tracks — 6-8 of them, and a pod time of
# ~1.3s every batch. At 0.05 the same wall came back as regions, tracks
# went to 0 and the pod to 162ms.
#
# The node has run 0.05 since that afternoon, written into
# /persist/state/screen_thr by hand. This constant said 0.3 the whole
# time, which meant the committed code and the running code disagreed
# about the one number the feature depends on, and deleting the override
# to "restore the defaults" would have turned the veto off (2026-08-25).
_SCREEN_THR_DEFAULT = 0.05
# A FILE on the shared volume, re-read per pass — same idea as
# _load_auth_token above. The pass runs every couple of minutes and
# "found nothing" is indistinguishable from "wrong wording" until you
# can try another one, and a container rebuild per attempt turns a
# ten-minute question into a day. Write one line and the next pass uses
# it; delete the file and the default returns.
#
#   echo 'television or framed picture' > /persist/state/screen_concept
#   echo '0.15'                         > /persist/state/screen_thr
_SCREEN_CONCEPT_FILE = "/persist/state/screen_concept"
_SCREEN_THR_FILE = "/persist/state/screen_thr"


def _read_line(path: str, whole: bool = False) -> str:
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
        return text if whole else text.strip()
    except OSError:
        return ""


# One turn counter PER CAMERA, not one for the whole process.
#
# A single global counter looks equivalent and is not. Each camera makes
# one screen pass per rotation, so a shared counter hands out concepts
# by arrival order — and when the number of cameras shares a factor with
# the number of concepts, every camera lands on the same index forever.
#
# Measured 2026-08-25, and caused by this file: the rotation was
# ("television or computer screen", "painting", "statue") with two
# cameras. 3 and 2 are coprime, so both cameras saw all three concepts
# over time and it worked. Dropping "statue" that morning — because it
# never grounded anything — left 2 concepts and 2 cameras, and the
# parity locked instantly:
#
#     bedroom  kind=static  every pass    (painting, forever)
#     TV Room  kind=screen  every pass    (never painting, forever)
#
# The TV Room's painted faces could no longer be grounded, its regions
# aged out (live 12 -> 11 -> 0), and the phantom people came back. The
# concept list is now free to be any length without the camera count
# silently deciding what each camera is allowed to look for.
_screen_turn: dict[str, int] = {}


def _screen_config() -> tuple[tuple[str, ...], float]:
    """(every concept in the rotation, threshold), resolved the way the
    pass resolves them: file first, then env, then the defaults. The
    file may hold one concept per line.

    Split out from `_screen_params` so /health can report what is
    actually in force without advancing the rotation — reading it used
    to mean skipping a concept.
    """
    raw = _read_line(_SCREEN_CONCEPT_FILE, whole=True) or os.environ.get(
        "NEMARA_SCREEN_CONCEPT", "")
    concepts = tuple(
        c.strip() for c in raw.splitlines() if c.strip()
    ) or _SCREEN_CONCEPTS_DEFAULT
    thr_raw = _read_line(_SCREEN_THR_FILE) or os.environ.get(
        "NEMARA_SCREEN_THR", "")
    try:
        thr = float(thr_raw) if thr_raw else _SCREEN_THR_DEFAULT
    except ValueError:
        thr = _SCREEN_THR_DEFAULT
    return concepts, thr


def _screen_params(camera_id: str = "default") -> tuple[str, float]:
    """(concept for THIS camera's next pass, threshold).

    Advances only THIS camera's turn, so every camera walks the whole
    rotation regardless of how many cameras or concepts there are.
    """
    concepts, thr = _screen_config()
    turn = _screen_turn.get(camera_id, 0)
    _screen_turn[camera_id] = turn + 1
    return concepts[turn % len(concepts)], thr


def _observe_sync(
    named_frames: list[tuple[int, "bytes | np.ndarray"]],
    person_thr: float,
    face_interval: int,
    face_cap: int,
    detect_screens: bool = False,
    camera_id: str = "default",
    client_stamp: str = "UNSTAMPED",
    session_state: "dict | None" = None,
    not_people: "set[int] | None" = None,
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
        #
        # STATELESS MODE (2026-08-16, shared-queue dispatch): when the
        # request carries `session_state`, the CALLER owns the camera's
        # cross-tick state — build a transient session from it, run the
        # tick, hand the updated state back in the response. No entry in
        # _SESSIONS, so any worker can serve any camera. The request
        # shape (not an env var) selects the mode: identity turns it on
        # per-request, workers need no coordinated config.
        stateless = session_state is not None
        if stateless:
            session = continuous.CameraSession.from_state(
                camera_id, session_state or None,
            )
        else:
            session = continuous.get_session(camera_id)
        session_out = session.observe(
            imgs, person_thr=person_thr, sam_pred=pred,
            not_people=not_people,
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
            _concept, _thr = _screen_params(camera_id)
            t_sc = time.time()
            try:
                for _pos, dets in sam_track.track_stream(
                    pred, [screen_src], concept=_concept, thr=_thr,
                ):
                    for _tid, bbox, _m, p in dets:
                        screen_regions.append(
                            [*(float(v) for v in bbox), float(p)],
                        )
            except Exception as exc:  # noqa: BLE001
                # COUNTED, not just logged. This threw on every attempt
                # for an unknown length of time — `RuntimeError: No
                # available kernel` out of the text encoder, because the
                # SDPA pin was off and the two models were fighting over
                # the process-global backend flags. A dead veto is
                # invisible from outside: tracking still works, /health
                # still says ok, and TVs and wall art quietly mint
                # phantom people on every camera in the fleet. It rides
                # /health now so a monitor can see it.
                _SCREEN_FAIL["n"] += 1
                _SCREEN_FAIL["last"] = f"{type(exc).__name__}: {exc}"[:200]
                log.exception("perception.screen_pass_failed n=%d",
                              _SCREEN_FAIL["n"])
            else:
                _SCREEN_FAIL["ok"] += 1
                _SCREEN_FAIL["found"] = len(screen_regions)
                log.warning(
                    "perception.screen_pass concept=%r thr=%.2f found=%d",
                    _concept, _thr, len(screen_regions),
                )
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
            # YOLO's person boxes for this tick, already NORMALISED to
            # fractions of the frame they were measured in — a second,
            # independent opinion on which tracks are really people.
            # Empty means "YOLO saw nobody" AND "YOLO is unavailable";
            # the consumer treats both as no-signal, so a dead detector
            # degrades to the behaviour we had before this existed.
            "person_boxes": [
                [round(v, 5) for v in b]
                for b in getattr(session, "person_boxes_frac", ())
            ],
            "screen_regions": screen_regions,
            # WHICH concept found them. A screen can show a moving
            # person and a painting cannot, so the consumer must be
            # able to tell the two kinds of region apart — see
            # display_veto.veto_track_ids.
            "screen_concept": (_concept if screen_src is not None else ""),
            # The frame these boxes were MEASURED IN, so the consumer
            # can normalise them without guessing.
            #
            # It used to guess, and got it wrong whenever the SFU
            # changed rung: identity divided by the last size its INTAKE
            # reader had seen, while the regions came from `imgs[-1]` of
            # a batch that could be a different size entirely. A
            # painting at 0.82 of a 640-wide grounding frame is pixel
            # 525; divided by 2304 that is 0.23, so every region landed
            # in the empty top-left of the frame and the veto covered
            # nothing (2026-08-25, TV Room: 16 regions grounded, painted
            # faces still tracked).
            "screen_frame_wh": (
                [int(screen_src.shape[1]), int(screen_src.shape[0])]
                if screen_src is not None else None
            ),
        }
        if stateless:
            # The caller owns this camera's continuity — hand back the
            # updated snapshot; nothing about it remains in this worker.
            payload["session_state"] = session.to_state()
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
        # Under the no-gc lever (NEMARA_SAM3_NO_GC) empty_cache is
        # a process-wide no-op and this call is inert by design.
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
        # session_state: {} = "stateless mode, first tick"; a populated
        # dict = continue from it; absent = stateful mode (_SESSIONS).
        _sess = header.get("session_state")
        payload = await asyncio.to_thread(
            _observe_sync, named_frames,
            float(header.get("person_thr", 0.4)),
            int(header.get("face_interval", 5)),
            int(header.get("face_cap", 8)),
            bool(int(header.get("detect_screens", 0))),
            str(header.get("camera_id", "default")),
            client_stamp,
            _sess if isinstance(_sess, dict) else None,
            _int_set(header.get("not_people")),
        )
    finally:
        global _last_request_ts
        _last_request_ts = time.time()
        _busy.release()
    if "_error" in payload:
        code, msg = payload["_error"]
        raise HTTPException(code, msg)
    return JSONResponse(payload)
