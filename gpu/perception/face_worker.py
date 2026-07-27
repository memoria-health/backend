"""Isolated face detect+embed worker — SCRFD + ArcFace on GPU without
the cuDNN clash (2026-07-20).

Why a separate PROCESS: onnxruntime-gpu and torch each bundle their
own cuDNN; imported into one process they fight and SCRFD fell back to
CPU (~1s per face check — the v1 latency bottleneck, ~5-6s of every
9-10s tick). In its own process, onnxruntime-gpu never shares an
address space with torch: SAM keeps the main process, faces run here
at GPU speed (~10-30ms per crop). Falls back to CPUExecutionProvider
transparently if CUDA init fails — slower, never broken.

Protocol (parent <-> child over multiprocessing Pipe):
  parent sends:  ("embed", crop_bgr_ndarray)   -> child replies
                 [(embedding_512_f32, det_score, (x1, y1, x2, y2)), ...]
                 (bbox in CROP coords; caller translates to frame)
  parent sends:  ("ping", None)                -> ("ok", provider_str)
  parent sends:  ("stop", None)                -> child exits

The child imports insightface INSIDE the process-entry function, after
setting its own env, so no heavy import leaks into the server process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time

import numpy as np

log = logging.getLogger("perception.face_worker")


def _real_provider(app) -> str | None:
    """Read the provider onnxruntime ACTUALLY selected. ORT does not
    raise when the CUDA EP fails to load with CPU in the list — it
    warns and falls back, so trusting the requested list over-reports
    GPU (review C6)."""
    try:
        model = next(iter(app.models.values()))
        return str(model.session.get_providers()[0])
    except Exception:  # noqa: BLE001
        return None


def _worker_main(conn, model_name: str, providers_env: str) -> None:
    """Child process entry. Owns insightface + onnxruntime(-gpu)."""
    providers = [p for p in providers_env.split(",") if p]
    provider_used = "uninitialized"
    app = None
    try:
        from insightface.app import FaceAnalysis  # heavy; child-only

        try:
            app = FaceAnalysis(name=model_name, providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            provider_used = _real_provider(app) or providers[0]
        except Exception:  # noqa: BLE001
            # CUDA EP failed (missing/mismatched libs) — CPU fallback.
            app = FaceAnalysis(
                name=model_name, providers=["CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            provider_used = "CPUExecutionProvider(fallback)"
    except Exception as exc:  # noqa: BLE001
        conn.send(("fatal", repr(exc)))
        return

    conn.send(("ready", provider_used))
    while True:
        try:
            kind, payload = conn.recv()
        except (EOFError, OSError):
            return
        if kind == "stop":
            return
        if kind == "ping":
            conn.send(("ok", provider_used))
            continue
        if kind == "embed":
            t0 = time.time()
            out = []
            try:
                faces = app.get(payload)
                for f in faces:
                    out.append((
                        f.normed_embedding.astype(np.float32),
                        float(f.det_score),
                        tuple(float(v) for v in f.bbox),
                    ))
                # MEASUREMENT capture (temporary, env-gated): dump per-
                # claim raw signals — det, 5 keypoints, RAW embedding
                # norm (ArcFace self-reported face-ness) — plus the crop,
                # so the face-vs-back-of-head gate is thresholded from
                # measured distributions, not guesses.
                dbg = os.environ.get("NEMARA_FACE_DEBUG_DIR")
                if dbg and faces:
                    import json
                    from pathlib import Path

                    d = Path(dbg)
                    d.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%H%M%S") + f"{time.time()%1:.3f}"[1:]
                    for i, f in enumerate(faces):
                        raw = getattr(f, "embedding", None)
                        rec = {
                            "t": time.time(),
                            "det": float(f.det_score),
                            "bbox": [float(v) for v in f.bbox],
                            "kps": (
                                [[float(a), float(b)] for a, b in f.kps]
                                if getattr(f, "kps", None) is not None
                                else None
                            ),
                            "raw_norm": (
                                float(np.linalg.norm(raw))
                                if raw is not None else None
                            ),
                        }
                        with open(d / "claims.jsonl", "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                        try:
                            import cv2 as _cv2
                            x1, y1, x2, y2 = [int(v) for v in f.bbox]
                            crop = payload[max(0, y1):y2, max(0, x1):x2]
                            if crop.size:
                                _cv2.imwrite(
                                    str(d / f"{stamp}_{i}.jpg"), crop,
                                )
                        except Exception:  # noqa: BLE001
                            pass
            except Exception:  # noqa: BLE001
                logging.exception("face_worker.embed_failed")
            conn.send(("faces", out, (time.time() - t0) * 1000.0))


class FaceWorker:
    """Parent-side handle. Lazy start, auto-restart on death, sync
    embed calls (the server's SAM loop is sync anyway)."""

    # Circuit breaker: after this many consecutive failures, stop
    # attempting restarts for the cooldown window — a broken face
    # stack must degrade to "no faces" instantly, not turn every
    # /observe into a multi-minute restart storm (reviews C8/S8).
    _BREAKER_THRESHOLD = 3
    _BREAKER_COOLDOWN_S = 120.0

    def __init__(
        self,
        *,
        model_name: str = "buffalo_l",
        providers: str = "CUDAExecutionProvider,CPUExecutionProvider",
        call_timeout_s: float = 20.0,
    ) -> None:
        self._model = model_name
        self._providers = providers
        self._timeout = call_timeout_s
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self.provider: str = "unstarted"
        self._fail_streak = 0
        self._cooldown_until = 0.0

    def start(self, *, timeout_s: float = 180.0) -> str:
        """Start (or restart) the child; block until model is ready.
        Returns the provider string actually in use. On timeout the
        child is KILLED before raising — a live child with a pending
        'ready' message would desync the pipe protocol and attach one
        crop's faces to the next crop's reply (review C1)."""
        self.stop()
        # NO LD_LIBRARY_PATH surgery: the pod image ships a system
        # CUDA/cuDNN that onnxruntime-gpu resolves correctly on its own
        # (measured ~100ms/attempt). Injecting torch's bundled cuDNN
        # here produced a version mismatch that slowed every face call
        # ~10x while the CUDA EP still reported loaded — if ort ever
        # fails to find cudnn on a new image, fix the image, don't
        # borrow torch's copy.
        parent, child = self._ctx.Pipe()
        self._proc = self._ctx.Process(
            target=_worker_main,
            args=(child, self._model, self._providers),
            daemon=True,
        )
        self._proc.start()
        child.close()
        self._conn = parent
        # First message is ready/fatal — model load can take ~10-30s
        # cold (weights + CUDA context), so poll generously at startup;
        # in-request restarts pass a much shorter budget.
        if parent.poll(timeout_s):
            kind, info = parent.recv()
            if kind == "ready":
                self.provider = info
                self._fail_streak = 0
                log.info("face_worker.ready provider=%s", info)
                return info
            self.stop()
            raise RuntimeError(f"face worker fatal: {info}")
        self.stop()
        raise RuntimeError("face worker start timeout")

    def _ensure(self) -> None:
        if self._proc is None or not self._proc.is_alive():
            log.warning("face_worker.restarting (dead or unstarted)")
            # In-request restart: short budget — a hung model load must
            # not stall the SAM loop for minutes (review C8).
            self.start(timeout_s=30.0)

    def _note_failure(self) -> None:
        self._fail_streak += 1
        if self._fail_streak >= self._BREAKER_THRESHOLD:
            self._cooldown_until = time.monotonic() + self._BREAKER_COOLDOWN_S
            log.error(
                "face_worker.breaker_open streak=%d cooldown=%.0fs",
                self._fail_streak, self._BREAKER_COOLDOWN_S,
            )

    def embed(self, crop_bgr: np.ndarray) -> tuple[list, float]:
        """[(embedding, det_score, bbox_crop_coords), ...], worker_ms.
        Empty list on worker failure (fail-open: a face-infra hiccup
        must degrade to 'no face seen this attempt', never crash the
        tick)."""
        if time.monotonic() < self._cooldown_until:
            return [], 0.0
        try:
            self._ensure()
            self._conn.send(("embed", crop_bgr))
            if self._conn.poll(self._timeout):
                msg = self._conn.recv()
                if msg[0] == "faces":
                    self._fail_streak = 0
                    return msg[1], float(msg[2])
                # Any other message here means the request/reply stream
                # is desynced (e.g. a stale 'ready'); continuing would
                # attach THIS crop's reply to the NEXT crop (review C1).
                log.error(
                    "face_worker.protocol_desync got=%r — killing child",
                    msg[0],
                )
                self.stop()
                self._note_failure()
            else:
                log.error("face_worker.timeout — killing child")
                self.stop()
                self._note_failure()
        except Exception:  # noqa: BLE001
            log.exception("face_worker.embed_call_failed")
            self.stop()
            self._note_failure()
        return [], 0.0

    def stop(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(("stop", None))
            except Exception:  # noqa: BLE001
                pass
            self._conn.close()
            self._conn = None
        if self._proc is not None:
            self._proc.join(timeout=2.0)
            if self._proc.is_alive():
                self._proc.kill()
            self._proc = None
