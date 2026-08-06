"""Backend-side client for the GPU SAM perception microservice v2.

Behind `reid_pipeline_backend == "sam"`, inference.py POSTs one tick's
frames to the pod's /observe and parses the raw-observation response
into `RemoteObservations` for `RemoteReplayPipeline` — which replays
them through the UNCHANGED BodyReIDPipeline brain. The pod makes zero
identity decisions, so this client carries no gallery, no labels, no
thresholds (v1's /gallery push is gone — the enrolled gallery never
leaves the backend).

Wire efficiency (v1 lesson): the extracted frames are ALREADY JPEGs
(base64 in ExtractedFrame). v1 decoded + re-encoded each one on the
Mac (~a third of the backend overhead); v2 base64-decodes and ships
the JPEG bytes as-is, named "<origidx>.jpg" so every observation
comes back keyed by original frame index — no remap layer.

A network/remote failure raises — inference.py catches it and returns
IdentityResult.empty (reid_ran=False), honoring the safety invariant
that a Re-ID infrastructure failure must never silence detection.
"""

from __future__ import annotations

import base64
import logging

import httpx
import numpy as np

from identity.wire import RemoteFaceObs, RemoteObservations

log = logging.getLogger(__name__)

_EMB_DIM = 512


# Payload sanity caps (security review S5): the brain consumes rows
# into per-track state; a hostile/corrupt pod response must not be
# able to allocate unbounded backend memory before fail-open kicks in.
_MAX_TRACKS = 64
_MAX_ROWS_PER_TRACK = 1024
_MAX_FACE_OBS_PER_TRACK = 64


class SamServiceClient:
    """One long-lived client (holds the httpx connection pool).
    Construct once per process and reuse across ticks.

    `expected_face_model`: refuse observations embedded by a different
    face model than the backend's enrolled gallery (review G4 — a
    misconfigured pod would otherwise silently harvest foreign-space
    vectors into the persistent DB gallery both arms share)."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_s: float,
        expected_face_model: str | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        # connect=5: a black-holed pod (firewall drop, dead tunnel)
        # must cost ~5s before fail-open, not the full read timeout
        # every tick (review E2).
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(
            connect=5.0, read=timeout_s, write=timeout_s, pool=timeout_s,
        ))
        # Client stamp (2026-07-27): every request identifies WHO is
        # driving the GPU — the pod logs it per batch, so a stale or
        # foreign client is visible in one grep instead of a forensic
        # nvidia-smi session.
        import socket
        self._client.headers["X-Nemara-Client"] = (
            f"identity-service@{socket.gethostname()}"
        )
        self._expected_face_model = expected_face_model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        resp = await self._client.get(f"{self._base}/health")
        resp.raise_for_status()
        return resp.json()

    async def faces(
        self,
        jpeg: bytes,
        boxes: list[tuple[int, float, float, float, float]],
        *,
        full: bool = False,
    ) -> list[tuple[int, "np.ndarray", float, float, float, float]]:
        """Out-of-band face embeddings for KNOWN person boxes (the
        2026-07-24 decoupling — tracking batches never pay for
        identity). `full=True` also detects faces on the whole frame
        (track_id=-1, caller assigns) — lag-immune for close faces.
        Returns (track_id, embedding, face_px, det, cx, cy) with cx/cy
        the face center in the sent frame's pixel space."""
        import json as _json

        resp = await self._client.post(
            f"{self._base}/faces",
            data={
                "boxes": _json.dumps([list(b) for b in boxes]),
                "full": "1" if full else "0",
            },
            files=[("frame", ("f.jpg", jpeg, "image/jpeg"))],
        )
        resp.raise_for_status()
        j = resp.json()
        if j.get("contract") != "faces-v1":
            raise RuntimeError(
                f"sam wire: wrong /faces contract {j.get('contract')!r}",
            )
        out = []
        for f in j.get("faces", [])[:64]:
            emb = np.frombuffer(
                base64.b64decode(f["emb"]), dtype=np.float32,
            )
            if emb.shape != (_EMB_DIM,):
                continue
            out.append((
                int(f["track_id"]), emb,
                float(f["face_px"]), float(f["det"]),
                float(f.get("cx", 0.0)), float(f.get("cy", 0.0)),
            ))
        return out

    async def observe(
        self,
        jpegs_by_index: list[tuple[int, bytes]],
        *,
        camera_id: str,
        person_thr: float,
        face_interval: int,
        face_cap: int,
        detect_screens: bool = False,
        capture_ts: "dict[int, float] | None" = None,
    ) -> RemoteObservations:
        """POST ordered (orig_idx, jpeg_bytes) frames; parse the raw
        observations. Raises on transport/HTTP errors (fail-open is
        the caller's job). `detect_screens` asks the pod for one
        screen-concept pass on the newest frame (display-veto map
        refresh, ~0.5s).

        `capture_ts` maps frame index → this host's `time.time()` when
        the frame arrived. The pod does not compute with it; it only
        stamps retained masks so the censor can find the masks belonging
        to a clip window in the clock the clip was timed in. Omitted
        (old caller, or a caller with no stamps), the pod falls back to
        its own processing time and the censor's matching gets sloppier
        by the pipeline lag — degraded, never wrong."""
        import json as _json

        data = {
            "camera_id": camera_id,
            "person_thr": str(person_thr),
            "face_interval": str(face_interval),
            "face_cap": str(face_cap),
            "detect_screens": "1" if detect_screens else "0",
        }
        if capture_ts:
            data["capture_ts"] = _json.dumps(
                {str(k): float(v) for k, v in capture_ts.items()},
            )
        files = [
            ("frames", (f"{idx:05d}.jpg", jpeg, "image/jpeg"))
            for idx, jpeg in jpegs_by_index
        ]
        resp = await self._client.post(
            f"{self._base}/observe", data=data, files=files,
        )
        resp.raise_for_status()
        j = resp.json()
        # Contract + face-model guards: a stale v1 server would parse
        # as "zero people all day" (review C11); a different face
        # model would poison the shared gallery (review G4). Raising
        # here rides the fail-open path — loud empty, never wrong.
        if j.get("contract") != "observe-v2":
            raise RuntimeError(
                f"sam wire: wrong contract {j.get('contract')!r}",
            )
        remote_model = j.get("stats", {}).get("face_model")
        if (
            self._expected_face_model
            and remote_model
            and remote_model != self._expected_face_model
        ):
            raise RuntimeError(
                f"sam wire: pod face model {remote_model!r} != "
                f"backend {self._expected_face_model!r}",
            )
        return _parse_observe(j)


def _parse_observe(j: dict) -> RemoteObservations:
    rows_by_frame: dict[int, list[list[float]]] = {}
    face_obs: dict[tuple[int, int], list[RemoteFaceObs]] = {}
    tracks = j.get("tracks", [])
    if len(tracks) > _MAX_TRACKS:
        log.warning(
            "sam_client.tracks_truncated %d -> %d", len(tracks), _MAX_TRACKS,
        )
        tracks = tracks[:_MAX_TRACKS]
    dropped_rows = 0
    for det_ind, trk in enumerate(tracks):
        tid = int(trk["track_id"])
        for row in trk.get("rows", [])[:_MAX_ROWS_PER_TRACK]:
            orig, x1, y1, x2, y2, prob = row
            candidate = [
                float(x1), float(y1), float(x2), float(y2),
                float(tid), float(prob), 0.0, float(det_ind),
            ]
            # Finiteness gate (review S5): NaN/Inf bboxes poison carry
            # position gates and annotation int() math downstream.
            if not all(np.isfinite(v) for v in candidate):
                dropped_rows += 1
                continue
            # BoT-SORT row shape: (x1, y1, x2, y2, tid, conf, cls,
            # det_ind) — what BodyReIDPipeline.process_frame unpacks.
            rows_by_frame.setdefault(int(orig), []).append(candidate)
        for fo in trk.get("face_obs", [])[:_MAX_FACE_OBS_PER_TRACK]:
            # 8th element = SAM-face countersign (oracle_face); absent
            # on older payloads → treated as confirmed.
            orig, det, fx1, fy1, fx2, fy2, emb_b64 = fo[:7]
            oracle_face = bool(fo[7]) if len(fo) > 7 else True
            emb = np.frombuffer(
                base64.b64decode(emb_b64), dtype=np.float32,
            )
            if emb.shape != (_EMB_DIM,):
                log.warning(
                    "sam_client.bad_embedding tid=%d frame=%d shape=%s",
                    tid, int(orig), emb.shape,
                )
                continue
            bbox = (float(fx1), float(fy1), float(fx2), float(fy2))
            if not (
                np.isfinite(float(det)) and all(np.isfinite(v) for v in bbox)
            ):
                dropped_rows += 1
                continue
            face_obs.setdefault((tid, int(orig)), []).append(
                RemoteFaceObs(
                    frame_idx=int(orig),
                    embedding=emb,
                    det_score=float(det),
                    bbox=bbox,
                    oracle_face=oracle_face,
                ),
            )
    if dropped_rows:
        log.warning("sam_client.nonfinite_dropped n=%d", dropped_rows)
    screen_regions: list[tuple[float, float, float, float, float]] = []
    for reg in j.get("screen_regions", [])[:16]:
        try:
            vals = tuple(float(v) for v in reg[:5])
        except (TypeError, ValueError):
            continue
        if len(vals) == 5 and all(np.isfinite(v) for v in vals):
            screen_regions.append(vals)
    stats = j.get("stats", {})
    return RemoteObservations(
        epoch=str(j.get("epoch", "")),
        rows_by_frame={
            k: np.asarray(v, dtype=np.float32)
            for k, v in rows_by_frame.items()
        },
        face_obs=face_obs,
        frames_seen=int(j.get("frames_seen", 0)),
        det_count=int(stats.get("det_count", 0)),
        max_prob=float(stats.get("max_prob", 0.0)),
        mean_prob=float(stats.get("mean_prob", 0.0)),
        person_thr=float(stats.get("person_thr", 0.0)),
        timings_ms={
            k: float(v) for k, v in j.get("timings_ms", {}).items()
        },
        screen_regions=tuple(screen_regions),
    )
