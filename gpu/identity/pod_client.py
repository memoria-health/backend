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

import asyncio
import base64
import logging
import os
import pathlib

import httpx
import numpy as np

from identity.wire import RemoteFaceObs, RemoteObservations
# Raw frame wire (2026-08-13): pixels as-is over loopback instead of a
# JPEG encode/decode pair — a top CPU stage under multi-camera load.
# The format lives in ONE module both services import, so pack and
# parse cannot drift. Importing perception here is the allowed
# direction (identity is the higher layer).
from perception.rawwire import pack_frames

log = logging.getLogger(__name__)

_EMB_DIM = 512

# Default matches perception/server.py's `_load_auth_token`.
_TOKEN_FILE = "/persist/secrets/perception_token"


def _wh(v) -> "tuple[int, int] | None":
    """(width, height) of the frame the screen regions were measured in.

    None from an older pod build, which is why the consumer must treat
    None as "cannot normalise" rather than substituting a guess — the
    guess is what put every region in the wrong place.
    """
    try:
        w, h = int(v[0]), int(v[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def load_perception_token() -> str:
    """The shared bearer for perception, from the environment or a file.

    MUST mirror the server's resolution. Perception reads env-then-file
    precisely so auth can be switched on by writing one file on the
    shared volume — no pod recycle. This side read the env ALONE, so
    writing that file turned auth on for the server and left the only
    legitimate client sending no bearer at all: every /observe_raw came
    back 401, every tracking step failed, and the service went on
    answering /state as `ready` with an empty camera map (2026-08-23).

    Not imported from `perception.server`: importing the server module
    would start it. Eight duplicated lines beat that.
    """
    env = os.environ.get("NEMARA_PERCEPTION_TOKEN", "").strip()
    if env:
        return env
    path = os.environ.get("NEMARA_PERCEPTION_TOKEN_FILE", _TOKEN_FILE)
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


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
        # Perception grew bearer auth on 2026-08-11. Same shared-token shape
        # as /state, and same "empty disables it" semantics on the server, so
        # an unset token leaves dev exactly as it was. Sent on every request
        # via the client's default headers rather than per-call, so a new
        # endpoint cannot forget it.
        _tok = load_perception_token()
        if _tok:
            self._client.headers["Authorization"] = f"Bearer {_tok}"
        self._expected_face_model = expected_face_model

    async def aclose(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict:
        resp = await self._client.get(f"{self._base}/health")
        resp.raise_for_status()
        return resp.json()

    async def faces(
        self,
        frame: "bytes | np.ndarray",
        boxes: list[tuple[int, float, float, float, float]],
        *,
        full: bool = False,
    ) -> list[tuple[int, "np.ndarray", float, float, float, float]]:
        """Out-of-band face embeddings for KNOWN person boxes (the
        2026-07-24 decoupling — tracking batches never pay for
        identity). `full=True` also detects faces on the whole frame
        (track_id=-1, caller assigns) — lag-immune for close faces.
        Returns (track_id, embedding, face_px, det, cx, cy) with cx/cy
        the face center in the sent frame's pixel space.

        `frame` as an ndarray rides the raw wire (/faces_raw, no codec
        work); JPEG bytes keep the multipart path."""
        import json as _json

        if isinstance(frame, np.ndarray):
            # to_thread: packing is a multi-MB memcpy — off the event
            # loop, same as the JPEG encode it replaced always was.
            body = await asyncio.to_thread(
                pack_frames,
                {
                    "v": 1,
                    "boxes": [list(b) for b in boxes],
                    "full": 1 if full else 0,
                },
                [(0, frame)],
            )
            resp = await self._client.post(
                f"{self._base}/faces_raw",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        else:
            resp = await self._client.post(
                f"{self._base}/faces",
                data={
                    "boxes": _json.dumps([list(b) for b in boxes]),
                    "full": "1" if full else "0",
                },
                files=[("frame", ("f.jpg", frame, "image/jpeg"))],
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
        frames_by_index: list[tuple[int, "bytes | np.ndarray"]],
        *,
        camera_id: str,
        person_thr: float,
        face_interval: int,
        face_cap: int,
        detect_screens: bool = False,
        session_state: "dict | None" = None,
    ) -> RemoteObservations:
        """POST ordered (orig_idx, frame) frames; parse the raw
        observations. Raises on transport/HTTP errors (fail-open is
        the caller's job). `detect_screens` asks the pod for one
        screen-concept pass on the newest frame (display-veto map
        refresh, ~0.5s).

        `session_state` (stateless-worker mode): the camera's portable
        cross-tick state — pass {} on the first tick. Presence of the
        key switches the WORKER to stateless for this request, and the
        updated snapshot comes back on the response. Raw wire only.

        ndarray frames ride the raw wire (/observe_raw — no JPEG
        encode here, no decode on the pod); bytes keep multipart."""
        if frames_by_index and isinstance(frames_by_index[0][1], np.ndarray):
            header = {
                "v": 1,
                "camera_id": camera_id,
                "person_thr": person_thr,
                "face_interval": face_interval,
                "face_cap": face_cap,
                "detect_screens": 1 if detect_screens else 0,
            }
            if session_state is not None:
                header["session_state"] = session_state
            # to_thread: packing a batch is a ~tens-of-MB memcpy — off
            # the event loop, same as the JPEG encode it replaced.
            body = await asyncio.to_thread(
                pack_frames, header, frames_by_index,
            )
            resp = await self._client.post(
                f"{self._base}/observe_raw",
                content=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        else:
            files = [
                ("frames", (f"{idx:05d}.jpg", jpeg, "image/jpeg"))
                for idx, jpeg in frames_by_index
            ]
            resp = await self._client.post(
                f"{self._base}/observe",
                data={
                    "camera_id": camera_id,
                    "person_thr": str(person_thr),
                    "face_interval": str(face_interval),
                    "face_cap": str(face_cap),
                    "detect_screens": "1" if detect_screens else "0",
                },
                files=files,
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
    person_boxes: list[tuple[float, float, float, float]] = []
    for box in j.get("person_boxes", [])[:32]:
        try:
            vals = tuple(float(v) for v in box[:4])
        except (TypeError, ValueError):
            continue
        if len(vals) == 4 and all(np.isfinite(v) for v in vals):
            person_boxes.append(vals)
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
        screen_concept=str(j.get("screen_concept", "")),
        screen_frame_wh=_wh(j.get("screen_frame_wh")),
        person_boxes=tuple(person_boxes),
        session_state=(
            j["session_state"]
            if isinstance(j.get("session_state"), dict) else None
        ),
    )
