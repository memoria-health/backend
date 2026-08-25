"""Per-camera identity loop + home orchestrator (async glue).

Each occupied camera runs one loop: drain the frame ring into a
contiguous overlapped batch → pod /observe (persistent EdgeTAM session)
→ screen veto → track state → sparse identity (face+body scoring, seat,
harvest) → publish overlay + state. The pod is single-flight, so all
cameras share one client-side lock; an empty camera polls slowly.

Identity is event-driven, exactly the agreed three rules: face evidence
is requested from the pod only when the seat is empty, an unseated
track is live, or on a slow opportunistic cadence while locked-solo
(free re-anchor + keeps harvest flowing). Everything heavy runs off the
event loop so LiveKit frame intake never stalls.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Optional
from uuid import UUID

import cv2
import numpy as np

from identity.gallery import CrossClipBodyGallery
from identity.body_embeddings import extract_body_embedding
from identity.display_veto import (
    ScreenMap, apply_veto,
    load_regions, save_regions,
)
from identity.matching import topk_mean_cosine_match
from identity.wire import RemoteObservations
from identity.pod_client import SamServiceClient
from identity.batching import plan_batch
from identity.fusion import SightingHistory
from identity.harvest import harvest_eligible
from identity.scoring import body_conf, face_conf
from identity.journal import EventJournal
from identity.seat import (
    _iou,
    carry_seat,
    Candidate,
    Seat,
    heal_seat,
    notary_contradiction,
    resolve_track_conflicts,
    update_seat,
)
from identity.state import CameraState
# Re-exported: IdentityHome moved to home.py, and `from identity.loop
# import IdentityHome` is what the service and the tests already say.
from identity.home import (  # noqa: F401
    DEAD_BOX_TTL_S, SEAT_DELTA, SEAT_FLOOR, IdentityHome,
)

log = logging.getLogger(__name__)

# Stateless-worker dispatch (2026-08-16): the camera's cross-tick
# session state travels with every request (CameraLoop owns it) and any
# free worker serves any camera — shared-queue dispatch instead of the
# camera→worker pin. Requires a worker image that echoes session_state
# (gpu-v3.3+); the loop logs loudly if it doesn't. Off = exactly the
# pinned behavior.
_STATELESS_DISPATCH = os.environ.get(
    "NEMARA_STATELESS_DISPATCH", "",
).strip() in ("1", "true")

# ── Cadence + identity constants (pilot-calibration targets) ─────────
ACTIVE_INTERVAL_S = 0.6        # batch cadence while the room is occupied
NONPATIENT_INTERVAL_S = 4.0    # tiny scheduler v1: a camera with people
                               # but NOT the patient seat runs light —
                               # guests don't need dense tracking; the
                               # patient's camera keeps the density.
PROBE_SCAN_GAP_S = 0.35        # while SEARCHING, the probe scans EVERY
                               # ring frame since the last probe, spaced
                               # this far apart — coverage, not
                               # sampling: a 0.5s camera crossing holds
                               # 1-2 scanned frames BY CONSTRUCTION
                               # (Aryaman's requirement, 2026-07-24; a
                               # cadence can always miss the window)
PROBE_SCAN_MAX = 4             # extra frames per probe beyond the newest
FACE_PROBE_SEARCH_S = 1.5      # out-of-band identity probe cadence while
                               # SEARCHING — decoupled from tracking
                               # (2026-07-24 /faces endpoint), so being
                               # aggressive costs the tracker NOTHING
FACE_PROBE_LOCKED_S = 20.0     # probe cadence once the seat is held
                               # (opportunistic re-anchor + harvest)
EMPTY_POLL_FRAMES = 4          # a motion-triggered empty-room poll ships
                               # a handful of FRESH frames — an existence
                               # check needs eyes, not density (a dense
                               # 40-frame poll cost ~4s GPU per 20s per
                               # empty room; measured 2026-07-24)
EMPTY_MIN_GAP_S = 2.0          # motion may cut the empty-room nap short
                               # (see run()), but never below this gap
                               # between two polls — the motion flag
                               # re-arms every 0.5s while anything moves,
                               # and non-person motion never ends
                               # empty-room mode, so without a floor a
                               # TV polls the GPU ~5x the design rate.
EMPTY_INTERVAL_S = 20.0        # poll cadence for an empty room. Each
                               # empty-room call is a ~3.5s SAM3 ground;
                               # at 5s two empty rooms saturated the pod
                               # (smoke 2026-07-24). 20s = worst-case
                               # pickup for a person entering an empty
                               # room, then active cadence takes over.
# NOTE: the batch-span reseed guard was DELETED in the 2026-07-26
# productionization. It fired on ordinary pod latency (39 times in
# minutes at 3.0s; disabled for the entire proven soak), and every
# firing dropped the overlap seed EdgeTAM's carry needs — THE
# 2026-07-24 churn. Contiguity beats latency; genuinely dead intake is
# STALE_INTAKE_S's job.
# STABILITY IS THE PRODUCT; LATENCY IS SACRIFICIAL (Aryaman,
# 2026-07-24): identity is ASYNC — detection ticks ride the published
# Patient State, so tracking lag costs nothing downstream.
# Env-overridable (2026-08-14) for the capacity experiments — the
# 15-cams-per-node program A/Bs 2.0 against the proven 3.0. Default
# unchanged; a lower value trades state-freshness for capacity and
# needs an accuracy soak before it becomes the fleet default.
TARGET_FPS = float(os.environ.get("NEMARA_TARGET_FPS", "3.0"))
_TARGET_FPS_PROVEN = 3.0       # THE GPU-duty lever (measured
                               # 2026-07-26: pod cost ≈ 200-250ms x
                               # FRAMES, near-independent of
                               # resolution). 3fps → 10-19 frame
                               # batches, ~5s cadence, 3-5s state
                               # freshness, 60-80% duty per occupied
                               # room; the overnight soak held identity
                               # with zero wrong locks at this stride
MAX_BATCH = 40                 # ceiling only — batches self-pace to
                               # 10-19 frames at 3fps; under multi-room
                               # load stride stretches instead of lag
                               # compounding (graceful degradation)
JPEG_QUALITY = 80
# Frame wire to perception (2026-08-13). "raw" ships BGR pixels as-is —
# identity and perception share the box, and the JPEG encode/decode pair
# on that loopback hop was a top CPU stage under multi-camera load
# (encode p50 391ms at 3 cams, plus the pod's decode out of the same
# starved cores). "jpeg" is the one-env revert: NEMARA_FRAME_WIRE=jpeg.
FRAME_WIRE = os.environ.get("NEMARA_FRAME_WIRE", "raw").strip().lower()
# Face harvest (2026-08-24). Learned only from a face-confirmed lock —
# see _harvest_face. Off by a single flag if it ever misbehaves.
FACE_HARVEST_ENABLED = os.environ.get(
    "NEMARA_FACE_HARVEST", "1").strip() not in ("0", "false", "no")
FACE_HARVEST_MIN_DET = float(os.environ.get("NEMARA_FACE_HARVEST_DET", "0.7"))
FACE_HARVEST_MIN_PX = float(os.environ.get("NEMARA_FACE_HARVEST_PX", "60"))
# Too close to a row already held = a view we have. Keeps the gallery N
# distinct looks rather than N copies of one lock.
FACE_HARVEST_DEDUP = float(os.environ.get("NEMARA_FACE_HARVEST_DEDUP", "0.75"))
# Learned rows can never swamp the 25 enrolled selfies.
FACE_HARVEST_MAX_ROWS = int(os.environ.get("NEMARA_FACE_HARVEST_MAX", "60"))
FACE_HARVEST_MIN_GAP_S = float(os.environ.get("NEMARA_FACE_HARVEST_GAP_S", "60"))

SCREEN_REFRESH_S = float(os.environ.get(
    "NEMARA_SCREEN_REFRESH_S", "120"))   # screens are furniture
HARVEST_MIN_AGE_S = 3.0        # never learn from newborn (messy) tracks
INSTANT_FACE_CONF = 0.65       # single-probe face evidence this strong
                               # locks on sight (close pass — the
                               # walk-past requirement). Briefly raised
                               # to 0.80 after the 19:13 aunt lock, then
                               # restored (Aryaman): that lock's 0.68
                               # frame was the PATIENT's face inside the
                               # aunt's overlapping box — an attribution
                               # failure, now killed at the source by
                               # the contested-position rule below, not
                               # by pricing real evidence out.
FUSION_MIN_HITS = 3            # sightings before the fused template
                               # speaks for the track
FACES_ENABLED = True           # False = pure tracking (the "yesterday"
                               # baseline preset — zero identity work)
STALE_INTAKE_S = 10.0          # no fresh frames this long → clear tracks
# Env-overridable so the native-vs-downscaled question can be A/B'd
# without a code change: NEMARA_TRACK_SEND_W=960 to test, unset to
# restore native. The measured claim being tested (2026-07-29) is that
# EdgeTAM "encodes at its own fixed internal res" so resolution can't
# reach it — which sits right next to a 36ms/frame figure that turned
# out to be 3-4x off, so it is worth re-checking rather than trusting.
TRACK_SEND_W = int(os.environ.get("NEMARA_TRACK_SEND_W", str(10 ** 9)))
_TRACK_SEND_W_DEFAULT = 10 ** 9  # NATIVE ship resolution (Aryaman,
                               # 2026-07-26). Measured: 640px shipping
                               # cut wire/encode ~9x but GPU only ~10%
                               # (EdgeTAM encodes at its own fixed
                               # internal res) — not worth the small-
                               # distant-person detection risk. Frame
                               # COUNT (TARGET_FPS) is the cost lever.


def series_bboxes(obs: "RemoteObservations", batch) -> dict[int, list]:
    """tid -> [(frame_wall_ts, bbox), ...] for every observed frame —
    feeds the follow-crop series (2026-08-18)."""
    # rows_by_frame keys are the pod-echoed ORIGINAL ring indexes
    # (monotonic), not batch positions — map idx -> ts via the batch.
    ts_by_idx = {e[0]: e[1] for e in batch}
    out: dict[int, list] = {}
    for fi, rows in obs.rows_by_frame.items():
        ts = ts_by_idx.get(fi)
        if ts is None:
            continue
        for row in rows:
            out.setdefault(int(row[4]), []).append(
                (float(ts), (float(row[0]), float(row[1]),
                             float(row[2]), float(row[3]))))
    for v in out.values():
        v.sort(key=lambda e: e[0])
    return out


def newest_bboxes(obs: RemoteObservations) -> dict[int, tuple]:
    """tid -> bbox from each track's newest observed frame."""
    out: dict[int, tuple[int, tuple]] = {}
    for fi, rows in obs.rows_by_frame.items():
        for row in rows:
            tid = int(row[4])
            prev = out.get(tid)
            if prev is None or fi >= prev[0]:
                out[tid] = (fi, (float(row[0]), float(row[1]),
                                 float(row[2]), float(row[3])))
    return {tid: bbox for tid, (_, bbox) in out.items()}


def best_face_per_track(obs: RemoteObservations) -> dict[int, tuple]:
    """tid -> (embedding, face_px) — the largest countersigned face."""
    best: dict[int, tuple] = {}
    for (tid, _fi), hits in obs.face_obs.items():
        for h in hits:
            if not h.oracle_face:
                continue
            px = float(h.bbox[3] - h.bbox[1])
            if tid not in best or px > best[tid][1]:
                best[tid] = (h.embedding, px)
    return best


def _contested(cx: float, cy: float, boxes: list, *, own: int) -> bool:
    """Is this face position ALSO inside another track's (padded) box?

    The adjacency defense (2026-07-24, the 19:13 aunt lock): the
    patient standing in front of the aunt put his face literally
    inside her crop, and crop hits inherit the crop's track id. A face
    position claimed by more than one box is NOBODY's evidence — while
    two people overlap on screen the camera cannot tell whose face is
    whose, so neither accumulates. Evidence resumes on separation;
    the seat holds through the gap on silence, harvest is solo-gated
    anyway. Ambiguity always loses to silence.
    """
    for tid, x1, y1, x2, y2 in boxes:
        if tid == own:
            continue
        px, py = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        if x1 - px <= cx <= x2 + px and y1 - py <= cy <= y2 + py:
            return True
    return False


def _assign_face(
    cx: float, cy: float, boxes: list,
) -> Optional[int]:
    """Assign a full-frame face (center cx,cy, ring space) to a track.

    The face must fall inside exactly ONE box (padded 20%) or it is
    discarded — NO single-track shortcut (live 2026-07-24, T20: the
    patient re-entered during the ~50s before his track existed, the
    room's only track was the aunt's, and the shortcut credited HIS
    walk-in face to HER track probe after probe until the consistency
    logic — correctly — seated her). An untracked person's face locks
    nothing until their track exists: identity never outruns tracking
    (the seat is a track property). A wrong-person assignment is the
    failure mode identity exists to prevent; ambiguity and orphanhood
    always lose to silence.
    """
    owners = []
    for tid, x1, y1, x2, y2 in boxes:
        px, py = (x2 - x1) * 0.2, (y2 - y1) * 0.2
        if x1 - px <= cx <= x2 + px and y1 - py <= cy <= y2 + py:
            owners.append(tid)
    return owners[0] if len(owners) == 1 else None


class CameraLoop:
    """One camera's continuous loop."""

    # How far a track must have travelled from where it was born to
    # count as something that walked in rather than something painted
    # on the wall. A fraction of the frame diagonal, so it means the
    # same thing at every resolution and in every room.
    _MOVED_FRAC = 0.05

    def _moved_since_birth(self) -> set:
        """Track ids that have travelled since they first appeared.

        These are never vetoed, however contained they are. It is the
        one guard that generalises: the containment floor is safe only
        while a room's art hangs higher than a person's box reaches,
        which is a fact about one room and not about the world.
        """
        out = set()
        for tid, tv in self.state.tracks.items():
            b0, b1 = getattr(tv, "birth_bbox", None), tv.bbox
            if not b0 or len(b0) < 4:
                continue
            dx = ((b1[0] + b1[2]) - (b0[0] + b0[2])) / 2.0
            dy = ((b1[1] + b1[3]) - (b0[1] + b0[3])) / 2.0
            wh = self._frame_wh() or (1.0, 1.0)
            diag = (wh[0] ** 2 + wh[1] ** 2) ** 0.5 or 1.0
            if ((dx * dx + dy * dy) ** 0.5) / diag > self._MOVED_FRAC:
                out.add(tid)
        return out

    def _log_batch(self, batch, send_frames, obs, encode_ms: float) -> None:
        """One line per batch, with the pod round-trip split into the
        pod's OWN reported compute and the wire. The single `pod=` number
        conflated them, so a slow GPU and a saturated link looked the
        same — an afternoon of guessing on 2026-07-29.
        """
        span = batch[-1][1] - batch[0][1]
        # getattr, not obs.timings_ms: an older pod build or a test double
        # has no such field, and losing a log line must never take the
        # tracking loop down with it.
        pt = getattr(obs, "timings_ms", None) or {}
        pod_total = pt.get("total", 0.0)
        log.info(
            "identity.batch cam=%s frames=%d span=%.1fs stride=%.0fms | "
            "encode=%.0f queue=%.0f pod=%.0f (compute=%.0f sam=%.0f "
            "decode=%.0f wire=%.0f) publish=%.0f (ms)",
            self.camera_id, len(send_frames), span,
            span * 1000.0 / max(1, len(send_frames) - 1),
            encode_ms, self.queue_ms, self.pod_ms,
            pod_total, pt.get("sam", 0.0), pt.get("decode", 0.0),
            self.pod_ms - pod_total,
            self.stage_ms["publish"],
        )
        # Same split on /state so the dashboard can show it live.
        self.stage_ms["pod_compute"] = round(pod_total, 1)
        self.stage_ms["wire"] = round(self.pod_ms - pod_total, 1)

    def _remember_dead(self, events, prev_meta, vetoed, now: float) -> None:
        """Keep each dead track's last box for the seat carry, briefly.

        `state.apply` has already forgotten them by the time `reap_seat`
        runs later this tick, which is why `carry_seat` could never find
        one. A VETOED track is excluded on purpose: we judged it a
        painting, and a seat carried onto its corpse hands somebody's
        identity to the wall.
        """
        for tid in events.deaths:
            if tid in (vetoed or ()):
                continue
            meta = prev_meta.get(tid)
            if meta:
                self._dead_boxes[tid] = (meta[0], now)
        # Bounded by time: an older box is not evidence about now.
        if self._dead_boxes:
            self._dead_boxes = {
                t: v for t, v in self._dead_boxes.items()
                if now - v[1] <= DEAD_BOX_TTL_S
            }

    def _absorb_screen_regions(self, obs) -> None:
        """Fold one screen-concept pass into the per-camera region map."""
        # The size the POD measured them in is the only size that can be
        # right: `_frame_wh()` is the intake reader's last-seen size, a
        # different frame whenever the SFU changed rung between the two.
        gwh = getattr(obs, "screen_frame_wh", None)
        if gwh is None:
            log.warning(
                "identity.screen_regions_unscaled cam=%s — pod did not "
                "report the grounding frame size; dropping this pass "
                "rather than placing regions by guesswork",
                self.camera_id,
            )
        else:
            self.screen_map.add(
                [r[:4] for r in obs.screen_regions],
                frame_wh=gwh, now=time.monotonic(),
            )
        save_regions(self._screen_path, self.screen_map.live(time.monotonic()))
        # HOW MANY the pod found. `screen_pass ok` only says the call did
        # not throw, so "running" and "finding anything" looked alike.
        log.warning(
            "identity.screen_regions cam=%s concept=%s found=%d live=%d",
            self.camera_id,
            (getattr(obs, "screen_concept", "") or "?"),
            len(obs.screen_regions),
            len(self.screen_map.live(time.monotonic())),
        )

    def _seat_protected(self) -> set:
        """Track ids identity has bound to a resident on this camera.

        These are never vetoed. A painted face is not face-matched to
        somebody's father, so this shield costs nothing on the confounds
        it exists beside — and unlike the containment floor and the
        movement test it is not geometric, so it still holds when the
        region is wrong or the tracker has renumbered.
        """
        try:
            return {s.track_id for _pid, s in self.home.seats_on(self.camera_id)}
        except Exception:  # noqa: BLE001 — a shield must never kill a tick
            log.exception("identity.seat_protected_failed cam=%s", self.camera_id)
            return set()

    def _report_deaths(self, events, prev_meta, vetoed, rows_before,
                       rows_after, now: float) -> None:
        """Say WHY each track died, once per death.

        Every explanation for the churn so far has been inference from
        second-order symptoms — seat lifetimes, event counts, commit
        timings — because nothing recorded the death itself. Three
        separate theories were argued and two were wrong. This line is
        meant to end that in one occupied hour.

        Fields, and what each one rules in or out:

          age_s        how long the track lived. Seconds means the
                       detector is unstable; minutes means something
                       ended a healthy track.
          gap_s        how long since it was last seen. Sub-second is a
                       detector blink; multi-second is a person who
                       actually left or became invisible.
          vetoed       was it removed by the display filter this batch.
                       Distinguishes "we threw it away" from "the
                       detector lost it" — the exact question the veto
                       theory could not answer.
          rows         detector rows before -> after the veto. `0 -> 0`
                       means the detector returned nothing at all;
                       `4 -> 0` means we discarded everything it found.
          reborn_iou   best overlap with a track BORN in this same
                       batch. High means a renumbering (the seat carry
                       case); 0.00 means nothing took its place.
          seat         did it hold an identity. A death with a seat is
                       the one that costs a face match.

        Never raises: instrumentation must not be able to end a tick.
        """
        try:
            births = {
                tid: self.state.tracks[tid].bbox
                for tid in events.births if tid in self.state.tracks
            }
            for tid in events.deaths:
                meta = prev_meta.get(tid)
                if meta is None:
                    continue
                box, last_seen, born_at, seen = meta
                best_iou = max(
                    (_iou(box, b) for b in births.values()), default=0.0)
                held = any(
                    st.track_id == tid for st in self.home.seats.values()
                ) if getattr(self.home, "seats", None) else False
                log.info(
                    "identity.track_death cam=%s tid=%s age_s=%.1f gap_s=%.2f "
                    "batches=%d vetoed=%s rows=%d->%d reborn_iou=%.2f seat=%s",
                    self.camera_id, tid, now - born_at, now - last_seen,
                    seen, tid in (vetoed or ()), rows_before, rows_after,
                    best_iou, held,
                )
        except Exception:  # noqa: BLE001 — a log line is never a dependency
            log.warning("identity.track_death_report_failed", exc_info=True)

    def _seat_boxes(self) -> dict:
        """(camera_id, track_id) -> current bbox, for the seat carry.

        Includes tracks that have just gone missing but are still held
        by the death hysteresis, which is exactly the window in which a
        renumbering happens.
        """
        return {
            (self.camera_id, tid): tv.bbox
            for tid, tv in self.state.tracks.items()
        }

    def _frame_wh(self) -> tuple[int, int] | None:
        """(width, height) of the frames currently arriving, or None.

        `_last_shape` is (h, w) and is stamped by the intake reader in
        service.py on every size CHANGE, which is the only place the
        real size is known — the observation bundle does not carry it.
        None before the first frame, which makes the veto treat boxes as
        already normalised and therefore veto nothing. That is the safe
        direction: no veto costs a phantom track, a wrong veto costs a
        person.
        """
        shape = getattr(self, "_last_shape", None)
        if not shape or len(shape) < 2:
            return None
        h, w = int(shape[0]), int(shape[1])
        return (w, h) if w > 0 and h > 0 else None

    def __init__(self, *, camera_id: str, home: IdentityHome) -> None:
        self.camera_id = camera_id
        self.home = home
        self.state = CameraState(camera_id=camera_id)
        # Tick health, surfaced on /state. "No tracks" is ambiguous —
        # an empty room and a perception service failing every call look
        # identical — so the loop counts which one it is.
        self.step_ok = 0
        self.step_fail = 0
        self.last_step_ok_at = 0.0
        self.last_step_error = ""
        # Where a track was when it died. `carry_seat` matches a
        # renumbered track against it; sourcing those boxes from
        # `state.tracks` (live only) meant it could never succeed.
        self._dead_boxes: dict[int, tuple[tuple, float]] = {}
        # Track ids this service last judged to be a display or a
        # painting, sent back to the pod so it stops counting them as
        # people. Never used to remove anything — see the note at the
        # trigger in continuous._observe_locked.
        self._not_people: set[int] = set()
        # Furniture survives a restart, so the regions do too. Without
        # this, every restart begins blind and re-tracks every painted
        # face until the next search two minutes later — and the tracks
        # born in that window live on. Restarts are not all deliberate:
        # the pod watchdog exits on its own several times a day.
        self._screen_path = os.path.join(
            os.environ.get("NEMARA_STATE_DIR", "/persist/state"),
            f"screen_regions_{camera_id}.json",
        )
        self.screen_map = ScreenMap()
        for _box in load_regions(self._screen_path):
            # No frame size: what is on disk is already normalised.
            self.screen_map.add([_box])
        self.ring: list[tuple[int, float, np.ndarray]] = []  # fed by intake
        # Frame idx is OWNED HERE, not by the intake reader — a LiveKit
        # re-subscribe spawns a fresh reader, and a reader-local counter
        # restarting at 0 would make every new frame sort "older" than
        # last_shipped and the batcher would starve forever.
        self._next_idx = 0
        self._last_shipped: Optional[tuple[int, float, np.ndarray]] = None
        self._overflow_warned_at = 0.0
        self._screens_at = 0.0
        self._probed_at = 0.0
        self._face_harvested_at = 0.0
        # Face harvests decided inside the threaded fold,
        # written by the async loop that called it.
        self._pending_face: list = []
        self._scanned_to = 0.0   # coverage-scan high-water mark (ring ts)
        self._pod_epoch: Optional[str] = None  # session-id-reuse guard
        # Stateless-worker mode: THIS loop owns the camera's cross-tick
        # perception state; {} = "stateless, no history yet" (the key's
        # presence in the request is what switches the worker).
        self._session_state: dict = {}
        self._probe_task: Optional[asyncio.Task] = None
        # Told on every step whether this camera is worth full-rate
        # video. Set by service.py, which owns the LiveKit publication;
        # the loop only knows WHEN the room is empty, not what to do
        # about it. None = nobody listening (unit harnesses).
        self.on_idle: Optional[Callable[[bool], None]] = None
        self._idle: Optional[bool] = None
        self.stage_ms: dict = {}     # per-batch stage timings (measured)
        self.probe_ms: dict = {}     # per-probe stage timings (measured)
        self._body_cos: dict[int, float] = {}   # latest probe body scores
        # Every probe's full scores (faces/body/fused cands), served at
        # /probes for the soak dashboard's tick-correlated analysis
        # view (Aryaman 2026-07-28). ~4000 ≈ 1.7h searching / days
        # locked; memory is a few MB of floats.
        self.probe_history: deque = deque(maxlen=4000)
        # Per-track sighting histories (2026-07-24, twice-revised): the
        # seat judges robust statistics over a track's recent face
        # sightings — top-K so back-of-head garbage can't dilute a real
        # patient (T11 walk-around), padded so a lone outlier can't
        # seat an impostor (the aunt). Dies with the track.
        # (track_id, patient_id) → that track's face evidence FOR THAT
        # RESIDENT. A track on a shared camera accumulates one history
        # per candidate resident, because "twelve sightings averaging
        # 0.46" is a claim about a specific person and pooling them
        # across residents would let Alice's strong history speak for a
        # Bob comparison.
        self._sightings: dict[tuple[int, UUID], SightingHistory] = {}
        # Set by intake: (ts, native-resolution BGR) — the probe scores
        # faces at NATIVE pixels (ring frames are downscaled for memory).
        self.latest_native: Optional[tuple[float, np.ndarray]] = None
        self.latest_jpeg: Optional[bytes] = None
        # When the last empty-room poll was ATTEMPTED (stamped whether
        # it succeeded or failed). run()'s nap floors itself off this,
        # so a failing pod or a re-arming motion flag can't turn the
        # early-wake into a retry storm.
        self._empty_poll_at = 0.0
        self.hud = "waiting…"
        self.pod_ms = 0.0
        self.queue_ms = 0.0
        self.lag_s = 0.0
        # Empty-room economy (2026-07-24, Aryaman): a still empty room
        # costs ZERO GPU. Intake LATCHES this on any real frame change
        # (latched, not sampled — someone entering then sitting still
        # between polls must never be missed); the empty-room poll
        # consumes it. Starts True so the first poll always runs.
        self.motion_pending = True

    # ── intake side (called from the LiveKit consumer) ───────────────
    # Ring depth: ~20s at INTAKE_FPS=15 (the production intake). The
    # 45-frame ring overflowed roughly once per session at 15fps, and
    # every overflow silently broke the carry chain — the churn class
    # of 2026-07-24. ~300 native 720p BGR frames ≈ 800MB/camera peak;
    # bounded, and the STREAM_CAPACITY=2 SDK queue caps what feeds it.
    RING_MAX = 300

    def push_frame(self, ts: float, bgr: np.ndarray) -> None:
        # Intake does NOT publish to the live view. `_publish` is the
        # single writer, so the view is always a frame the tracker
        # actually analysed, with boxes where it saw the person. A
        # second writer here pushing fresher unboxed (or differently
        # boxed) frames is what caused first the rewind and then the
        # trailing/blank boxes — 2026-07-29, twice. Empty rooms still
        # refresh: the empty-room poll publishes the newest frame every
        # EMPTY_INTERVAL_S.
        self._next_idx += 1
        self.ring.append((self._next_idx, ts, bgr))
        if len(self.ring) > self.RING_MAX:
            drop = len(self.ring) - self.RING_MAX
            shipped_idx = self._last_shipped[0] if self._last_shipped else 0
            unshipped = sum(1 for f in self.ring[:drop] if f[0] > shipped_idx)
            if unshipped:
                # Observability only — dropping old UNSHIPPED candidates
                # does NOT break the batch chain: plan_batch simply picks
                # from the remaining (newer) frames, all still after
                # last_shipped, exactly the viewer's normal condition.
                # NEVER reset last_shipped here: the old "re-seed after
                # the gap" reset made every in-flight push count as
                # unshipped (shipped_idx=0) → _overflowed re-latched
                # every cycle → permanent whole-ring batches → the pod
                # session rewound 30s per tick and renumbered on every
                # reground under motion (live 2026-07-24, the churn).
                if self.state.tracks and ts - self._overflow_warned_at >= 5.0:
                    log.warning(
                        "identity.ring_overflow cam=%s dropped_unshipped=%d "
                        "(tracks=%d)",
                        self.camera_id, unshipped, len(self.state.tracks),
                    )
                    self._overflow_warned_at = ts
            del self.ring[:drop]

    async def run(self) -> None:
        if self._probe_task is None:
            # Out-of-band identity: fully decoupled from the tracking
            # cadence (2026-07-24 — face batches inside /observe were
            # strangling the tracker; see probe_loop).
            self._probe_task = asyncio.ensure_future(self.probe_loop())
        while True:
            # Tiny scheduler v1: the patient's camera (or any camera
            # while searching) runs dense; a guest-occupied camera runs
            # light; an empty room polls slowly (every empty-room call
            # is a SAM3 ground on the shared GPU).
            # "Someone we know is elsewhere" is what makes THIS camera
            # cheap — unchanged for one resident, and for several it
            # takes any seated resident as the signal rather than
            # requiring all of them to be located.
            if not self.state.tracks and self._last_shipped is not None:
                interval = EMPTY_INTERVAL_S
            elif self.home.seats and self.home.seat_on(self.camera_id) is None:
                interval = NONPATIENT_INTERVAL_S
            else:
                interval = ACTIVE_INTERVAL_S
            if interval == EMPTY_INTERVAL_S:
                # Empty-room nap that motion can cut short (so a
                # pass-by isn't missed for up to 20s), with two rules
                # that the first version got wrong and that must not be
                # removed (both found in review 2026-07-29):
                #
                # 1. ALWAYS wait at least once per pass. The first
                #    version skipped the wait entirely when motion was
                #    already flagged. A pass that never waits never
                #    hands control back to the rest of the program, and
                #    when intake has stalled (_step returns straight
                #    away) this loop spins millions of times a second
                #    with everything else frozen: the video reader, the
                #    pod watchdog, the crash-and-restart paths, the
                #    resubscribe walk. The web server runs on its own
                #    thread, so /state keeps answering with frozen data
                #    and the service looks healthy while being dead.
                #
                # 2. Never poll the GPU faster than EMPTY_MIN_GAP_S.
                #    Motion re-flags within half a second of each
                #    check for as long as anything is moving, and
                #    moving things that aren't people (a TV, a fan, a
                #    pet) never produce a track — so the room stays
                #    "empty" while the flag keeps re-arming. Without a
                #    floor that polls the shared GPU ~5x more than the
                #    design intends, on a room with nobody in it.
                deadline = self._empty_poll_at + EMPTY_INTERVAL_S
                while True:
                    await asyncio.sleep(0.25)
                    now_w = time.time()
                    if now_w >= deadline:
                        break
                    if (self.motion_pending
                            and now_w - self._empty_poll_at >= EMPTY_MIN_GAP_S):
                        break
            else:
                await asyncio.sleep(interval)
            try:
                await self._step()
                self.step_ok += 1
                self.last_step_ok_at = time.time()
            except Exception as exc:         # noqa: BLE001 — keep looping
                # Counted, not just logged. Swallowing is right — one bad
                # tick must not stop the loop — but an empty room and a
                # pod throwing on every call looked identical from
                # outside, and the dashboard stayed green through both.
                self.step_fail += 1
                self.last_step_error = f"{type(exc).__name__}: {exc}"[:200]
                log.exception("identity.step_failed cam=%s", self.camera_id)

    async def _step(self) -> None:
        now = time.time()
        # Dead-reader watchdog: if intake stopped (LiveKit reader died,
        # camera offline), clear the tracks so a frozen track can never
        # hold the patient seat forever (the ghost-seat hole).
        if self.ring and now - self.ring[-1][1] > STALE_INTAKE_S:
            if self.state.tracks:
                log.warning(
                    "identity.stale_intake cam=%s age=%.1fs — clearing tracks",
                    self.camera_id, now - self.ring[-1][1],
                )
                self.state.apply({}, now=now)
            return
        # Empty of PEOPLE, not of tracks. Judging on `state.tracks`
        # alone is why a room whose wall art is tracked never stepped
        # down, and was billed at full resolution with nobody in it.
        _people = set(self.state.tracks) - self._not_people
        empty_poll = not _people and self._last_shipped is not None
        # Nothing tracked and nothing moved is exactly why the pod is
        # skipped below — and the same reason the SFU should not be
        # sending us full-resolution video for it. Transitions only.
        idle = empty_poll and not self.motion_pending
        if idle is not self._idle:
            self._idle = idle
            if self.on_idle is not None:
                try:
                    self.on_idle(idle)
                except Exception:      # noqa: BLE001 — economy, not a dependency
                    log.warning(
                        "identity.idle_hint_failed cam=%s", self.camera_id,
                        exc_info=True,
                    )
        if empty_poll:
            # Stamp the ATTEMPT before anything can fail — run()'s nap
            # floors off this, and a pod error that returns in
            # milliseconds would otherwise let the loop retry as fast
            # as the failure comes back.
            self._empty_poll_at = now
            # Empty-room economy: still room → skip the pod entirely
            # (zero GPU); motion latched → a small fresh-frame existence
            # check. Chain continuity is irrelevant with nothing tracked,
            # so the batch is just the newest frames, no overlap seed.
            if not self.motion_pending:
                if self.ring:
                    await asyncio.to_thread(self._publish, self.ring[-1])
                return
            batch = list(self.ring)[-EMPTY_POLL_FRAMES:]
        else:
            batch = plan_batch(
                self.ring, last_shipped=self._last_shipped,
                target_fps=TARGET_FPS, max_batch=MAX_BATCH,
            )
        if not batch:
            return
        # (Span guard deleted 2026-07-26 — see the constants block: it
        # broke the carry chain on ordinary latency and ran disabled
        # through the whole proven soak. STALE_INTAKE_S owns stalls.)
        want_screens = now - self._screens_at >= SCREEN_REFRESH_S

        # DECOUPLED (2026-07-24): tracking batches are ALWAYS identity-
        # free and downscaled — face/body/harvest live in probe_loop via
        # the pod's /faces endpoint. The tracking cadence never pays for
        # identity again (face batches inside /observe were 3-5x the
        # cost and strangled the carry).
        t_e = time.time()
        send_frames, inv_scale = await asyncio.to_thread(
            self._encode, batch, TRACK_SEND_W,
        )
        encode_ms = (time.time() - t_e) * 1000.0
        t_q = time.time()
        if _STATELESS_DISPATCH:
            # Stateless mode: this loop owns the camera's cross-tick
            # state and any free worker can serve the tick.
            _pod = self.home.pod_for_dispatch(self.camera_id)
        else:
            _pod = self.home.sam_for(self.camera_id)
        async with self.home.lock_for(_pod):  # per-WORKER admission
            t0 = time.time()                 # queue wait ends here
            obs = await _pod.observe(
                send_frames,
                camera_id=f"identity_{self.camera_id}",
                person_thr=0.4,
                face_interval=999,
                face_cap=0,
                detect_screens=want_screens,
                session_state=(
                    self._session_state if _STATELESS_DISPATCH else None
                ),
                # LAST tick's verdict — this tick's is not known until
                # the pod has answered. One tick stale is harmless: it
                # only suppresses a COUNT, and furniture does not move.
                not_people=self._not_people,
            )
            self.home.last_pod_ok = time.time()  # pod-progress watchdog feed
        if _STATELESS_DISPATCH:
            if obs.session_state is not None:
                self._session_state = obs.session_state
            else:
                # A worker that doesn't echo state is running a pre-
                # stateless image — continuing would re-ground every
                # tick and renumber ids silently. Loud, once per tick,
                # until the fleet image matches the flag.
                log.error(
                    "identity.stateless_not_echoed cam=%s — worker image "
                    "predates stateless mode; state kept as-is",
                    self.camera_id,
                )
        self.pod_ms = (time.time() - t0) * 1000.0
        self.queue_ms = (t0 - t_q) * 1000.0  # other cameras' turn ahead of us
        # Epoch guard (2026-07-25 "born again as T13"): pod restarts
        # reset the session's id counter, so a recycled number could
        # silently inherit a track's history — and the SEAT. On epoch
        # change: hard-reset this camera's world before applying rows.
        if obs.epoch and obs.epoch != self._pod_epoch:
            if self._pod_epoch is not None:
                log.warning(
                    "identity.pod_epoch_changed cam=%s %s->%s — resetting "
                    "tracks (ids recycled)", self.camera_id,
                    self._pod_epoch, obs.epoch,
                )
                self.state.apply({}, now=now)
                self._sightings = {}
                self.home.reap_seat(now)
            self._pod_epoch = obs.epoch
        if empty_poll:
            # Consume the latch only on a SUCCESSFUL poll — a pod error
            # leaves it set so the next poll retries the check.
            self.motion_pending = False
        if inv_scale != 1.0:
            # Bring EVERYTHING back to ring-frame space so state/veto/
            # overlay/probe all live on ONE coordinate system.
            for rows in obs.rows_by_frame.values():
                rows[:, :4] *= inv_scale
            if obs.screen_regions:
                obs.screen_regions = tuple(
                    (x1 * inv_scale, y1 * inv_scale,
                     x2 * inv_scale, y2 * inv_scale, p)
                    for x1, y1, x2, y2, p in obs.screen_regions
                )
        self._last_shipped = batch[-1]
        if want_screens:
            self._screens_at = now
            self._absorb_screen_regions(obs)
        # How much the DETECTOR returned, before anything of ours
        # removed rows. `rows_after` below is what survived the veto, so
        # the two together separate "the detector saw nothing" from "we
        # threw it away" — the distinction every theory about track
        # churn so far has had to guess at.
        rows_before = sum(len(r) for r in obs.rows_by_frame.values())
        # Ids the pod is reporting, read BEFORE the veto strips them —
        # afterwards the surviving rows can never contain a vetoed id,
        # so retirement below would empty the set every tick.
        pod_ids = {
            int(r[4]) for rows in obs.rows_by_frame.values() for r in rows
        }

        obs, vetoed = apply_veto(
            obs, self.screen_map,
            frame_wh=self._frame_wh(),
            moved=self._moved_since_birth(),
            protected=self._seat_protected(),
            now=time.monotonic(),
        )
        rows_after = sum(len(r) for r in obs.rows_by_frame.values())
        # Accumulate, don't replace: a painted face is spared on the odd
        # tick (mask wobble), and replacing would drop it exactly then.
        # Retired as soon as the pod stops reporting the id.
        self._not_people = (self._not_people | set(vetoed or ())) & pod_ids

        prev_boxes = {t: tv.bbox for t, tv in self.state.tracks.items()}
        # Snapshot what we are about to lose. `state.apply` mutates in
        # place, so anything read after it describes the survivor.
        prev_meta = {
            t: (tv.bbox, tv.last_seen, tv.born_at, tv.batches_seen)
            for t, tv in self.state.tracks.items()
        }
        events = self.state.apply(
            newest_bboxes(obs), now=now,
            batch_series=series_bboxes(obs, batch),
            rejected=set(vetoed or ()))
        self._remember_dead(events, prev_meta, vetoed, now)
        if events.deaths:
            self._report_deaths(
                events, prev_meta, vetoed, rows_before, rows_after, now)
        if events.births or events.deaths:
            log.info(
                "identity.tracks cam=%s births=%s deaths=%s live=%s",
                self.camera_id, events.births, events.deaths,
                sorted(self.state.tracks),
            )
        if (events.births or events.deaths) and self.home.journal is not None:
            # Unsupervised-soak journal: every birth/death, with the
            # picture — and the box of the track in question drawn on
            # it (image EVIDENCE, not a scene photo).
            from identity.journal import annotate

            snap = batch[-1][2]
            live_ctx = [
                (t, tv.bbox) for t, tv in self.state.tracks.items()
            ]
            for tid in events.births:
                tv = self.state.tracks.get(tid)
                self.home.journal.log(
                    "track_birth", cam=self.camera_id,
                    cam_name=self.home.name_of(self.camera_id), tid=tid,
                    frame_bgr=annotate(
                        snap, tv.bbox if tv else None,
                        f"T{tid} BORN", "track_birth",
                        context_boxes=[c for c in live_ctx if c[0] != tid],
                    ),
                )
            for tid in events.deaths:
                self.home.journal.log(
                    "track_death", cam=self.camera_id,
                    cam_name=self.home.name_of(self.camera_id), tid=tid,
                    frame_bgr=annotate(
                        snap, prev_boxes.get(tid),
                        f"T{tid} DIED", "track_death",
                        context_boxes=live_ctx,
                    ),
                )
        if events.deaths:
            # Seat-dies-with-track must fire from the tracking side too
            # — probes don't run in empty rooms (ghost-seat, 2026-07-24).
            self.home.reap_seat(now)

        newest = batch[-1]
        t_p = time.time()
        await asyncio.to_thread(self._publish, newest)
        self.stage_ms = {
            "encode": round(encode_ms, 1),
            "queue": round(self.queue_ms, 1),
            "pod": round(self.pod_ms, 1),
            "publish": round((time.time() - t_p) * 1000.0, 1),
        }
        self._log_batch(batch, send_frames, obs, encode_ms)

    def _encode(
        self, batch: list, send_w: Optional[int],
    ) -> tuple[list[tuple[int, "bytes | np.ndarray"]], float]:
        """Prepare a batch for the pod, optionally downscaled to send_w
        wide. Raw wire (default): the frames themselves, untouched
        pixels — pod_client ships them codec-free. JPEG wire: encoded
        bytes as before.

        Returns (frames, inv_scale) — multiply pod bbox coords by
        inv_scale to get back to ring-frame space.
        """
        out = []
        inv = 1.0
        for idx, _ts, bgr in batch:
            img = bgr
            if send_w and bgr.shape[1] > send_w:
                s = send_w / bgr.shape[1]
                inv = 1.0 / s
                img = cv2.resize(bgr, (send_w, int(bgr.shape[0] * s)))
            if FRAME_WIRE == "raw":
                out.append((idx, img))
                continue
            ok, jpg = cv2.imencode(
                ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            if ok:
                out.append((idx, jpg.tobytes()))
        return out, inv

    # ── overlay publish (sync, off the event loop) ───────────────────
    def _publish(self, pipeline_frame: tuple) -> None:
        """Publish the live view — the ONLY writer of latest_jpeg.

        Always draws on the frame the pipeline actually analysed, so
        the boxes sit exactly where the tracker saw the person. That
        is the whole job of this view: it is a DEBUG/DEV instrument
        for judging whether tracking and the seat are correct, and
        misplaced boxes make it useless for that (Aryaman
        2026-07-29). The picture therefore runs one pod round-trip
        behind real time; `lag_s` reports exactly how far, and /state
        plus the watchdogs read it.

        Being the only writer is what makes the view monotonic:
        batches always advance, so the picture can never jump
        backwards. The earlier design had a second, fresher writer
        racing this one, which is what produced the visible rewind.
        """
        _idx, pipeline_ts, frame = pipeline_frame
        self.lag_s = time.time() - pipeline_ts
        self.latest_jpeg = self._overlay(
            frame, self.home.seats_on(self.camera_id), self._body_cos,
        )

    # ── OUT-OF-BAND IDENTITY (2026-07-24 decoupling) ─────────────────
    # Face (pod /faces, no SAM, ~25ms/crop) + body (local OSNet) + seat
    # fold + harvest — on the newest frame + current track boxes, at its
    # own cadence. The tracking loop never pays for any of this.
    async def probe_loop(self) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                if not FACES_ENABLED:
                    continue
                now = time.time()
                # Dense while NOBODY is located, light once anyone is.
                # Deliberately "anyone" and not "everyone the camera
                # covers": the stricter reading would hold a shared
                # camera at search cadence for as long as one resident
                # is out of the house, multiplying face-probe cost for
                # a person who isn't there. Finding the second resident
                # a little slower is the cheaper mistake.
                gap = (
                    FACE_PROBE_SEARCH_S if not self.home.seats
                    else FACE_PROBE_LOCKED_S
                )
                if now - self._probed_at < gap:
                    continue
                if not self.state.tracks or not self.ring:
                    continue
                _idx, f_ts, frame = self.ring[-1]
                if now - f_ts > STALE_INTAKE_S:
                    continue
                boxes = [
                    (tid, *tv.bbox) for tid, tv in self.state.tracks.items()
                ]
                # Faces are scored on the NATIVE frame (the ring is
                # downscaled; a ~60px real face is ~22px at 1280 and
                # scores as untrustworthy — measured 2026-07-24). Boxes
                # are ring-space → scale up for the /faces call.
                probe_frame = frame
                face_boxes = boxes
                if self.latest_native is not None:
                    n_ts, native = self.latest_native
                    if now - n_ts <= STALE_INTAKE_S:
                        s = native.shape[1] / frame.shape[1]
                        probe_frame = native
                        face_boxes = [
                            (tid, x1 * s, y1 * s, x2 * s, y2 * s)
                            for tid, x1, y1, x2, y2 in boxes
                        ]
                self._probed_at = now
                # Raw wire: ship pixels as-is — probing encoded NATIVE
                # frames to JPEG every 1.5s while searching, for a
                # loopback hop (see FRAME_WIRE). t_e/t_w bracket the
                # payload prep so the probe stage timings below survive
                # (encode ≈ 0 on the raw wire, by design).
                t_e = time.time()
                if FRAME_WIRE == "raw":
                    payload = probe_frame
                else:
                    payload = await asyncio.to_thread(
                        lambda: cv2.imencode(
                            ".jpg", probe_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 88],
                        )[1].tobytes(),
                    )
                t_w = time.time()
                # While SEARCHING, also detect faces on the WHOLE frame:
                # crop boxes ride the tracking pipeline and lag the world
                # by 5-13s, so a close walk-past crops where the person
                # WAS (live 2026-07-24, seat stayed empty through a
                # clear pass). Full-frame is lag-immune for close faces.
                # Full-frame scanning is a SEARCH aid: crop boxes lag the
                # world by 5-13s, so a close walk-past crops where the
                # person was. Run it while a resident THIS CAMERA COVERS
                # is still unlocated.
                #
                # Not "nobody is seated on this camera" — that would keep
                # full-frame on for a camera showing a guest while the
                # only resident is already located elsewhere, which is
                # cost this never used to pay. With one resident this
                # expression is exactly the old home-wide `seat is None`.
                searching = any(
                    pid not in self.home.seats
                    for pid in self.home.covered(self.camera_id)
                )
                hits = await self.home.sam_for(self.camera_id).faces(
                    payload, face_boxes, full=searching,
                )
                # COVERAGE scan (searching only): every ring frame since
                # the last probe, ≥PROBE_SCAN_GAP_S apart, full-frame in
                # parallel — the perfect frame of a brief crossing can
                # never fall BETWEEN checks (coverage, not sampling —
                # Aryaman's requirement 2026-07-24). Only when the ring
                # is native-res (probe_frame is the ring's newest
                # object); a downscaled ring crushes faces.
                if searching and probe_frame is frame:
                    extra, last_t = [], f_ts
                    for _i, ts2, fr2 in reversed(self.ring[:-1]):
                        if ts2 <= self._scanned_to:
                            break
                        if last_t - ts2 >= PROBE_SCAN_GAP_S:
                            extra.append(fr2)
                            last_t = ts2
                        if len(extra) >= PROBE_SCAN_MAX:
                            break
                    if extra:
                        if FRAME_WIRE == "raw":
                            payloads = extra
                        else:
                            payloads = await asyncio.to_thread(lambda: [
                                cv2.imencode(
                                    ".jpg", fr,
                                    [cv2.IMWRITE_JPEG_QUALITY, 88],
                                )[1].tobytes() for fr in extra
                            ])
                        # Politeness cap: 2 concurrent scans, never a
                        # burst — a 5-wide burst landing on a SAM3
                        # reground ground the shared A40 to a 35s pod
                        # call (measured 2026-07-24). Coverage is
                        # unchanged; the burst is flattened.
                        sem = asyncio.Semaphore(2)

                        async def _scan(j):
                            async with sem:
                                return await self.home.sam_for(self.camera_id).faces(
                                    j, [], full=True,
                                )

                        for more in await asyncio.gather(
                            *[_scan(j) for j in payloads],
                            return_exceptions=True,
                        ):
                            if isinstance(more, list):
                                hits.extend(more)
                self._scanned_to = f_ts
                # Full-frame hits arrive with cx/cy in probe-frame
                # space; bring them back to ring space for assignment.
                if probe_frame is not frame:
                    inv = frame.shape[1] / probe_frame.shape[1]
                    hits = [
                        (tid, emb, px, det, cx * inv, cy * inv)
                        for tid, emb, px, det, cx, cy in hits
                    ]
                t_b = time.time()
                await asyncio.to_thread(
                    self._fold_probe, hits, frame, boxes, now,
                )
                while self._pending_face:
                    await self._write_face(*self._pending_face.pop(0))
                self.probe_ms = {
                    "encode": round((t_w - t_e) * 1000.0, 1),
                    "faces_wire": round((t_b - t_w) * 1000.0, 1),
                    "body_fold": round((time.time() - t_b) * 1000.0, 1),
                }
            except Exception:                # noqa: BLE001 — keep probing
                log.exception("identity.probe_failed cam=%s", self.camera_id)

    def _harvest_face(
        self, patient_id: "UUID", best: "tuple | None",
    ) -> None:
        """Add this locked look at the patient to their face gallery.

        Caller has already passed `harvest_eligible`: face-confirmed,
        confirmed at least twice, solo in frame, track old enough. So
        WHO this is has been settled by the strongest signal available;
        all that is decided here is whether the sample is worth keeping.

        Two rules, both borrowed from the body pool because it learned
        them the hard way:

        DIVERSITY, not repetition. A lock that lingers would otherwise
        deposit a hundred near-identical embeddings and crowd out every
        other view. Anything too close to a row already held is dropped.

        A CAP, so the learned rows can never swamp the enrolled ones.
        `source='selfie'` is ground truth — a human said this is them —
        and `source='harvested'` is the system's own opinion. They stay
        distinguishable in the table forever, so a bad run is undone by
        deleting one source rather than by re-enrolling.

        Never raises: teaching the gallery must not be able to cost a
        probe.
        """
        if best is None or not FACE_HARVEST_ENABLED:
            return
        emb, det, px = best
        if det < FACE_HARVEST_MIN_DET or px < FACE_HARVEST_MIN_PX:
            return
        now = time.monotonic()
        if now - self._face_harvested_at < FACE_HARVEST_MIN_GAP_S:
            return
        matrix = self.home.face_galleries.get(patient_id)
        if matrix is not None and matrix.size:
            # Already have this view. `topk_mean_cosine_match` is the
            # same comparison the matcher uses, so "too similar to keep"
            # means the same thing here as it does there.
            if topk_mean_cosine_match(emb, matrix).score >= FACE_HARVEST_DEDUP:
                return
            if int(matrix.shape[0]) >= FACE_HARVEST_MAX_ROWS:
                return
        self._face_harvested_at = now
        # QUEUED, not scheduled here. `_fold_probe` runs inside
        # asyncio.to_thread, so there is no event loop in this thread and
        # `ensure_future` would raise — the probe's own except would then
        # swallow it and the harvest would silently never happen. The
        # async caller drains this straight after the fold returns.
        self._pending_face.append((patient_id, emb, det, px))

    async def _write_face(
        self, patient_id: "UUID", emb, det: float, px: float,
    ) -> None:
        try:
            from src.db.session import get_sessionmaker
            from src.repos import face_embedding_repo

            sm = get_sessionmaker()
            async with sm() as session:
                await face_embedding_repo.insert_harvested(
                    session, patient_id=patient_id,
                    embedding=emb.astype(np.float32).tobytes(),
                    det_score=float(det),
                )
                await session.commit()
            log.warning(
                "identity.face_harvested patient=%s det=%.2f px=%.0f cam=%s",
                str(patient_id)[:8], det, px, self.camera_id,
            )
        except Exception:  # noqa: BLE001 — learning must not cost a probe
            log.warning("identity.face_harvest_failed", exc_info=True)

    def _fold_probe(
        self, hits: list, frame: np.ndarray, boxes: list, now: float,
    ) -> None:
        # The residents THIS camera watches. One entry for a private
        # camera (and then everything below is the pre-multi-patient
        # path with a loop of length one around it); several for a
        # shared room. Empty means nobody on this camera has enrolled
        # a face, so there is no identity question to ask.
        covered = self.home.covered(self.camera_id)
        # (track_id, patient_id) → best (cosine, face_px) this probe.
        faces_by_key: dict[tuple[int, UUID], tuple[float, float]] = {}
        # track_id → (embedding, det) for the best face seen this probe.
        # Kept for the face harvest at the bottom, which learns only from
        # a face-confirmed lock and therefore needs the embedding that
        # was in front of it at that moment.
        face_emb_by_tid: dict[int, tuple[np.ndarray, float, float]] = {}
        for tid, emb, px, det, cx, cy in hits:
            if tid == -1:
                # Full-frame hit — assign to a track, never guess: it
                # must fall inside exactly ONE (padded) box or it is
                # discarded.
                tid = _assign_face(cx, cy, boxes)
                if tid is None:
                    continue
            elif _contested(cx, cy, boxes, own=tid):
                # Crop hit whose position another track's box also
                # claims — overlap/adjacency, nobody's evidence.
                continue
            # One face, asked about each resident in turn. The galleries
            # are separate matrices rather than one stacked matrix with
            # labels: each is ~10-30 rows, the scan is microseconds, and
            # keeping them apart means a resident's top-K mean is
            # computed over their OWN enrollment — stacking would let a
            # crowded gallery borrow another person's best rows.
            # Biggest face this probe wins — the harvest below wants the
            # best look at the person, not the last one parsed.
            keep = face_emb_by_tid.get(tid)
            if keep is None or px > keep[2]:
                face_emb_by_tid[tid] = (emb, det, px)
            for pid in covered:
                matrix = self.home.face_galleries.get(pid)
                if matrix is None or not matrix.size:
                    continue
                cos = topk_mean_cosine_match(emb, matrix).score
                # Every sighting feeds that resident's history for this
                # track — the judgment below reads the HISTORY's robust
                # statistic, not the frame.
                self._sightings.setdefault(
                    (tid, pid), SightingHistory(),
                ).add(cos, px, now=now)
                prev = faces_by_key.get((tid, pid))
                if prev is None or px > prev[1]:
                    faces_by_key[(tid, pid)] = (cos, px)
        # Histories die with their tracks (seat-reference lifecycle).
        live_tids = {b[0] for b in boxes}
        self._sightings = {
            k: v for k, v in self._sightings.items() if k[0] in live_tids
        }

        cands: list[Candidate] = []
        body_by_key: dict[tuple[int, UUID], float] = {}
        for tid, x1, y1, x2, y2 in boxes:
            # ONE body embedding per track, scored against every covered
            # resident's pool. The embedding extraction is the expensive
            # part and it does not repeat per resident.
            body_cos: dict[UUID, float] = {}
            emb = extract_body_embedding(
                frame_bgr=frame, bbox=(x1, y1, x2, y2),
            )
            if emb is not None and covered:
                m = self.home.body_gallery.match(
                    patient_ids=covered, probe_embedding=emb,
                )
                if m is not None:
                    body_cos[m.patient_id] = m.score
                    body_by_key[(tid, m.patient_id)] = m.score
            for pid in covered:
                f = faces_by_key.get((tid, pid))
                # Face side: the track's HISTORY for THIS resident
                # speaks (top-K robust to back-of-head garbage;
                # consistency waiver lets N marginal sightings
                # substitute for one big face). A single overwhelming
                # sighting (close pass) still speaks for itself
                # instantly; anything weaker with a thin history is
                # PROVISIONAL (can hold a seat, can't take one).
                hist = self._sightings.get((tid, pid))
                hist_ev = hist.evidence(now=now) if hist is not None else 0.0
                single_ev = face_conf(f[0], f[1]) if f else 0.0
                # A single frame speaks for itself ONLY when overwhelming
                # (close pass); otherwise the history's verdict IS the
                # evidence. Never max() a marginal single over the history
                # — live 2026-07-24 (T13): 25 near-zeros then one 0.46,
                # and the max() bypassed the history while the history's
                # depth disabled the provisional gate. The two defenses
                # cancelled at their seam; an impostor locked.
                if single_ev >= INSTANT_FACE_CONF:
                    face_ev = max(hist_ev, single_ev)
                    provisional = False
                else:
                    face_ev = hist_ev
                    provisional = (
                        hist is None or hist.hits_at(now) < FUSION_MIN_HITS
                    )
                body_ev = body_conf(body_cos.get(pid))
                if face_ev == 0.0 and body_ev == 0.0:
                    continue
                if face_ev >= body_ev:
                    score, anchor = face_ev, "face"
                else:
                    score, anchor = body_ev, "body"
                cands.append(Candidate(
                    camera_id=self.camera_id, track_id=tid, patient_id=pid,
                    score=score, anchor=anchor,
                    provisional=(anchor == "face" and provisional),
                ))
        self._body_cos = {
            tid: sc for (tid, _pid), sc in body_by_key.items()
        }
        self.home.fold_candidates(cands, now)
        self.probe_history.append({
            "ts": round(now, 1),
            "locked": bool(self.home.seats),
            "tracks": len(boxes),
            "faces": [
                [tid, str(pid)[:8], round(cos, 3), int(px)]
                for (tid, pid), (cos, px) in faces_by_key.items()
            ],
            "body": [
                [tid, str(pid)[:8], round(cos, 3)]
                for (tid, pid), cos in body_by_key.items()
            ],
            "cands": [
                [c.track_id, str(c.patient_id)[:8], round(c.score, 3),
                 c.anchor, bool(c.provisional)]
                for c in cands
            ],
        })
        log.info(
            "identity.probe cam=%s tracks=%d covered=%d faces=%s cands=%s",
            self.camera_id, len(boxes), len(covered),
            [
                (tid, str(pid)[:8], round(cos, 2), int(px))
                for (tid, pid), (cos, px) in faces_by_key.items()
            ],
            [
                (c.track_id, str(c.patient_id)[:8], round(c.score, 2),
                 c.anchor + ("?" if c.provisional else ""))
                for c in cands
            ],
        )

        # `seat_on` returns the strongest seat on this camera, which is
        # unambiguous here: harvest_eligible requires `solo`, so there is
        # exactly one live track and therefore at most one seat to pick
        # from. Multi-person harvesting still does not exist — a shared
        # lounge with two people in it harvests for neither, which is
        # the pre-existing rule and the right one (a crossing is exactly
        # when the pool gets poisoned).
        held = self.home.seat_on(self.camera_id)
        if held is not None:
            seat_pid, seat = held
            tv = self.state.tracks.get(seat.track_id)
            if tv is not None and harvest_eligible(
                seat, camera_id=self.camera_id, track_id=seat.track_id,
                solo=self.state.solo, track_age_s=tv.age(now=now),
                min_age_s=HARVEST_MIN_AGE_S,
            ):
                # FACE HARVEST — same gate, same moment, one more write.
                #
                # The enrolled gallery is selfies (source='selfie', 25 of
                # them from one session) and this camera is on a wall
                # looking down across a room. Measured on a two-minute
                # walk: 55 face sightings, best cosine 0.39 against a
                # ~0.40 bar, and size did not help — a 132px face scored
                # 0.20 while a 48px one scored 0.39. That is a viewpoint
                # gap, not a recognition failure, and no amount of extra
                # frames or pixels closes it.
                #
                # A face-confirmed lock is the one moment we KNOW who
                # this is and have their face in front of us at this
                # camera's angle, distance and light. Learning there
                # widens the gallery toward the walking case over days.
                #
                # Learned FROM A FACE LOCK, deliberately — not from the
                # body lock that would bootstrap faster. Face is the
                # decisive signal, so its gallery must never be taught
                # by a weaker one; that inversion is where poisoning
                # lives, and `harvest_eligible` (face_confirmed, two
                # confirmations, solo, track age) is already hardened
                # against it for the body pool.
                self._harvest_face(seat_pid, face_emb_by_tid.get(
                    seat.track_id))

                emb = extract_body_embedding(frame_bgr=frame, bbox=tv.bbox)
                if emb is not None:
                    # Harvest into the pool of the resident who actually
                    # holds this seat — never a room default. A harvest
                    # filed under the wrong person poisons that pool for
                    # its whole 4h TTL and drives body false-positives
                    # long after the mistake (the aunt lesson).
                    self.home.body_gallery.harvest(
                        patient_id=seat_pid, embedding=emb,
                        confidence=seat.score,
                    )
                    self.home.body_gallery.remember_recent(
                        patient_id=seat_pid, embedding=emb,
                        confidence=seat.score,
                    )

    def _overlay(
        self, frame: np.ndarray, held: list[tuple[UUID, Seat]],
        body_cos: dict[int, float],
    ) -> bytes:
        img = frame.copy()
        h = img.shape[0]
        # Use the LOCAL snapshot throughout — home.seats can be
        # reassigned by another camera's fold between two reads (TOCTOU).
        # EVERY seated resident is drawn, not just the strongest: this
        # view is how a shared camera gets verified, and showing one of
        # two people would make a working two-resident lock look broken.
        by_track = {s.track_id: (pid, s) for pid, s in held}
        for rx1, ry1, rx2, ry2 in self.screen_map.live(time.monotonic()):
            cv2.rectangle(img, (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                          (0, 0, 200), 2)
        for tid, tv in self.state.tracks.items():
            x1, y1, x2, y2 = (int(v) for v in tv.bbox)
            seated = by_track.get(tid)
            col = (80, 220, 80) if seated else (60, 200, 255)
            tag = (
                f"{str(seated[0])[:8]} T{tid} "
                f"{seated[1].score:.2f}/{seated[1].anchor}"
                if seated
                else f"T{tid}" + (
                    f" b{body_cos[tid]:.2f}" if tid in body_cos else ""
                )
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if seated else 2)
            cv2.putText(img, tag, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        self.hud = (
            f"{self.home.name_of(self.camera_id)} | "
            f"lag {self.lag_s:.1f}s pod {self.pod_ms:.0f}ms"
            f" q {self.queue_ms:.0f}ms | tracks {len(self.state.tracks)}"
            # ASCII only: cv2.putText renders non-ASCII as "???"
            + (" | " + " ".join(
                f"{str(pid)[:8]} {s.score:.2f}@{s.anchor}"
                + ("*face" if s.face_confirmed else "")
                for pid, s in held
            ) if held else " | seat -")
        )
        cv2.rectangle(img, (0, h - 26), (len(self.hud) * 9 + 16, h), (0, 0, 0), -1)
        cv2.putText(img, self.hud, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return out.tobytes() if ok else b""
