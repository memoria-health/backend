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
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import cv2
import numpy as np

from identity.gallery import (
    LABEL_PATIENT,
    CrossClipBodyGallery,
)
from identity.body_embeddings import extract_body_embedding
from identity.display_veto import ScreenMap, apply_veto
from identity.matching import topk_mean_cosine_match
from identity.wire import RemoteObservations
from identity.pod_client import SamServiceClient
from identity.batching import plan_batch
from identity.fusion import SightingHistory
from identity.harvest import harvest_eligible
from identity.scoring import body_conf, face_conf
from identity.journal import EventJournal
from identity.seat import (
    Candidate,
    Seat,
    heal_seat,
    notary_contradiction,
    update_seat,
)
from identity.state import CameraState

log = logging.getLogger(__name__)

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
SCREEN_REFRESH_S = 120.0       # screens are furniture
SEAT_FLOOR = 0.4               # combined-score floor to take an empty seat
                               # (calibrated with scoring.py 2026-07-24 so
                               # the legacy brain's proven face locks clear it)
SEAT_DELTA = 0.1               # beat-by margin (and ambiguity margin)
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
                               # (a dead LiveKit reader must not leave a
                               # frozen track holding the seat forever)
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


@dataclass
class IdentityHome:
    """Home-wide shared state: the single patient seat + shared models."""

    patient_id: UUID
    sam: SamServiceClient
    body_gallery: CrossClipBodyGallery
    face_matrix: np.ndarray                  # (N,512) enrollment, refreshed
    # Pod admission control. Was a single Lock ("single-flight A40"),
    # which measurement on 2026-07-29 showed was the actual bottleneck:
    # with both rooms occupied the machine sat ~97% IDLE (GPU 5-8%, 3 of
    # 96 CPU cores, 5.6GB of 46GB) while cameras queued seconds behind
    # each other. Measured back-to-back vs concurrent, same payloads:
    # parallel finished 1.5-1.9x faster and pod-side compute per call
    # did NOT rise (~1.9s either way) — the pod serves distinct camera
    # sessions concurrently without contending.
    #
    # Still bounded, not removed: sessions are per-camera on the pod, so
    # concurrency across cameras is safe, but unbounded admission would
    # let a backlog burst arrive all at once. One slot per camera is the
    # intent; NEMARA_POD_CONCURRENCY overrides.
    #
    # NOTE: a single camera still issues one call at a time — each
    # CameraLoop awaits its own batch before starting the next — so
    # per-session ordering on the pod is preserved by construction.
    pod_lock: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(
            max(1, int(os.environ.get("NEMARA_POD_CONCURRENCY", "2")))
        )
    )
    # Guards `seat`: candidates fold from per-camera worker THREADS
    # (asyncio.to_thread), so the read-modify-write must serialize.
    seat_lock: threading.Lock = field(default_factory=threading.Lock)
    seat: Optional[Seat] = None
    cameras: dict[str, "CameraLoop"] = field(default_factory=dict)
    # Extra pod workers, one entry per additional base URL in
    # NEMARA_POD_URLS (comma-separated). Empty = every camera talks to
    # the single `sam` client, i.e. exactly today's behaviour — which is
    # what makes this revertible by unsetting one variable.
    pods: list[SamServiceClient] = field(default_factory=list)

    def sam_for(self, camera_id: str) -> SamServiceClient:
        """Which pod worker serves this camera.

        The pod serialises ALL inference behind a process-global lock,
        so two cameras on one worker take turns no matter how idle the
        GPU is (measured 2026-07-29: 5-8% SM, 3 of 96 cores, 5.6GB of
        46GB, while cameras queued seconds apart). Two SEPARATE pod
        processes ran 1.5-1.9x faster with per-call compute unchanged,
        because each has its own lock and its own model copy.

        Assignment is a stable hash, not round-robin, because EdgeTAM
        sessions live in a process-global `_SESSIONS` dict on the pod:
        a camera that moved between workers would find no session and
        silently re-ground every tick. Stable hash keeps each camera on
        one worker for as long as the worker lives.
        """
        if not self.pods:
            return self.sam
        pool = [self.sam, *self.pods]
        # sum of bytes, not hash(): PYTHONHASHSEED randomises str hashing
        # per process, so a restart would reshuffle every camera onto a
        # different worker and throw away all their sessions.
        return pool[sum(camera_id.encode()) % len(pool)]
    journal: Optional[EventJournal] = None
    # camera-id-8-prefix → room name (from the cameras table) — every
    # user-facing surface shows ROOM NAMES, never raw ids (Aryaman).
    cam_names: dict = field(default_factory=dict)
    # Swap-healing streak (seat-follows-the-face): consecutive folds
    # whose verdict was "the notary moved" (see notary_contradiction).
    _contra_key: Optional[tuple] = None
    _contra_streak: int = 0
    # (anchor, confirm_latency_s) per lock — the "how long until we
    # know it's the patient" metric, measured per track (soak stats).
    confirm_stats: list = field(default_factory=list)

    def name_of(self, cam_id: str) -> str:
        return self.cam_names.get(cam_id[:8], cam_id[:8])

    def live_keys(self) -> set[tuple[str, int]]:
        # Reads the frozenset snapshots (GIL-atomic ref reads) — safe
        # from worker threads while event-loop apply() mutates tracks.
        return {
            (cam_id, tid)
            for cam_id, cam_loop in list(self.cameras.items())
            for tid in cam_loop.state.live_ids
        }

    SWAP_HEAL_ROUNDS = 3   # consecutive contradiction verdicts to heal

    def fold_candidates(self, cands: list[Candidate], now: float) -> None:
        with self.seat_lock:
            old = self.seat
            seat = update_seat(
                self.seat, cands, live=self.live_keys(),
                floor=SEAT_FLOOR, delta=SEAT_DELTA, now=now,
            )
            # Swap healing: the tracker can trade two people's track
            # numbers (live 2026-07-24, twice) and the seat rides the
            # number. When the holder's evidence collapses while
            # another track sustains history-backed face evidence for
            # SWAP_HEAL_ROUNDS consecutive folds, the seat follows the
            # face. The ratcheted reference makes an ordinary steal
            # impossible post-swap — this is the only recovery path.
            ch = notary_contradiction(seat, cands, floor=SEAT_FLOOR)
            if ch is not None and ch.key == self._contra_key:
                self._contra_streak += 1
            elif ch is not None:
                self._contra_key, self._contra_streak = ch.key, 1
            else:
                self._contra_key, self._contra_streak = None, 0
            if ch is not None and self._contra_streak >= self.SWAP_HEAL_ROUNDS:
                seat = heal_seat(ch, now=now)
                self._contra_key, self._contra_streak = None, 0
            self.seat = seat
        self._journal_transition(old, seat, now)

    def _journal_transition(
        self, old: Optional[Seat], new: Optional[Seat], now: float,
    ) -> None:
        """Journal seat changes with a snapshot (outside seat_lock —
        JPEG writes must never sit inside the hot mutex)."""
        if self.journal is None or old == new:
            return
        from identity.journal import annotate

        if new is not None and (old is None or old.key != new.key):
            kind = "seat_locked" if old is None else "seat_moved"
            cam = self.cameras.get(new.camera_id)
            frame = cam.ring[-1][2] if cam and cam.ring else None
            tv = cam.state.tracks.get(new.track_id) if cam else None
            latency = round(now - tv.born_at, 1) if tv else None
            if latency is not None:
                self.confirm_stats.append((new.anchor, latency))
                del self.confirm_stats[:-200]
            if frame is not None:
                ctx = [
                    (t, v.bbox) for t, v in cam.state.tracks.items()
                    if t != new.track_id
                ] if cam else []
                frame = annotate(
                    frame, tv.bbox if tv else None,
                    f"T{new.track_id} PATIENT {new.score:.2f}@{new.anchor}",
                    kind, context_boxes=ctx,
                )
            self.journal.log(
                kind, cam=new.camera_id,
                cam_name=self.name_of(new.camera_id), tid=new.track_id,
                score=round(new.score, 2), anchor=new.anchor,
                confirm_latency_s=latency, frame_bgr=frame,
            )
        elif new is None and old is not None:
            cam = self.cameras.get(old.camera_id)
            frame = cam.ring[-1][2] if cam and cam.ring else None
            tv = cam.state.tracks.get(old.track_id) if cam else None
            if frame is not None:
                ctx = [
                    (t, v.bbox) for t, v in cam.state.tracks.items()
                    if t != old.track_id
                ] if cam else []
                frame = annotate(
                    frame, tv.bbox if tv else None,
                    f"T{old.track_id} PATIENT UNSET", "seat_died",
                    context_boxes=ctx,
                )
            self.journal.log(
                "seat_died", cam=old.camera_id,
                cam_name=self.name_of(old.camera_id), tid=old.track_id,
                frame_bgr=frame,
            )

    def reap_seat(self, now: float) -> None:
        """Seat-dies-with-track, enforced from the TRACKING side.

        Probes don't run in an empty room, so a seat whose track died
        with nobody left would survive as a ghost until the next probe
        anywhere (live 2026-07-24: dead track 10 still 'present' —
        harmless in the viewer, WRONG as the detection-facing state
        flag). An empty candidate fold applies rule 1 only.
        """
        with self.seat_lock:
            old = self.seat
            self.seat = update_seat(
                self.seat, [], live=self.live_keys(),
                floor=SEAT_FLOOR, delta=SEAT_DELTA, now=now,
            )
            new = self.seat
        self._journal_transition(old, new, now)


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

    def __init__(self, *, camera_id: str, home: IdentityHome) -> None:
        self.camera_id = camera_id
        self.home = home
        self.state = CameraState(camera_id=camera_id)
        self.screen_map = ScreenMap()
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
        self._scanned_to = 0.0   # coverage-scan high-water mark (ring ts)
        self._pod_epoch: Optional[str] = None  # session-id-reuse guard
        self._probe_task: Optional[asyncio.Task] = None
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
        self._sightings: dict[int, SightingHistory] = {}
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
            seat = self.home.seat
            if not self.state.tracks and self._last_shipped is not None:
                interval = EMPTY_INTERVAL_S
            elif seat is not None and seat.camera_id != self.camera_id:
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
            except Exception:                # noqa: BLE001 — keep looping
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
        empty_poll = not self.state.tracks and self._last_shipped is not None
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
        async with self.home.pod_lock:       # bounded pod admission
            t0 = time.time()                 # queue wait ends here
            obs = await self.home.sam_for(self.camera_id).observe(
                send_frames,
                camera_id=f"identity_{self.camera_id}",
                person_thr=0.4,
                face_interval=999,
                face_cap=0,
                detect_screens=want_screens,
            )
            self.home.last_pod_ok = time.time()  # pod-progress watchdog feed
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
            self.screen_map.add(
                [r[:4] for r in obs.screen_regions], now=time.monotonic(),
            )
        obs, vetoed = apply_veto(obs, self.screen_map, now=time.monotonic())

        prev_boxes = {t: tv.bbox for t, tv in self.state.tracks.items()}
        events = self.state.apply(newest_bboxes(obs), now=now)
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
        span = batch[-1][1] - batch[0][1]
        # Split the pod round-trip into the pod's OWN reported compute
        # and everything else (upload + response over the tunnel). The
        # single `pod=` number conflated them, which made it impossible
        # to tell a slow GPU from a saturated wire — and cost a whole
        # afternoon of guessing (2026-07-29). pod_sam is the tracker
        # itself; wire is what we pay to reach it.
        # getattr, not obs.timings_ms: an older pod build (or a test
        # double) has no such field, and losing a LOG LINE must never
        # take the tracking loop down with an AttributeError.
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
        self.latest_jpeg = self._overlay(frame, self.home.seat, self._body_cos)

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
                seat = self.home.seat
                gap = (
                    FACE_PROBE_SEARCH_S if seat is None
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
                hits = await self.home.sam_for(self.camera_id).faces(
                    payload, face_boxes, full=(seat is None),
                )
                # COVERAGE scan (searching only): every ring frame since
                # the last probe, ≥PROBE_SCAN_GAP_S apart, full-frame in
                # parallel — the perfect frame of a brief crossing can
                # never fall BETWEEN checks (coverage, not sampling —
                # Aryaman's requirement 2026-07-24). Only when the ring
                # is native-res (probe_frame is the ring's newest
                # object); a downscaled ring crushes faces.
                if seat is None and probe_frame is frame:
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
                self.probe_ms = {
                    "encode": round((t_w - t_e) * 1000.0, 1),
                    "faces_wire": round((t_b - t_w) * 1000.0, 1),
                    "body_fold": round((time.time() - t_b) * 1000.0, 1),
                }
            except Exception:                # noqa: BLE001 — keep probing
                log.exception("identity.probe_failed cam=%s", self.camera_id)

    def _fold_probe(
        self, hits: list, frame: np.ndarray, boxes: list, now: float,
    ) -> None:
        matrix = self.home.face_matrix
        faces_by_tid: dict[int, tuple[float, float]] = {}
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
            cos = (
                topk_mean_cosine_match(emb, matrix).score
                if matrix.size else None
            )
            if cos is None:
                continue
            # Every sighting feeds the track's history — the judgment
            # below reads the HISTORY's robust statistic, not the frame.
            self._sightings.setdefault(tid, SightingHistory()).add(
                cos, px, now=now,
            )
            prev = faces_by_tid.get(tid)
            if prev is None or px > prev[1]:
                faces_by_tid[tid] = (cos, px)
        # Histories die with their tracks (seat-reference lifecycle).
        live_tids = {b[0] for b in boxes}
        self._sightings = {
            t: v for t, v in self._sightings.items() if t in live_tids
        }

        cands: list[Candidate] = []
        body_cos_by_tid: dict[int, float] = {}
        for tid, x1, y1, x2, y2 in boxes:
            f = faces_by_tid.get(tid)
            body_cos = None
            emb = extract_body_embedding(
                frame_bgr=frame, bbox=(x1, y1, x2, y2),
            )
            if emb is not None:
                m = self.home.body_gallery.match(
                    patient_id=self.home.patient_id, probe_embedding=emb,
                )
                if m is not None and m.label == LABEL_PATIENT:
                    body_cos = m.score
                    body_cos_by_tid[tid] = m.score
            # Face side: the track's HISTORY speaks (top-K robust to
            # back-of-head garbage; consistency waiver lets N marginal
            # sightings substitute for one big face). A single
            # overwhelming sighting (close pass) still speaks for
            # itself instantly; anything weaker with a thin history is
            # PROVISIONAL (can hold a seat, can't take one).
            hist = self._sightings.get(tid)
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
            body_ev = body_conf(body_cos)
            if face_ev == 0.0 and body_ev == 0.0:
                continue
            if face_ev >= body_ev:
                score, anchor = face_ev, "face"
            else:
                score, anchor = body_ev, "body"
            cands.append(Candidate(
                camera_id=self.camera_id, track_id=tid,
                score=score, anchor=anchor,
                provisional=(anchor == "face" and provisional),
            ))
        self._body_cos = body_cos_by_tid
        self.home.fold_candidates(cands, now)
        self.probe_history.append({
            "ts": round(now, 1),
            "locked": self.home.seat is not None,
            "tracks": len(boxes),
            "faces": [
                [tid, round(cos, 3), int(px)]
                for tid, (cos, px) in faces_by_tid.items()
            ],
            "body": [
                [tid, round(cos, 3)] for tid, cos in body_cos_by_tid.items()
            ],
            "cands": [
                [c.track_id, round(c.score, 3), c.anchor,
                 bool(c.provisional)]
                for c in cands
            ],
        })
        log.info(
            "identity.probe cam=%s tracks=%d faces=%s cands=%s",
            self.camera_id, len(boxes),
            [
                (tid, round(cos, 2), int(px))
                for tid, (cos, px) in faces_by_tid.items()
            ],
            [
                (c.track_id, round(c.score, 2),
                 c.anchor + ("?" if c.provisional else ""))
                for c in cands
            ],
        )

        seat = self.home.seat
        if seat is not None and seat.camera_id == self.camera_id:
            tv = self.state.tracks.get(seat.track_id)
            if tv is not None and harvest_eligible(
                seat, camera_id=self.camera_id, track_id=seat.track_id,
                solo=self.state.solo, track_age_s=tv.age(now=now),
                min_age_s=HARVEST_MIN_AGE_S,
            ):
                emb = extract_body_embedding(frame_bgr=frame, bbox=tv.bbox)
                if emb is not None:
                    self.home.body_gallery.harvest(
                        patient_id=self.home.patient_id, embedding=emb,
                        label=LABEL_PATIENT, confidence=seat.score,
                    )
                    self.home.body_gallery.remember_recent(
                        patient_id=self.home.patient_id, embedding=emb,
                        confidence=seat.score,
                    )

    def _overlay(
        self, frame: np.ndarray, seat: Optional[Seat],
        body_cos: dict[int, float],
    ) -> bytes:
        img = frame.copy()
        h = img.shape[0]
        for rx1, ry1, rx2, ry2 in self.screen_map.live(time.monotonic()):
            cv2.rectangle(img, (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                          (0, 0, 200), 2)
        for tid, tv in self.state.tracks.items():
            x1, y1, x2, y2 = (int(v) for v in tv.bbox)
            is_seat = seat is not None and seat.key == (self.camera_id, tid)
            col = (80, 220, 80) if is_seat else (60, 200, 255)
            tag = (
                f"PATIENT T{tid} {seat.score:.2f}/{seat.anchor}" if is_seat
                else f"T{tid}" + (
                    f" b{body_cos[tid]:.2f}" if tid in body_cos else ""
                )
            )
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if is_seat else 2)
            cv2.putText(img, tag, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # Use the LOCAL seat snapshot — home.seat can be reassigned to
        # None by another camera's fold between two reads (TOCTOU).
        self.hud = (
            f"{self.home.name_of(self.camera_id)} | "
            f"lag {self.lag_s:.1f}s pod {self.pod_ms:.0f}ms"
            f" q {self.queue_ms:.0f}ms | tracks {len(self.state.tracks)}"
            + (f" | seat {seat.score:.2f}@{seat.anchor}"
               + ("*face" if seat.face_confirmed else "")
               if seat else " | seat -")  # ASCII: cv2.putText renders
                                          # non-ASCII as "???"
        )
        cv2.rectangle(img, (0, h - 26), (len(self.hud) * 9 + 16, h), (0, 0, 0), -1)
        cv2.putText(img, self.hud, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
        ok, out = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return out.tobytes() if ok else b""
