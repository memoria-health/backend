"""IdentityHome — the home-wide half of the identity service.

Seats, residents and the journal live here: state that spans every
camera in a home. `CameraLoop` (loop.py) is the per-camera half, with
its own intake, ticks and probes. The two were one 1900-line module;
they have different lifetimes and almost no shared state.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

from identity.gallery import CrossClipBodyGallery
from identity.journal import EventJournal
from identity.pod_client import SamServiceClient
from identity.seat import (
    Candidate, Seat, carry_seat, resolve_track_conflicts, update_seat,
)

# Same logger name as before the split, so every existing grep, alert
# and runbook that keys on "identity.loop" keeps working.
log = logging.getLogger("identity.loop")


SEAT_FLOOR = 0.4               # combined-score floor to take an empty seat
                               # (calibrated with scoring.py 2026-07-24 so
                               # the legacy brain's proven face locks clear it)

                               # (calibrated with scoring.py 2026-07-24 so
                               # the legacy brain's proven face locks clear it)
SEAT_DELTA = 0.1               # beat-by margin (and ambiguity margin)

                               # (a dead LiveKit reader must not leave a
                               # frozen track holding the seat forever)
# How long a dead track's last box stays available for `carry_seat` to
# match a renumbered track against. Not a grace period for the SEAT —
# the seat still dies this tick if nothing overlaps — only the window in
# which "the person who was standing here" is still a meaningful claim.
#
# 6s: the measured gap between a seated track dying and its replacement
# appearing is a median 1.9s and a max 2.8s, so this clears the observed
# range twice over while staying far short of the time it takes anyone
# to leave a room and someone else to occupy the same box.
DEAD_BOX_TTL_S = 6.0


@dataclass
class IdentityHome:
    """Room-wide shared state: one seat per resident + shared models."""

    sam: SamServiceClient
    body_gallery: CrossClipBodyGallery
    # patient_id → (N,512) ArcFace enrollment, refreshed on a timer.
    # One entry per resident that ANY camera in this room covers; a
    # camera asks only about the residents IT covers (see `covered`).
    face_galleries: dict[UUID, np.ndarray]
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
    # PER-WORKER admission (2026-08-15 review finding 3 on the v3.0
    # worker pool): one shared semaphore sized for a single pod let one
    # wedged worker's 25s-timeout calls hold the room's whole admission
    # budget and starve cameras on a HEALTHY sibling worker — and
    # capped total concurrency at single-pod levels no matter how many
    # workers exist. One semaphore per pod client, created lazily; the
    # legacy pod_lock above stays for anything still holding it.
    pod_locks: dict = field(default_factory=dict)

    def lock_for(self, client) -> asyncio.Semaphore:
        sem = self.pod_locks.get(id(client))
        if sem is None:
            sem = self.pod_locks[id(client)] = asyncio.Semaphore(
                max(1, int(os.environ.get("NEMARA_POD_CONCURRENCY", "2")))
            )
        return sem
    # Guards `seats`: candidates fold from per-camera worker THREADS
    # (asyncio.to_thread), so the read-modify-write must serialize.
    seat_lock: threading.Lock = field(default_factory=threading.Lock)
    # patient_id → that resident's seat. A resident with no entry is
    # simply not located right now, which is the honest answer and the
    # one every consumer must treat as "unknown" rather than "absent".
    seats: dict[UUID, Seat] = field(default_factory=dict)
    # camera-id-8-prefix → the residents that camera watches. Refreshed
    # alongside the galleries, so sharing a camera with a second
    # resident takes effect without restarting the node.
    coverage: dict[str, list[UUID]] = field(default_factory=dict)
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

    def pod_for_dispatch(self, camera_id: str) -> SamServiceClient:
        """Shared-queue dispatch (stateless-worker mode, 2026-08-16):
        first worker with a free admission slot serves the camera.

        Only valid when the camera's session state travels WITH the
        request (NEMARA_STATELESS_DISPATCH=1) — the pin in sam_for
        exists solely because state used to live in worker memory.
        One line of queueing theory motivates this: one line feeding k
        cashiers beats k separate lines (measured: the pinned 2-worker
        fleet sat at 6.1s@10 cams where a shared queue ran ~3.1s).

        The scan starts at the camera's hash slot so an all-busy
        moment degrades to exactly the old pinned behaviour, and
        different cameras don't herd onto the same first worker."""
        pool = [self.sam, *self.pods]
        if len(pool) == 1:
            return self.sam
        start = sum(camera_id.encode()) % len(pool)
        for i in range(len(pool)):
            c = pool[(start + i) % len(pool)]
            # No await between this check and the caller's acquire —
            # asyncio is single-threaded, so the free slot cannot be
            # stolen before the caller enters the semaphore.
            if self.lock_for(c)._value > 0:
                return c
        return pool[start]
    journal: Optional[EventJournal] = None
    # camera-id-8-prefix → room name (from the cameras table) — every
    # user-facing surface shows ROOM NAMES, never raw ids (Aryaman).
    cam_names: dict = field(default_factory=dict)
    # Swap-healing streak (seat-follows-the-face): consecutive folds
    # whose verdict was "the notary moved" (see notary_contradiction).
    # Per resident — a swap on Alice's seat says nothing about Bob's.
    _contra: dict = field(default_factory=dict)   # pid -> (key, streak)
    # (anchor, confirm_latency_s) per lock — the "how long until we
    # know it's the patient" metric, measured per track (soak stats).
    confirm_stats: list = field(default_factory=list)

    def name_of(self, cam_id: str) -> str:
        return self.cam_names.get(cam_id[:8], cam_id[:8])

    def covered(self, cam_id: str) -> list[UUID]:
        """Residents this camera watches, restricted to those with a
        loaded face gallery.

        Coverage and enrollment are separate facts: a caregiver can
        point a camera at a second resident before that resident has
        enrolled a face. Such a resident is dropped here rather than
        carried as an empty gallery, because an empty gallery scores
        0.0 against everything and would only ever add a losing entry
        to every comparison.
        """
        return [
            pid for pid in self.coverage.get(cam_id[:8], ())
            if self.face_galleries.get(pid) is not None
        ]

    def seats_on(self, cam_id: str) -> list[tuple[UUID, Seat]]:
        """Every (resident, seat) held on this camera, strongest first.

        There can be more than one: `resolve_track_conflicts` forbids
        two residents sharing a TRACK, not a camera, and two residents
        on two tracks of one camera is the ordinary shared-lounge
        state.
        """
        return sorted(
            (
                (pid, s) for pid, s in list(self.seats.items())
                if s.camera_id == cam_id
            ),
            key=lambda e: e[1].score, reverse=True,
        )

    def seat_on(self, cam_id: str) -> Optional[tuple[UUID, Seat]]:
        """The strongest seat held on this camera, if any — for callers
        that only need "is SOMEONE located here" (probe cadence, the
        harvest, which `solo` already restricts to one live track)."""
        held = self.seats_on(cam_id)
        return held[0] if held else None

    def home_seat_boxes(self) -> dict:
        """(camera_id, track_id) -> bbox across EVERY camera in the home.

        `CameraLoop._seat_boxes` is the per-camera version; this one
        exists because seats live on the home, not on a loop, and the
        home has no single camera to ask. Wrong owner was a real bug:
        calling the loop's method from here raised AttributeError inside
        `fold_candidates`, every probe failed, and NOTHING could lock a
        seat — the symptom was staring at a camera and never being
        recognised (2026-08-25).
        """
        out: dict = {}
        now = time.time()
        for cam_id, loop in (self.cameras or {}).items():
            # Just-dead tracks FIRST, so a live track of the same number
            # always wins. A recycled id must describe where it is now.
            for tid, (bbox, died_at) in getattr(
                    loop, "_dead_boxes", {}).items():
                if now - died_at <= DEAD_BOX_TTL_S:
                    out[(cam_id, tid)] = bbox
            state = getattr(loop, "state", None)
            for tid, tv in getattr(state, "tracks", {}).items():
                out[(cam_id, tid)] = tv.bbox
        return out

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
        """Fold one round of scores into the seats — once per resident.

        Every resident with a seat is folded even when this round
        produced no candidate for them, because an absent candidate is
        how "silence holds the seat" and "the seat dies with its track"
        both reach `update_seat`.
        """
        by_patient: dict[UUID, list[Candidate]] = defaultdict(list)
        for c in cands:
            by_patient[c.patient_id].append(c)

        with self.seat_lock:
            old = dict(self.seats)
            live = self.live_keys()
            seats: dict[UUID, Seat] = {}
            for pid in set(self.seats) | set(by_patient):
                mine = by_patient.get(pid, [])
                # Offer a re-numbered track before the seat is allowed
                # to die. The tracker renumbers a stationary person
                # constantly; without this, every renumber throws away
                # a face match and costs 20-60s to earn back.
                carried = carry_seat(
                    self.seats.get(pid), live, self.home_seat_boxes(),
                    # Track numbers other residents are sitting on. A
                    # carry must never take one, or this resident's
                    # identity lands on their housemate.
                    taken=frozenset(
                        s.key for q, s in self.seats.items() if q != pid
                    ),
                )
                if carried is not None and self.seats.get(pid) is not None \
                        and carried.track_id != self.seats[pid].track_id:
                    log.info(
                        "identity.seat_carried pid=%s %s -> %s (same person, "
                        "new track number)",
                        str(pid)[:8], self.seats[pid].track_id,
                        carried.track_id,
                    )
                seat = update_seat(
                    carried, mine, live=live,
                    floor=SEAT_FLOOR, delta=SEAT_DELTA, now=now,
                )
                # Swap healing: the tracker can trade two people's track
                # numbers (live 2026-07-24, twice) and the seat rides the
                # number. When the holder's evidence collapses while
                # another track sustains history-backed face evidence for
                # SWAP_HEAL_ROUNDS consecutive folds, the seat follows the
                # face. The ratcheted reference makes an ordinary steal
                # impossible post-swap — this is the only recovery path.
                ch = notary_contradiction(seat, mine, floor=SEAT_FLOOR)
                key, streak = self._contra.get(pid, (None, 0))
                if ch is not None and ch.key == key:
                    streak += 1
                elif ch is not None:
                    key, streak = ch.key, 1
                else:
                    key, streak = None, 0
                if ch is not None and streak >= self.SWAP_HEAL_ROUNDS:
                    seat = heal_seat(ch, now=now)
                    key, streak = None, 0
                if key is None and streak == 0:
                    self._contra.pop(pid, None)
                else:
                    self._contra[pid] = (key, streak)
                if seat is not None:
                    seats[pid] = seat
            # One body is one person — a track that won two residents'
            # seats keeps only the stronger claim.
            self.seats = resolve_track_conflicts(seats)
            # Swap-heal streaks belong to a seat. A resident who lost
            # theirs (unseated here, or their track died) must not carry
            # a half-built streak into their next lock.
            self._contra = {
                pid: v for pid, v in self._contra.items() if pid in self.seats
            }
            new = dict(self.seats)
        self._journal_transitions(old, new, now)

    def _journal_transitions(
        self, old: dict[UUID, Seat], new: dict[UUID, Seat], now: float,
    ) -> None:
        """Journal every resident whose seat changed this fold."""
        if self.journal is None:
            return
        for pid in set(old) | set(new):
            self._journal_transition(pid, old.get(pid), new.get(pid), now)

    def _journal_transition(
        self, pid: UUID, old: Optional[Seat], new: Optional[Seat], now: float,
    ) -> None:
        """Journal one resident's seat change with a snapshot (outside
        seat_lock — JPEG writes must never sit inside the hot mutex)."""
        if self.journal is None or old == new:
            return
        from identity.journal import annotate

        who = str(pid)[:8]

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
                    f"T{new.track_id} {who} {new.score:.2f}@{new.anchor}",
                    kind, context_boxes=ctx,
                )
            self.journal.log(
                kind, cam=new.camera_id,
                cam_name=self.name_of(new.camera_id), tid=new.track_id,
                score=round(new.score, 2), anchor=new.anchor,
                patient=who,
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
                    f"T{old.track_id} {who} UNSET", "seat_died",
                    context_boxes=ctx,
                )
            self.journal.log(
                "seat_died", cam=old.camera_id,
                cam_name=self.name_of(old.camera_id), tid=old.track_id,
                patient=who, frame_bgr=frame,
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
            old = dict(self.seats)
            live = self.live_keys()
            boxes = self.home_seat_boxes()
            kept: dict[UUID, Seat] = {}
            for pid, seat in self.seats.items():
                carried = carry_seat(
                    seat, live, boxes,
                    taken=frozenset(
                        t.key for q, t in self.seats.items() if q != pid
                    ),
                )
                # SAY SO. This path had no logging at all, which is why a
                # carry that could never succeed looked exactly like a
                # carry that was never needed — for as long as the
                # function has existed.
                if carried is not None and carried.track_id != seat.track_id:
                    log.info(
                        "identity.seat_carried pid=%s %s -> %s (reap; same "
                        "person, new track number)",
                        str(pid)[:8], seat.track_id, carried.track_id,
                    )
                elif carried is None and seat.key not in live:
                    log.info(
                        "identity.seat_lost pid=%s track=%s cam=%s — died "
                        "with no single overlapping successor",
                        str(pid)[:8], seat.track_id, seat.camera_id,
                    )
                s = update_seat(
                    carried, [], live=live,
                    floor=SEAT_FLOOR, delta=SEAT_DELTA, now=now,
                )
                if s is not None:
                    kept[pid] = s
            self.seats = kept
            new = dict(self.seats)
        self._journal_transitions(old, new, now)
