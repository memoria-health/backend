"""Body Re-ID pipeline — the per-frame engine shared by `inference.py`
(per-tick identification) and `clip_annotate.py` (per-archived-clip
bbox burn-in).

Wraps the experiment's identifier flow:

  YOLO -> BoT-SORT (with OSNet body Re-ID) -> periodic face check
       -> state machine -> locked PATIENT track + body bbox

Designed to be fed frames in temporal order. `process_frame()` returns
the current body bbox of any locked-PATIENT track (or None). Both
caller types want this — inference.py just keeps the last bbox seen,
clip_annotate.py wants the bbox at every frame so the box visibly
tracks the patient through the clip.

Per-pipeline state (the tracker, the per-track lock state) is
intentionally session-scoped: one instance per tick / per clip.
Build a fresh one each time. Cross-clip continuity is deferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.services.reid import body_detector, embeddings as reid_embeddings
from src.services.reid.tracker import PersonTracker

log = logging.getLogger(__name__)


LABEL_PATIENT = "PATIENT"
LABEL_OTHER = "OTHER"
LABEL_UNKNOWN = "UNKNOWN"


@dataclass
class PipelineConfig:
    """Tunables. All defaults match the experiment, and the
    production caller pulls them from `Settings`."""
    sample_fps: int = 5
    yolo_model: str = "yolo11s.pt"
    yolo_confidence: float = 0.4
    face_check_interval: int = 5
    hits_to_lock: int = 2
    face_min_det_score: float = 0.5
    face_checks_for_other: int = 5
    match_threshold_base: float = 0.40
    insightface_model: str = "buffalo_l"
    # Identity-persistence W6 (2026-07-19): YOLO detects down to the
    # floor so BoT-SORT's second association pass can CONTINUE an
    # existing track on weak (seated/lying) detections; `yolo_confidence`
    # becomes the first-pass bar and `track_birth_confidence` the
    # track-BIRTH bar — weak boxes never create tracks. Defaults are
    # None = PRESERVE pre-W6 behavior (detector at yolo_confidence,
    # boxmot's stock 0.6 birth) so untouched callers (clip_annotate's
    # caregiver-facing archive annotation) don't silently shift;
    # inference.py opts in explicitly from Settings (review F11).
    yolo_detect_floor: float | None = None
    track_birth_confidence: float | None = None
    # Identity-persistence W5: dim faces VOTE, only quality faces LOCK.
    # A match below these floors still counts a hit (which also keeps
    # blocking the OTHER lock — a maybe-patient must never be branded
    # not-patient) but cannot flip the label to PATIENT. Kills the
    # lunch green-box class: 15-30px guest faces passing the
    # quality-ADAPTIVE match bar. Height in extracted-frame pixels.
    face_lock_min_px: float = 36.0
    face_lock_min_det: float = 0.6
    # Wrong-green fix (2026-07-19 late soak, Aryaman's aunt): in dim
    # light the true patient AND a blood relative both score 0.38-0.51
    # against the enrollment — the flat 0.40 bar cannot separate them,
    # so the green box alternated between the two. Locking (label flip
    # + harvest eligibility) now demands more than voting:
    #   single-person clip: score ≥ face_lock_score_single
    #   multi-person clip:  score ≥ face_lock_score_multi AND the
    #     winner clears every other track's best face score by
    #     face_lock_margin (near-ties must never lock).
    # Votes (hits, OTHER-block) stay at match_threshold_base.
    face_lock_score_single: float = 0.45
    face_lock_score_multi: float = 0.55
    face_lock_margin: float = 0.15


@dataclass
class _TrackState:
    track_id: int
    label: str = LABEL_UNKNOWN
    # Pre-label assigned from a cross-clip body gallery match the first
    # time this track was seen. None when no gallery match. The track's
    # `label` is initialised to this value so that even a clip whose
    # face_check produces no result (Path A / B every attempt) carries
    # forward the gallery's earned identity. Within-clip face match
    # (Path D → PATIENT, Path C → OTHER) still wins if it succeeds.
    pre_label: Optional[str] = None
    # Where the pre-label came from: "gallery" (body-pool match — real
    # cross-clip evidence). The TrackLedger treats only gallery
    # pre-labels as fresh binding evidence; "held" is stamped by the
    # ledger itself when a binding relabels a track.
    pre_label_source: Optional[str] = None
    confidence: float = 0.0
    face_hits: int = 0
    # Total face_check attempts on this track, regardless of whether
    # SCRFD actually returned a usable face. Kept for diagnostics.
    face_checks: int = 0
    # Number of face_check attempts where SCRFD returned a face of
    # high-enough quality (det_score >= face_min_det_score) to actually
    # embed and score against the gallery. This is the load-bearing
    # counter for the OTHER lock decision: a track is only labeled
    # OTHER after `face_checks_for_other` such verified attempts have
    # failed to match the patient gallery.
    #
    # Why this distinction matters: previously `face_checks` was the
    # gate, which conflated three very different epistemic states into
    # one OTHER label:
    #   (A) SCRFD returned no face at all — we have NO identity evidence
    #   (B) SCRFD returned a face too low-quality to embed — same, NO evidence
    #   (C) Face embedded + scored against gallery + failed match —
    #       POSITIVE evidence this isn't the patient
    # Only (C) is "verified not patient." (A) and (B) leave the track
    # UNKNOWN; without this counter, a patient with their face turned
    # away the whole clip would be mis-labeled OTHER, which is the
    # exact failure mode we ruled out at design time.
    face_seen_attempts: int = 0
    last_face_check_frame: int = -10_000
    last_match_score: float = 0.0
    # Best gallery match score seen on this track this clip (any
    # quality). Feeds the ledger's sustained small-face accumulator.
    best_match_score: float = 0.0
    # Attempts that cleared the single-person LOCK bar and the det
    # floor but failed ONLY the face-px quality floor — the
    # "identity information present, pixels insufficient" class
    # (couch-at-20ft, 2026-07-21). The ledger converts a sustained
    # run of these across ticks into a PATIENT bind.
    smallface_events: int = 0
    last_seen_bbox: Optional[tuple[float, float, float, float]] = None
    # For cross-clip body gallery harvest: remember the frame where
    # this track's body box had the highest YOLO confidence — that's
    # the best frame to extract an OSNet embedding from for the gallery.
    best_frame_idx: Optional[int] = None
    best_bbox: Optional[tuple[float, float, float, float]] = None
    best_det_score: float = 0.0
    # True when this clip's face match has reached a terminal verdict
    # for this track (PATIENT lock via face_hits, OR OTHER lock via
    # face_seen_attempts). Stops further face_check work on the track.
    # Crucially: pre-labeled tracks (label set from cross-clip gallery)
    # do NOT count as face-locked — they should keep face-checking so
    # this clip's actual face evidence can override the gallery match.
    face_locked: bool = False
    # Every (frame_idx, body bbox) this track was reported at, in
    # temporal order — the raw material for the v2 signals layer
    # (per-track motion scoping, env-motion masks, crop targets).
    # Pixel space of the frames fed to process_frame (the Re-ID
    # extracted-frame space, NOT native camera resolution). ~75 frames
    # × 4 floats per track per clip — negligible memory.
    bbox_series: list[tuple[int, tuple[float, float, float, float]]] = field(
        default_factory=list,
    )
    # Tallest usable face seen on this track (SCRFD bbox height, px in
    # extracted-frame space; det_score already above the quality gate).
    # 0.0 = no usable face this clip. Feeds the v2 `face_valid(min_px)`
    # signal — the FACE-bucket legibility floor + distance warning.
    face_px_max: float = 0.0


@dataclass(frozen=True)
class TrackObservation:
    """Consumer-facing snapshot of one track after a clip run — the
    per-track unit the v2 detection layers work in (per-track motion,
    subject filtering, crop targeting). Immutable; built once by
    `track_observations()` when the clip is done.

    `label` is LABEL_PATIENT / LABEL_OTHER / LABEL_UNKNOWN.
    `confidence` is the face-match score for PATIENT locks, 0.0
    otherwise. `pre_labeled` means the label came from the cross-clip
    body gallery rather than (only) this clip's face evidence.
    `bbox_series` coordinates are in extracted-frame pixel space —
    scale with `rapid_call.scale_bbox_to_frame` before touching native
    buffer frames.
    """
    track_id: int
    label: str
    confidence: float
    bbox_series: tuple[tuple[int, tuple[float, float, float, float]], ...]
    face_locked: bool
    pre_labeled: bool
    # Tallest usable face on this track, px in extracted-frame space
    # (0.0 = none). Scale by native_width/extracted_width before
    # comparing against the ~300 px FACE-bucket floor.
    face_px_max: float = 0.0
    # Pre-label provenance: "gallery" | "carry" | "held" | None. The
    # identity lock keys fresh-evidence decisions on this.
    pre_label_source: str | None = None
    # Ledger accumulator inputs (2026-07-21 two-way evidence):
    # verified attempts that scored AND failed to match, on a track
    # with ZERO hits this clip (pure not-patient evidence, sub-lock
    # count) — accumulates toward OTHER across ticks.
    verified_mismatches: int = 0
    # Above-lock-bar matches refused only by the face-px floor —
    # accumulates toward PATIENT across ticks.
    smallface_events: int = 0
    # Best gallery match score on this track this clip (any quality).
    best_match_score: float = 0.0


@dataclass
class FrameResult:
    """Per-frame output. `patient_bbox` is the body bbox of a locked
    PATIENT track on this frame, or None if no PATIENT is currently
    locked. `face_check_ran` tells callers whether SCRFD ran on this
    frame (useful for harvest accounting)."""
    patient_bbox: Optional[tuple[float, float, float, float]] = None
    face_check_ran: bool = False
    best_match_score_this_frame: float = 0.0
    best_face_embedding: Optional[np.ndarray] = None
    best_face_det_score: float = 0.0


class BodyReIDPipeline:
    """Stateful per-clip identifier.

    Usage:

        pipeline = BodyReIDPipeline(gallery=gallery, config=cfg)
        for frame_idx, frame_bgr in enumerate(frames_bgr):
            result = pipeline.process_frame(frame_bgr, frame_idx)
            if result.patient_bbox is not None:
                draw_box(frame_bgr, result.patient_bbox)
        summary = pipeline.summary()  # locked tracks, best score, etc.

    Perception seam (GPU port, 2026-07-20): detect+track and face
    detect+embed are isolated behind `_perceive()` / `_face_hits()`.
    The defaults below run the local YOLO + BoT-SORT + SCRFD path,
    byte-identical to the pre-seam behavior. `RemoteReplayPipeline`
    (remote_replay.py) overrides both to feed pod-computed SAM tracks
    and face observations through this SAME decision machinery — every
    label, lock, threshold, and invariant runs here either way; the
    brain never knows the eyes changed.
    """

    # Subclasses running on remote observations set this False so
    # __init__ skips constructing the local YOLO detector + BoT-SORT
    # tracker they will never call.
    LOCAL_PERCEPTION: bool = True

    def __init__(
        self,
        *,
        gallery: np.ndarray,
        config: PipelineConfig,
        pre_match_callback: Optional[
            Callable[[np.ndarray, tuple[float, float, float, float]], Optional[str]]
        ] = None,
    ) -> None:
        # Empty gallery is supported: body tracking still runs (YOLO +
        # BoT-SORT are gallery-independent) so the caller gets
        # `num_tracks`/`max_concurrent_tracks` for the presence signal,
        # but face matching is skipped (no patient identity possible).
        # This is the no-enrollment path: a freshly-paired camera can
        # still drive the open↔closed state machine from body track
        # presence alone before the caregiver enrolls a face.
        self._gallery = gallery
        self._has_gallery = gallery.size > 0
        self._cfg = config
        # Detector runs at the FLOOR (not the confidence bar) so weak
        # seated/lying detections reach the tracker's second
        # association pass. The bar/birth split lives in the tracker.
        # floor=None (legacy callers) keeps the old single-bar mode.
        self._detect_floor = (
            config.yolo_detect_floor
            if config.yolo_detect_floor is not None
            else config.yolo_confidence
        )
        if self.LOCAL_PERCEPTION:
            self._detector = body_detector.PersonDetector(
                model_name=config.yolo_model,
                confidence=self._detect_floor,
            )
            self._tracker = self._build_tracker()
        else:
            self._detector = None
            self._tracker = None
        self._tracks: dict[int, _TrackState] = {}
        # Cross-clip body gallery hook. When set, called once per track
        # the first time it's seen this clip. Signature:
        #   (frame_bgr, bbox) → Optional[LABEL_PATIENT | LABEL_OTHER]
        # Caller is responsible for OSNet inference + gallery match.
        # Returning a label seeds the track's initial state so a
        # face-hidden clip can still benefit from prior identity work.
        self._pre_match_callback = pre_match_callback
        # Track IDs we've already attempted pre-match for (avoid
        # repeating expensive OSNet inference on every frame the track
        # appears in).
        self._pre_match_attempted: set[int] = set()
        # Aggregate stats accumulate across frames so the caller can
        # build a single IdentityResult at the end of the clip.
        self._face_count = 0
        self._best_score = -1.0
        self._best_frame_idx: Optional[int] = None
        self._best_face_embedding: Optional[np.ndarray] = None
        self._best_face_det_score = 0.0
        # Peak number of concurrently-active BoT-SORT tracks observed
        # in any single frame. `num_tracks` is cumulative (every track
        # ID ever assigned), which over-counts when BoT-SORT loses and
        # re-acquires a track. For the per-tick `people_count` signal
        # we want concurrent presence, not lifetime tracks.
        self._max_concurrent_tracks = 0
        # Per-clip YOLO detection stats. Accumulated across every frame
        # so we can log a clip-level summary at the end. These exist to
        # diagnose the "Gemini saw a person but YOLO didn't" failure
        # mode: if `total_detections=0` across the whole clip and
        # Gemini later reports a person, that's a YOLO recall miss,
        # not a re-ID problem.
        self._yolo_frames_processed = 0
        self._yolo_frames_with_detections = 0
        self._yolo_total_detections = 0
        self._yolo_max_confidence = 0.0
        self._yolo_confidence_sum = 0.0

    def _build_tracker(self) -> PersonTracker:
        cfg = self._cfg
        if cfg.yolo_detect_floor is None:
            # PRESERVE pre-W6 behavior for floor-less callers
            # (clip_annotate's caregiver-facing archive annotation):
            # boxmot stock association bars, exactly as before the
            # floor/birth split. Passing yolo_confidence/floor here
            # moved [0.40, 0.50) detections from the IoU-only second
            # pass into the ReID-fused first pass and shifted ID
            # continuity in archived clips (review C9).
            return PersonTracker(
                frame_rate=max(1, cfg.sample_fps),
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.6,
            )
        return PersonTracker(
            frame_rate=max(1, cfg.sample_fps),
            track_high_thresh=cfg.yolo_confidence,
            track_low_thresh=self._detect_floor,
            # None preserves boxmot's stock 0.6 birth bar; inference.py
            # passes the Settings value.
            new_track_thresh=(
                cfg.track_birth_confidence
                if cfg.track_birth_confidence is not None else 0.6
            ),
        )

    def reset_tracker(self) -> None:
        """Rebuild BoT-SORT and clear the per-track state machine.
        Keeps the gallery + face/body models in place. Use this when
        the input stream has a known discontinuity (e.g., the concat
        boundary between the pre-event and post-event halves of an
        archived clip) so the post-discontinuity frames get their
        own clean lock instead of inheriting stale tracks."""
        if self.LOCAL_PERCEPTION:
            self._tracker = self._build_tracker()
        self._tracks = {}

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameResult:
        """Push one frame through detect → track → maybe-face-check →
        state machine. Returns the locked PATIENT bbox AT THE CURRENT
        FRAME's position (None if no PATIENT track was reported by
        BoT-SORT this frame). Mirrors the experiment's overlay flow —
        we only emit a bbox tied to a track present in the current
        update, never a stale last_seen position from frames ago."""
        cfg = self._cfg
        tracked = self._perceive(frame_bgr, frame_idx)

        frame_result = FrameResult()

        # Collect tids reported by BoT-SORT this frame so the resolve
        # step below only considers CURRENTLY-active tracks (not stale
        # entries in self._tracks left over from past frames).
        seen_this_frame: dict[int, tuple[float, float, float, float]] = {}
        for row in tracked:
            x1, y1, x2, y2, tid_f, det_conf, _cls, _det_ind = row.tolist()
            tid = int(tid_f)
            state = self._tracks.setdefault(tid, _TrackState(track_id=tid))
            bbox = (float(x1), float(y1), float(x2), float(y2))
            state.last_seen_bbox = bbox
            state.bbox_series.append((frame_idx, bbox))
            seen_this_frame[tid] = bbox

            # Per-track best-frame tracking for the cross-clip body
            # gallery harvest. Pick the frame where YOLO was most
            # confident — gives the cleanest body crop to embed.
            conf = float(det_conf)
            if conf > state.best_det_score:
                state.best_det_score = conf
                state.best_frame_idx = frame_idx
                state.best_bbox = bbox

            # Cross-clip body gallery pre-match. Runs once per track the
            # first time it's seen. If the callback returns a label, the
            # track starts pre-labeled (instead of UNKNOWN) so future
            # face dropout in this clip doesn't lose the identity. Face
            # match within this clip still overrides (Path D → PATIENT
            # wins; Path C only re-locks OTHER if pre-label was also
            # OTHER or UNKNOWN, never overrides a PATIENT pre-label).
            if (
                self._pre_match_callback is not None
                and tid not in self._pre_match_attempted
            ):
                self._pre_match_attempted.add(tid)
                try:
                    pre_label = self._pre_match_callback(frame_bgr, bbox)
                except Exception:
                    log.exception("body_pipeline.pre_match_callback raised")
                    pre_label = None
                # Callback may return a bare label (legacy: gallery
                # match) or (label, source) with source "gallery".
                pre_source = "gallery"
                if isinstance(pre_label, tuple):
                    pre_label, pre_source = pre_label[0], pre_label[1]
                if pre_label == LABEL_PATIENT and any(
                    s is not state and s.label == LABEL_PATIENT
                    for s in self._tracks.values()
                ):
                    # Single-PATIENT invariant (W5): a body-gallery
                    # pre-label never seats a SECOND patient — the
                    # earlier evidence keeps the label; this track
                    # stays UNKNOWN and face checks can still speak.
                    log.info(
                        "reid.pre_match_skipped_second_patient tid=%d",
                        tid,
                    )
                elif pre_label in (LABEL_PATIENT, LABEL_OTHER):
                    state.pre_label = pre_label
                    state.pre_label_source = pre_source
                    state.label = pre_label

            # Face check runs while we still have something to learn:
            #   - There's a gallery to match against, AND
            #   - This track's identity hasn't been face-locked in this
            #     clip yet (pre-labeled tracks ARE still checked so a
            #     stale gallery entry can be overridden by fresh face
            #     evidence), AND
            #   - The interval has elapsed since the last check.
            #
            # Without a gallery, body tracking still drives presence but
            # face matching is impossible.
            if (
                self._has_gallery
                and not state.face_locked
                and frame_idx - state.last_face_check_frame >= cfg.face_check_interval
            ):
                state.last_face_check_frame = frame_idx
                state.face_checks += 1
                self._face_check(state, frame_bgr, bbox, frame_idx, frame_result)

        # Peak-concurrent track count (this frame's active track set
        # size vs. the running max).
        if len(seen_this_frame) > self._max_concurrent_tracks:
            self._max_concurrent_tracks = len(seen_this_frame)

        # Emit a PATIENT bbox only for a track active this frame.
        # If BoT-SORT lost the patient track (motion / occlusion),
        # the bbox is None for this frame rather than a frozen
        # position from earlier in the clip.
        for tid, bbox in seen_this_frame.items():
            state = self._tracks.get(tid)
            if state is not None and state.label == LABEL_PATIENT:
                frame_result.patient_bbox = bbox
                break

        return frame_result

    # ── perception seam (overridden by RemoteReplayPipeline) ──────

    def _perceive(self, frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
        """Detect + track one frame. Returns BoT-SORT-shaped rows
        (x1, y1, x2, y2, tid, det_conf, cls, det_ind). The default is
        the local YOLO + BoT-SORT path; per-frame detection stats
        accumulate here (col 4 = confidence) so `yolo_stats` stays
        truthful for whichever perception ran."""
        dets = self._detector.detect(frame_bgr)
        n_dets = int(dets.shape[0])
        self._yolo_frames_processed += 1
        self._yolo_total_detections += n_dets
        if n_dets > 0:
            self._yolo_frames_with_detections += 1
            frame_confs = dets[:, 4].astype(float)
            frame_max = float(frame_confs.max())
            if frame_max > self._yolo_max_confidence:
                self._yolo_max_confidence = frame_max
            self._yolo_confidence_sum += float(frame_confs.sum())
        return self._tracker.update(dets, frame_bgr)

    def _face_hits(
        self,
        state: _TrackState,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
    ) -> list["reid_embeddings.FaceHit"]:
        """Detect + embed faces inside this track's body bbox. Default
        = local SCRFD + ArcFace via `embed_in_box`. RemoteReplayPipeline
        overrides with a lookup of pod-computed observations for
        (track, frame). Returned bboxes are in frame coords."""
        return reid_embeddings.embed_in_box(
            frame_bgr, bbox, model_name=self._cfg.insightface_model,
        )

    def _face_check(
        self,
        state: _TrackState,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
        frame_result: FrameResult,
    ) -> None:
        cfg = self._cfg
        hits = self._face_hits(state, frame_bgr, bbox, frame_idx)
        # Path A: SCRFD didn't detect any face in this body crop. We
        # have NO identity evidence on this attempt. Track stays at
        # whatever label it had (UNKNOWN unless a prior attempt landed
        # something). Do NOT lock OTHER — that would conflate "no face
        # seen" with "face seen and failed match," and patient with
        # face turned away the whole clip would be mis-labeled.
        if not hits:
            return

        self._face_count += 1
        frame_result.face_check_ran = True
        best = max(hits, key=lambda f: f.det_score)
        # Path B: face found but too dim/blurry/small to extract a
        # trustworthy embedding. Same reasoning as Path A — we never
        # actually attempted to match, so this attempt contributes no
        # identity evidence. Track stays at its current label.
        if best.det_score < cfg.face_min_det_score:
            return

        # SAM-face countersign gate (B3 incident, 2026-07-21): a claim
        # the face CONCEPT session refuted (SCRFD firing on the back of
        # a head — those scored ~0.07 vs the gallery and branded the
        # patient OTHER) is counterfeit evidence. Epistemically Path B:
        # no vote, no OTHER testimony, no best-score pollution. The
        # original det bar below keeps full guest-branding recall for
        # ORACLE-CONFIRMED faces; unconfirmed ones are silence.
        if not getattr(best, "oracle_face", True):
            return

        # We have a high-enough-quality, oracle-confirmed face to embed
        # and compare. From this point on, the attempt counts as a
        # "verified attempt" — used by `_maybe_lock_other` to gate the
        # OTHER decision so it only fires after several genuine match
        # failures.
        state.face_seen_attempts += 1
        state.face_px_max = max(
            state.face_px_max, float(best.bbox[3] - best.bbox[1]),
        )

        match = reid_embeddings.topk_mean_cosine_match(best.embedding, self._gallery)
        threshold = reid_embeddings.effective_match_threshold(
            cfg.match_threshold_base, best.det_score, best.face_pixel_area,
        )
        state.last_match_score = match.score
        if match.score > state.best_match_score:
            state.best_match_score = match.score
        frame_result.best_match_score_this_frame = match.score
        frame_result.best_face_embedding = best.embedding
        frame_result.best_face_det_score = best.det_score

        if match.score > self._best_score:
            self._best_score = match.score
            self._best_frame_idx = frame_idx
            self._best_face_embedding = best.embedding
            self._best_face_det_score = best.det_score

        # Path D: match succeeded above threshold. Accumulate hits;
        # lock as PATIENT after enough — but only on a QUALITY face
        # (W5): sub-floor matches vote (and keep blocking the OTHER
        # lock via face_hits > 0) yet can never flip the label. The
        # floors exist because tiny/blurred faces produce embeddings
        # that collide across people — the lunch green-box incident.
        if match.score >= threshold:
            state.face_hits += 1
            face_h = float(best.bbox[3] - best.bbox[1])
            lock_quality = (
                face_h >= cfg.face_lock_min_px
                and best.det_score >= cfg.face_lock_min_det
            )
            if (
                not lock_quality
                and match.score >= cfg.face_lock_score_single
                and best.det_score >= cfg.face_lock_min_det
                and face_h < cfg.face_lock_min_px
            ):
                # Above the LOCK bar on a confident detection, refused
                # only by pixels: a sustained-evidence event.
                state.smallface_events += 1
            if state.face_hits >= cfg.hits_to_lock and lock_quality:
                self._lock_patient(state, match.score, frame_idx)
            elif state.face_hits >= cfg.hits_to_lock:
                log.info(
                    "reid.lock_deferred_quality frame=%d tid=%d score=%.3f "
                    "face_px=%.0f det=%.2f (floors %.0fpx/%.2f)",
                    frame_idx, state.track_id, match.score, face_h,
                    best.det_score, cfg.face_lock_min_px,
                    cfg.face_lock_min_det,
                )
            return

        # Path C: face embedded + scored + match failed. This is
        # positive disambiguation; counts toward the OTHER lock.
        self._maybe_lock_other(state)

    def _lock_patient(
        self, state: _TrackState, score: float, frame_idx: int,
    ) -> None:
        """Face-lock `state` as PATIENT, enforcing the in-clip
        SINGLE-PATIENT invariant (identity-persistence W5, Aryaman:
        one patient per home — a fortiori one per clip): if another
        track already holds PATIENT, the higher face score keeps it
        and the loser reverts to UNKNOWN (never OTHER) with its face
        checking re-opened so fresh evidence can still speak."""
        cfg = self._cfg
        multi = self._max_concurrent_tracks >= 2
        bar = cfg.face_lock_score_multi if multi else cfg.face_lock_score_single
        if score < bar:
            log.info(
                "reid.lock_deferred_score frame=%d tid=%d score=%.3f "
                "< bar %.2f (%s-person clip) — voting only",
                frame_idx, state.track_id, score, bar,
                "multi" if multi else "single",
            )
            return
        if multi:
            rival = max(
                (
                    float(t.last_match_score)
                    for t in self._tracks.values()
                    if t is not state
                ),
                default=0.0,
            )
            if score - rival < cfg.face_lock_margin:
                # Near-tie between two faces (the aunt case): a lock
                # here is a coin flip on who gets the green box.
                # Neither locks; both keep voting.
                log.info(
                    "reid.lock_deferred_margin frame=%d tid=%d "
                    "score=%.3f rival=%.3f margin=%.3f < %.2f",
                    frame_idx, state.track_id, score, rival,
                    score - rival, cfg.face_lock_margin,
                )
                return
        for other in self._tracks.values():
            if other is state or other.label != LABEL_PATIENT:
                continue
            if float(other.confidence) >= score:
                log.info(
                    "reid.lock_conflict_kept_existing tid=%d(%.3f) over "
                    "tid=%d(%.3f)",
                    other.track_id, other.confidence,
                    state.track_id, score,
                )
                return
            other.label = LABEL_UNKNOWN
            other.face_locked = False
            log.info(
                "reid.lock_conflict_demoted tid=%d(%.3f) to UNKNOWN — "
                "tid=%d(%.3f) wins",
                other.track_id, other.confidence, state.track_id, score,
            )
        state.label = LABEL_PATIENT
        state.confidence = score
        # Face evidence has overridden whatever pre-label the gallery
        # match may have set. Stop further face checks on this track
        # for the rest of the clip.
        state.face_locked = True
        log.info(
            "reid.track_locked frame=%d tid=%d score=%.3f hits=%d",
            frame_idx, state.track_id, score, state.face_hits,
        )

    def _maybe_lock_other(self, state: _TrackState) -> None:
        cfg = self._cfg
        # Gate now uses `face_seen_attempts` (verified attempts that
        # actually embedded + matched) instead of `face_checks` (raw
        # attempts including no-face and low-quality cases). This is
        # what keeps a face-turned-away patient from being silently
        # mis-labeled OTHER.
        #
        # The eligibility check is `state.label in {UNKNOWN, OTHER}`
        # because a pre-labeled OTHER track from the cross-clip gallery
        # is allowed to graduate to a face-locked OTHER (same label,
        # just stronger evidence) — but a PATIENT label (whether from
        # face or pre-match) is NEVER demoted to OTHER here. Demotion
        # would require positive evidence FROM the face match — and
        # that path goes through Path D's hits accumulator, not here.
        if (
            state.label in (LABEL_UNKNOWN, LABEL_OTHER)
            and state.face_hits == 0
            and state.face_seen_attempts >= cfg.face_checks_for_other
        ):
            state.label = LABEL_OTHER
            # OTHER lock from face evidence is terminal for this clip
            # too — same reasoning as PATIENT lock.
            state.face_locked = True

    # ── Accessors for the per-tick summary ────────────────────────

    @property
    def face_count(self) -> int:
        return self._face_count

    @property
    def best_score(self) -> float:
        return max(0.0, self._best_score)

    @property
    def best_frame_idx(self) -> Optional[int]:
        return self._best_frame_idx

    @property
    def best_face_embedding(self) -> Optional[np.ndarray]:
        return self._best_face_embedding

    @property
    def best_face_det_score(self) -> float:
        return self._best_face_det_score

    @property
    def has_any_patient_track(self) -> bool:
        return any(s.label == LABEL_PATIENT for s in self._tracks.values())

    @property
    def latest_patient_bbox(self) -> Optional[tuple[float, float, float, float]]:
        for state in self._tracks.values():
            if state.label == LABEL_PATIENT and state.last_seen_bbox is not None:
                return state.last_seen_bbox
        return None

    @property
    def num_tracks(self) -> int:
        return len(self._tracks)

    @property
    def max_concurrent_tracks(self) -> int:
        """Peak count of simultaneously-active BoT-SORT tracks observed
        in any single frame this clip. The right signal for "how many
        people were in this clip" — `num_tracks` over-counts when
        BoT-SORT churns track IDs (loses a person then re-acquires).
        """
        return self._max_concurrent_tracks

    @property
    def yolo_stats(self) -> dict[str, float | int]:
        """Per-clip YOLO detection telemetry. Returned to the caller so
        it can be logged at the same point as `reid.label_clip`.

        Fields:
          - frames_processed: total frames YOLO ran on this clip
          - frames_with_detections: frames where YOLO returned >= 1 box
          - zero_detection_frame_pct: frames-with-zero / frames_processed
          - total_detections: cumulative detection count
          - max_confidence: highest YOLO confidence observed any frame
          - mean_confidence: mean confidence across all detections
            (0.0 if total_detections == 0)
          - confidence_threshold: the threshold YOLO was run with
            (loops back the config value so log readers can correlate)

        Useful for the "Gemini saw a person, YOLO didn't" diagnosis:
        frames_with_detections=0 + max_confidence=0.0 means YOLO never
        saw anything; a Gemini call on that clip is YOLO recall miss,
        not a re-ID issue. Conversely high max_confidence with low
        mean tells us a single confident detection but lots of marginal
        boxes — useful when tuning the threshold down.
        """
        mean_conf = (
            self._yolo_confidence_sum / self._yolo_total_detections
            if self._yolo_total_detections > 0 else 0.0
        )
        zero_pct = (
            1.0 - (self._yolo_frames_with_detections / self._yolo_frames_processed)
            if self._yolo_frames_processed > 0 else 0.0
        )
        return {
            "frames_processed": self._yolo_frames_processed,
            "frames_with_detections": self._yolo_frames_with_detections,
            "zero_detection_frame_pct": round(zero_pct, 3),
            "total_detections": self._yolo_total_detections,
            "max_confidence": round(self._yolo_max_confidence, 3),
            "mean_confidence": round(mean_conf, 3),
            # The bar the detector actually RAN at (the W6 floor when
            # enabled) plus the association/birth bars, so soak
            # telemetry stays honest about what 0-detection means.
            "confidence_threshold": self._detect_floor,
            "association_threshold": self._cfg.yolo_confidence,
        }

    @property
    def has_only_verified_other_tracks(self) -> bool:
        """True iff num_tracks > 0 AND every track in the clip ended
        labeled OTHER (verified non-patient). Equivalent to:
            no PATIENT track AND no UNKNOWN track AND at least one OTHER track.

        Used by the camera_ingest identity gate to SKIP the LLM call
        when we have positive evidence that everyone in frame is not
        the patient. OTHER is earned via face match failure on at
        least `face_checks_for_other` verified attempts — OR inherited
        from a cross-clip body-gallery pre-label (which itself
        originated in some earlier clip's face evidence). The pre-label
        path means this CAN be true with zero in-clip face evidence
        (face_count=0); trusting it is a deliberate cross-clip-identity
        decision, and the gate additionally requires attribution_ran so
        a tick that couldn't attribute never suppresses.
        """
        if not self._tracks:
            return False
        return all(s.label == LABEL_OTHER for s in self._tracks.values())

    def track_observations(self) -> list[TrackObservation]:
        """Immutable per-track summaries for the clip, sorted by
        track_id. Includes every track BoT-SORT ever assigned this
        clip — callers that only care about substantial tracks can
        filter on len(bbox_series). Empty list when no tracks."""
        return [
            TrackObservation(
                track_id=s.track_id,
                label=s.label,
                confidence=float(s.confidence),
                bbox_series=tuple(s.bbox_series),
                face_locked=s.face_locked,
                pre_labeled=s.pre_label is not None,
                face_px_max=float(s.face_px_max),
                pre_label_source=s.pre_label_source,
                verified_mismatches=(
                    s.face_seen_attempts if s.face_hits == 0 else 0
                ),
                smallface_events=s.smallface_events,
                best_match_score=float(s.best_match_score),
            )
            for s in sorted(self._tracks.values(), key=lambda s: s.track_id)
        ]
