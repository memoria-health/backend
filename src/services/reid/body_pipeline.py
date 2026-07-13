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
    """

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
        self._detector = body_detector.PersonDetector(
            model_name=config.yolo_model,
            confidence=config.yolo_confidence,
        )
        self._tracker = PersonTracker(frame_rate=max(1, config.sample_fps))
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

    def reset_tracker(self) -> None:
        """Rebuild BoT-SORT and clear the per-track state machine.
        Keeps the gallery + face/body models in place. Use this when
        the input stream has a known discontinuity (e.g., the concat
        boundary between the pre-event and post-event halves of an
        archived clip) so the post-discontinuity frames get their
        own clean lock instead of inheriting stale tracks."""
        self._tracker = PersonTracker(frame_rate=max(1, self._cfg.sample_fps))
        self._tracks = {}

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> FrameResult:
        """Push one frame through detect → track → maybe-face-check →
        state machine. Returns the locked PATIENT bbox AT THE CURRENT
        FRAME's position (None if no PATIENT track was reported by
        BoT-SORT this frame). Mirrors the experiment's overlay flow —
        we only emit a bbox tied to a track present in the current
        update, never a stale last_seen position from frames ago."""
        cfg = self._cfg
        dets = self._detector.detect(frame_bgr)
        tracked = self._tracker.update(dets, frame_bgr)

        # Accumulate per-frame YOLO stats. `dets` is Nx6 where col 4 is
        # confidence. Tracked here rather than inside _detector so we
        # don't have to plumb a callback through PersonDetector.
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
                if pre_label in (LABEL_PATIENT, LABEL_OTHER):
                    state.pre_label = pre_label
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

    def _face_check(
        self,
        state: _TrackState,
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        frame_idx: int,
        frame_result: FrameResult,
    ) -> None:
        cfg = self._cfg
        hits = reid_embeddings.embed_in_box(
            frame_bgr, bbox, model_name=cfg.insightface_model,
        )
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

        # We have a high-enough-quality face to embed and compare. From
        # this point on, the attempt counts as a "verified attempt" —
        # used by `_maybe_lock_other` to gate the OTHER decision so it
        # only fires after several genuine match failures.
        state.face_seen_attempts += 1
        state.face_px_max = max(
            state.face_px_max, float(best.bbox[3] - best.bbox[1]),
        )

        match = reid_embeddings.topk_mean_cosine_match(best.embedding, self._gallery)
        threshold = reid_embeddings.effective_match_threshold(
            cfg.match_threshold_base, best.det_score, best.face_pixel_area,
        )
        state.last_match_score = match.score
        frame_result.best_match_score_this_frame = match.score
        frame_result.best_face_embedding = best.embedding
        frame_result.best_face_det_score = best.det_score

        if match.score > self._best_score:
            self._best_score = match.score
            self._best_frame_idx = frame_idx
            self._best_face_embedding = best.embedding
            self._best_face_det_score = best.det_score

        # Path D: match succeeded above threshold. Accumulate hits;
        # lock as PATIENT after enough.
        if match.score >= threshold:
            state.face_hits += 1
            if state.face_hits >= cfg.hits_to_lock:
                state.label = LABEL_PATIENT
                state.confidence = match.score
                # Face evidence has overridden whatever pre-label the
                # gallery match may have set. Stop further face checks
                # on this track for the rest of the clip.
                state.face_locked = True
                log.info(
                    "reid.track_locked frame=%d tid=%d score=%.3f hits=%d",
                    frame_idx, state.track_id, match.score, state.face_hits,
                )
            return

        # Path C: face embedded + scored + match failed. This is
        # positive disambiguation; counts toward the OTHER lock.
        self._maybe_lock_other(state)

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
            "confidence_threshold": self._cfg.yolo_confidence,
        }

    @property
    def has_only_verified_other_tracks(self) -> bool:
        """True iff num_tracks > 0 AND every track in the clip ended
        labeled OTHER (verified non-patient). Equivalent to:
            no PATIENT track AND no UNKNOWN track AND at least one OTHER track.

        Used by the camera_ingest identity gate to SKIP the LLM call
        when we have positive evidence that everyone in frame is not
        the patient. Safe because OTHER under the new label semantic
        is earned only via face match failure on at least
        `face_checks_for_other` verified attempts — never speculative.
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
            )
            for s in sorted(self._tracks.values(), key=lambda s: s.track_id)
        ]
