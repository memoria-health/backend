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
from typing import Optional

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
    confidence: float = 0.0
    face_hits: int = 0
    face_checks: int = 0
    last_face_check_frame: int = -10_000
    last_match_score: float = 0.0
    last_seen_bbox: Optional[tuple[float, float, float, float]] = None


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
    ) -> None:
        if gallery.size == 0:
            raise ValueError("body_pipeline: gallery must be non-empty")
        self._gallery = gallery
        self._cfg = config
        self._detector = body_detector.PersonDetector(confidence=config.yolo_confidence)
        self._tracker = PersonTracker(frame_rate=max(1, config.sample_fps))
        self._tracks: dict[int, _TrackState] = {}
        # Aggregate stats accumulate across frames so the caller can
        # build a single IdentityResult at the end of the clip.
        self._face_count = 0
        self._best_score = -1.0
        self._best_frame_idx: Optional[int] = None
        self._best_face_embedding: Optional[np.ndarray] = None
        self._best_face_det_score = 0.0

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

        frame_result = FrameResult()

        # Collect tids reported by BoT-SORT this frame so the resolve
        # step below only considers CURRENTLY-active tracks (not stale
        # entries in self._tracks left over from past frames).
        seen_this_frame: dict[int, tuple[float, float, float, float]] = {}
        for row in tracked:
            x1, y1, x2, y2, tid_f, _conf, _cls, _det_ind = row.tolist()
            tid = int(tid_f)
            state = self._tracks.setdefault(tid, _TrackState(track_id=tid))
            bbox = (float(x1), float(y1), float(x2), float(y2))
            state.last_seen_bbox = bbox
            seen_this_frame[tid] = bbox

            if (
                state.label == LABEL_UNKNOWN
                and frame_idx - state.last_face_check_frame >= cfg.face_check_interval
            ):
                state.last_face_check_frame = frame_idx
                state.face_checks += 1
                self._face_check(state, frame_bgr, bbox, frame_idx, frame_result)

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
        if not hits:
            self._maybe_lock_other(state)
            return

        self._face_count += 1
        frame_result.face_check_ran = True
        best = max(hits, key=lambda f: f.det_score)
        if best.det_score < cfg.face_min_det_score:
            self._maybe_lock_other(state)
            return

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

        if match.score >= threshold:
            state.face_hits += 1
            if state.face_hits >= cfg.hits_to_lock:
                state.label = LABEL_PATIENT
                state.confidence = match.score
                log.info(
                    "reid.track_locked frame=%d tid=%d score=%.3f hits=%d",
                    frame_idx, state.track_id, match.score, state.face_hits,
                )
            return

        self._maybe_lock_other(state)

    def _maybe_lock_other(self, state: _TrackState) -> None:
        cfg = self._cfg
        if (
            state.label == LABEL_UNKNOWN
            and state.face_hits == 0
            and state.face_checks >= cfg.face_checks_for_other
        ):
            state.label = LABEL_OTHER

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
