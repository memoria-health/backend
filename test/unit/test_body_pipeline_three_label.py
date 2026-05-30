"""Phase 1 unit tests for the three-label semantic in `BodyReIDPipeline`.

The pre-refactor `_maybe_lock_other` conflated three different
epistemic states into one OTHER label:

  - Path A: SCRFD returned no face → no identity evidence
  - Path B: SCRFD returned face too low-quality to embed → no evidence
  - Path C: face embedded + matched against gallery + failed → VERIFIED not-patient

Old code locked OTHER after `face_checks_for_other` raw attempts,
regardless of path. New code locks OTHER only after
`face_checks_for_other` *verified* attempts (path C), i.e. when
`face_seen_attempts >= face_checks_for_other`.

These tests pin that semantic by driving `_face_check` directly with
stubbed embed_in_box / gallery responses, then asserting the resulting
track labels.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from src.services.reid.body_pipeline import (
    LABEL_OTHER,
    LABEL_PATIENT,
    LABEL_UNKNOWN,
    BodyReIDPipeline,
    FrameResult,
    PipelineConfig,
    _TrackState,
)
from src.services.reid.embeddings import FaceHit, MatchResult


# ─── Test scaffolding ────────────────────────────────────────────


def _make_pipeline(
    *,
    gallery: np.ndarray | None = None,
    face_checks_for_other: int = 5,
    hits_to_lock: int = 2,
    face_min_det_score: float = 0.5,
) -> BodyReIDPipeline:
    """Construct a pipeline with heavy submodels stubbed.

    We only need the pipeline's per-track state machine + the
    `_face_check` method. The PersonDetector + PersonTracker would load
    YOLO + OSNet weights respectively; both are bypassed since these
    tests drive `_face_check` directly.
    """
    if gallery is None:
        # 1 gallery entry — non-empty, otherwise the pipeline takes its
        # body-tracking-only path which doesn't run face checks at all.
        gallery = np.ones((1, 512), dtype=np.float32) / np.sqrt(512)

    with (
        patch("src.services.reid.body_pipeline.body_detector.PersonDetector"),
        patch("src.services.reid.body_pipeline.PersonTracker"),
    ):
        return BodyReIDPipeline(
            gallery=gallery,
            config=PipelineConfig(
                sample_fps=5,
                face_check_interval=5,
                face_min_det_score=face_min_det_score,
                face_checks_for_other=face_checks_for_other,
                hits_to_lock=hits_to_lock,
                match_threshold_base=0.40,
            ),
        )


def _hit(det_score: float, embedding: np.ndarray | None = None) -> FaceHit:
    """Build a synthetic FaceHit. `embedding` defaults to a unit vector
    orthogonal to the canonical gallery direction so the cosine match
    score is ~0 (well below any threshold) unless overridden.
    """
    if embedding is None:
        embedding = np.zeros(512, dtype=np.float32)
        embedding[7] = 1.0   # orthogonal to the all-ones gallery
    return FaceHit(
        embedding=embedding,
        det_score=det_score,
        bbox=(0.0, 0.0, 100.0, 100.0),
    )


def _run_face_check(
    pipeline: BodyReIDPipeline,
    state: _TrackState,
    *,
    hits: list[FaceHit] | None,
    match_score: float = 0.0,
    frame_idx: int = 0,
) -> FrameResult:
    """Drive one `_face_check` call with a stubbed embed + match."""
    frame_result = FrameResult()
    embed_return = hits if hits is not None else []
    match_return = MatchResult(score=match_score, top_k_used=1)
    bbox = (0.0, 0.0, 100.0, 100.0)
    fake_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    with (
        patch(
            "src.services.reid.body_pipeline.reid_embeddings.embed_in_box",
            return_value=embed_return,
        ),
        patch(
            "src.services.reid.body_pipeline.reid_embeddings.topk_mean_cosine_match",
            return_value=match_return,
        ),
    ):
        # face_checks increments in process_frame, not in _face_check —
        # mirror that here so the test is faithful to the production path.
        state.face_checks += 1
        pipeline._face_check(state, fake_frame, bbox, frame_idx, frame_result)
    return frame_result


# ─── Path A: SCRFD returned no face ──────────────────────────────


def test_path_a_no_face_never_locks_other_even_after_many_attempts() -> None:
    """The whole point of the new semantic: a track whose face was
    never detected by SCRFD must stay UNKNOWN, no matter how many
    attempts. Mirrors the patient-face-turned-away scenario."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    for i in range(20):
        _run_face_check(pipeline, state, hits=None, frame_idx=i * 5)
    assert state.label == LABEL_UNKNOWN
    assert state.face_seen_attempts == 0
    assert state.face_checks == 20   # raw attempt counter still ticks


# ─── Path B: face too low quality ────────────────────────────────


def test_path_b_low_quality_face_never_locks_other() -> None:
    """SCRFD returns a face but `det_score < face_min_det_score` means
    we never even embed it. Same epistemic state as Path A — no
    identity evidence — so we must not lock OTHER."""
    pipeline = _make_pipeline(face_checks_for_other=5, face_min_det_score=0.5)
    state = _TrackState(track_id=1)
    low_quality = [_hit(det_score=0.3)]
    for i in range(20):
        _run_face_check(pipeline, state, hits=low_quality, frame_idx=i * 5)
    assert state.label == LABEL_UNKNOWN
    assert state.face_seen_attempts == 0


# ─── Path C: face embedded + matched + failed ────────────────────


def test_path_c_face_failed_match_locks_other_after_threshold() -> None:
    """Face was detected, embedded, scored against gallery, and didn't
    meet the match threshold. Positive disambiguation: this is NOT the
    patient. After `face_checks_for_other` such attempts, lock OTHER."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    good_face_no_match = [_hit(det_score=0.9)]  # quality OK; match=0 by default
    for i in range(5):
        _run_face_check(
            pipeline, state, hits=good_face_no_match, match_score=0.0, frame_idx=i * 5,
        )
    assert state.label == LABEL_OTHER
    assert state.face_seen_attempts == 5


def test_path_c_locks_other_at_exactly_the_threshold() -> None:
    """Boundary: lock fires the moment face_seen_attempts hits
    face_checks_for_other, not one attempt later."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    good_face_no_match = [_hit(det_score=0.9)]
    for i in range(4):
        _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=i * 5)
        assert state.label == LABEL_UNKNOWN
    _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=20)
    assert state.label == LABEL_OTHER


# ─── Path D: match succeeds ──────────────────────────────────────


def test_path_d_match_above_threshold_locks_patient_after_hits_to_lock() -> None:
    """A track that face-matches at high confidence enough times locks
    as PATIENT. PATIENT precedence is preserved."""
    pipeline = _make_pipeline(hits_to_lock=2)
    state = _TrackState(track_id=1)
    good_face = [_hit(det_score=0.9)]
    # First hit: registers but no lock yet.
    _run_face_check(pipeline, state, hits=good_face, match_score=0.85, frame_idx=0)
    assert state.label == LABEL_UNKNOWN
    assert state.face_hits == 1
    # Second hit: locks as PATIENT.
    _run_face_check(pipeline, state, hits=good_face, match_score=0.85, frame_idx=5)
    assert state.label == LABEL_PATIENT


# ─── Mixed paths ─────────────────────────────────────────────────


def test_mixed_paths_only_path_c_counts_toward_other_lock() -> None:
    """A clip with mostly no-face attempts (Path A) and a few quality
    face attempts that didn't match (Path C). The total face_checks is
    high enough to have locked OTHER under the OLD rule, but
    face_seen_attempts is below threshold — track stays UNKNOWN."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    good_face_no_match = [_hit(det_score=0.9)]

    # 3 Path C attempts: face seen + matched + failed
    for i in range(3):
        _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=i * 5)
    # 12 Path A attempts: no face seen
    for i in range(12):
        _run_face_check(pipeline, state, hits=None, frame_idx=(3 + i) * 5)

    assert state.face_checks == 15
    assert state.face_seen_attempts == 3   # only Path C counts
    assert state.label == LABEL_UNKNOWN     # 3 < 5, no lock


def test_mixed_paths_then_two_more_path_c_locks_other() -> None:
    """Continuation of the mixed scenario above: after 2 more Path C
    attempts, total verified attempts hits 5, OTHER lock fires."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    good_face_no_match = [_hit(det_score=0.9)]

    # 3 path C
    for i in range(3):
        _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=i * 5)
    # 5 path A
    for i in range(5):
        _run_face_check(pipeline, state, hits=None, frame_idx=(3 + i) * 5)
    assert state.label == LABEL_UNKNOWN

    # Two more path C — push verified attempts to 5
    for i in range(2):
        _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=(8 + i) * 5)
    assert state.face_seen_attempts == 5
    assert state.label == LABEL_OTHER


def test_path_d_takes_precedence_after_initial_path_c() -> None:
    """A track that face-matches eventually still locks PATIENT, even
    after some path C failures earlier in the clip. Patient lock
    semantics aren't degraded by the new OTHER semantic."""
    pipeline = _make_pipeline(face_checks_for_other=10, hits_to_lock=2)
    state = _TrackState(track_id=1)
    good_face_no_match = [_hit(det_score=0.9)]
    # 3 failures (don't reach OTHER lock; threshold=10)
    for i in range(3):
        _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=i * 5)
    assert state.label == LABEL_UNKNOWN
    assert state.face_hits == 0
    # Now 2 matching hits — PATIENT lock fires.
    good_face_match = [_hit(det_score=0.9)]
    _run_face_check(pipeline, state, hits=good_face_match, match_score=0.85, frame_idx=15)
    _run_face_check(pipeline, state, hits=good_face_match, match_score=0.85, frame_idx=20)
    assert state.label == LABEL_PATIENT


# ─── Aggregate accessor: has_only_verified_other_tracks ──────────


def test_has_only_verified_other_tracks_false_when_no_tracks() -> None:
    pipeline = _make_pipeline()
    assert pipeline.has_only_verified_other_tracks is False


def test_has_only_verified_other_tracks_false_when_any_unknown() -> None:
    pipeline = _make_pipeline()
    pipeline._tracks[1] = _TrackState(track_id=1, label=LABEL_OTHER)
    pipeline._tracks[2] = _TrackState(track_id=2, label=LABEL_UNKNOWN)
    assert pipeline.has_only_verified_other_tracks is False


def test_has_only_verified_other_tracks_false_when_patient_present() -> None:
    pipeline = _make_pipeline()
    pipeline._tracks[1] = _TrackState(track_id=1, label=LABEL_OTHER)
    pipeline._tracks[2] = _TrackState(track_id=2, label=LABEL_PATIENT)
    assert pipeline.has_only_verified_other_tracks is False


def test_has_only_verified_other_tracks_true_when_all_other() -> None:
    pipeline = _make_pipeline()
    pipeline._tracks[1] = _TrackState(track_id=1, label=LABEL_OTHER)
    pipeline._tracks[2] = _TrackState(track_id=2, label=LABEL_OTHER)
    assert pipeline.has_only_verified_other_tracks is True


def test_has_only_verified_other_tracks_true_for_single_other_track() -> None:
    """Single-track happy case from the production logs."""
    pipeline = _make_pipeline()
    pipeline._tracks[1] = _TrackState(track_id=1, label=LABEL_OTHER)
    assert pipeline.has_only_verified_other_tracks is True


# ─── Regression guard: patient-face-hidden scenario ──────────────


def test_patient_face_hidden_entire_clip_stays_unknown() -> None:
    """Hard constraint from the design: a patient who turns their face
    away the whole clip MUST NOT be mis-labeled OTHER. The new
    semantic gates the OTHER lock on `face_seen_attempts` (verified
    match attempts), so 0 face_seen_attempts means stay UNKNOWN
    regardless of how many face_checks ran.
    """
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    # Simulate a 15-second clip at 5fps with face_check every 5 frames
    # → ~15 face check attempts. SCRFD returns no face every time
    # (patient consistently turned away).
    for i in range(15):
        _run_face_check(pipeline, state, hits=None, frame_idx=i * 5)
    assert state.face_checks == 15
    assert state.face_seen_attempts == 0
    assert state.label == LABEL_UNKNOWN   # the load-bearing assertion
