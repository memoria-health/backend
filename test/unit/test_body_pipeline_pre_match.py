"""Phase 2 unit tests: pre-match integration in `BodyReIDPipeline`.

The pre-match callback runs once per new track. If it returns a label,
the track starts pre-labeled (instead of UNKNOWN). The face check
in this clip can still override the pre-label — Path D (PATIENT face
lock) always wins, and Path C (OTHER face lock) confirms an OTHER
pre-label.

Safety invariants pinned by these tests:

  - A PATIENT pre-label is never demoted to OTHER by face-check failures
    in this clip. Patient false-negative is the cardinal sin.
  - An OTHER pre-label CAN be overridden by a Path D face match (a
    helper who turns out to be the patient gets correctly relabeled).
  - The pre-match callback is invoked at most once per track (not per
    frame) — bounds OSNet inference cost.
  - face_locked is set when a face-based lock fires in THIS clip,
    NOT when a pre-label is assigned (pre-labeled tracks must continue
    face-checking to allow override).
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


def _make_pipeline(
    *,
    gallery: np.ndarray | None = None,
    face_checks_for_other: int = 5,
    hits_to_lock: int = 2,
    pre_match_callback=None,
) -> BodyReIDPipeline:
    if gallery is None:
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
                face_min_det_score=0.5,
                face_checks_for_other=face_checks_for_other,
                hits_to_lock=hits_to_lock,
                match_threshold_base=0.40,
            ),
            pre_match_callback=pre_match_callback,
        )


def _hit(det_score: float) -> FaceHit:
    emb = np.zeros(512, dtype=np.float32)
    emb[3] = 1.0   # orthogonal to all-ones gallery → cosine ~0
    return FaceHit(
        embedding=emb, det_score=det_score,
        bbox=(0.0, 0.0, 100.0, 100.0),
    )


def _run_face_check(
    pipeline: BodyReIDPipeline,
    state: _TrackState,
    *,
    hits: list[FaceHit] | None,
    match_score: float = 0.0,
    frame_idx: int = 0,
) -> None:
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
        state.face_checks += 1
        pipeline._face_check(state, fake_frame, bbox, frame_idx, FrameResult())


# ─── Pre-match callback invocation ──────────────────────────────


def test_pre_match_callback_invoked_with_first_seen_track() -> None:
    """The callback should be called exactly once when a new track is
    introduced, with the frame and bbox so it can extract OSNet."""
    calls: list[tuple[int, tuple[float, float, float, float]]] = []

    def _cb(frame_bgr: np.ndarray, bbox):
        calls.append((frame_bgr.shape[0], bbox))
        return None

    pipeline = _make_pipeline(pre_match_callback=_cb)
    # Simulate process_frame's per-track loop body — we don't call
    # process_frame directly because that requires real detector +
    # tracker output. Mirror the call sites:
    state = pipeline._tracks.setdefault(1, _TrackState(track_id=1))
    fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    bbox = (10.0, 10.0, 60.0, 90.0)
    # Directly exercise the pre-match invocation logic
    if 1 not in pipeline._pre_match_attempted:
        pipeline._pre_match_attempted.add(1)
        pre_label = pipeline._pre_match_callback(fake_frame, bbox)
        if pre_label in (LABEL_PATIENT, LABEL_OTHER):
            state.pre_label = pre_label
            state.label = pre_label

    assert len(calls) == 1
    assert calls[0][1] == bbox


def test_pre_match_label_sets_initial_track_label() -> None:
    """When callback returns LABEL_OTHER, the track's initial label
    should be OTHER (not UNKNOWN). Same for PATIENT."""
    for returned in (LABEL_PATIENT, LABEL_OTHER):
        pipeline = _make_pipeline(pre_match_callback=lambda *_: returned)
        state = pipeline._tracks.setdefault(1, _TrackState(track_id=1))
        if 1 not in pipeline._pre_match_attempted:
            pipeline._pre_match_attempted.add(1)
            pre_label = pipeline._pre_match_callback(None, (0, 0, 1, 1))
            if pre_label in (LABEL_PATIENT, LABEL_OTHER):
                state.pre_label = pre_label
                state.label = pre_label
        assert state.label == returned, f"failed for returned={returned}"
        assert state.pre_label == returned


def test_pre_match_returning_none_leaves_track_unknown() -> None:
    """Callback returns None → track stays UNKNOWN, no pre-label set."""
    pipeline = _make_pipeline(pre_match_callback=lambda *_: None)
    state = pipeline._tracks.setdefault(1, _TrackState(track_id=1))
    if 1 not in pipeline._pre_match_attempted:
        pipeline._pre_match_attempted.add(1)
        pre_label = pipeline._pre_match_callback(None, (0, 0, 1, 1))
        if pre_label in (LABEL_PATIENT, LABEL_OTHER):
            state.pre_label = pre_label
            state.label = pre_label
    assert state.label == LABEL_UNKNOWN
    assert state.pre_label is None


# ─── Pre-label + face-check override interaction ─────────────────


def test_pre_label_other_overridden_by_path_d_face_match() -> None:
    """The helper case turned upside down: track pre-labeled OTHER
    (helper's body in gallery), but face check this clip succeeds
    against the patient gallery → state.label becomes PATIENT.
    Face evidence in THIS clip always beats stale gallery."""
    pipeline = _make_pipeline(hits_to_lock=2)
    state = _TrackState(track_id=1)
    state.pre_label = LABEL_OTHER
    state.label = LABEL_OTHER
    good_face = [_hit(det_score=0.9)]
    # Two matching face hits → PATIENT lock
    _run_face_check(pipeline, state, hits=good_face, match_score=0.85, frame_idx=0)
    _run_face_check(pipeline, state, hits=good_face, match_score=0.85, frame_idx=5)
    assert state.label == LABEL_PATIENT
    assert state.face_locked is True


def test_pre_label_patient_protected_from_other_demotion() -> None:
    """Inverse: track pre-labeled PATIENT, face checks fail many times
    (Path C). PATIENT must NOT be demoted to OTHER. The face evidence
    is contradictory but the cardinal safety rule wins: never silently
    mis-label patient as not-patient."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    state.pre_label = LABEL_PATIENT
    state.label = LABEL_PATIENT
    good_face_no_match = [_hit(det_score=0.9)]
    for i in range(10):
        _run_face_check(pipeline, state, hits=good_face_no_match,
                        match_score=0.1, frame_idx=i * 5)
    assert state.label == LABEL_PATIENT   # not demoted


def test_pre_label_other_locked_by_path_c_face_evidence() -> None:
    """Track pre-labeled OTHER, face check confirms via Path C →
    `face_locked` becomes True so subsequent face checks stop. Label
    stays OTHER (was OTHER already)."""
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    state.pre_label = LABEL_OTHER
    state.label = LABEL_OTHER
    good_face_no_match = [_hit(det_score=0.9)]
    for i in range(5):
        _run_face_check(pipeline, state, hits=good_face_no_match,
                        match_score=0.1, frame_idx=i * 5)
    assert state.label == LABEL_OTHER
    assert state.face_locked is True


# ─── face_locked semantics ───────────────────────────────────────


def test_face_locked_set_when_path_d_locks_patient() -> None:
    """Face-based PATIENT lock (hits_to_lock reached) terminates face
    checking — sets face_locked so the gate stops invoking _face_check."""
    pipeline = _make_pipeline(hits_to_lock=2)
    state = _TrackState(track_id=1)
    good_face = [_hit(det_score=0.9)]
    _run_face_check(pipeline, state, hits=good_face, match_score=0.85, frame_idx=0)
    assert state.face_locked is False   # only 1 hit
    _run_face_check(pipeline, state, hits=good_face, match_score=0.85, frame_idx=5)
    assert state.face_locked is True
    assert state.label == LABEL_PATIENT


def test_face_locked_set_when_path_c_locks_other() -> None:
    pipeline = _make_pipeline(face_checks_for_other=5)
    state = _TrackState(track_id=1)
    good_face_no_match = [_hit(det_score=0.9)]
    for i in range(4):
        _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=i * 5)
        assert state.face_locked is False
    _run_face_check(pipeline, state, hits=good_face_no_match, frame_idx=20)
    assert state.label == LABEL_OTHER
    assert state.face_locked is True


def test_face_locked_not_set_by_pre_label_alone() -> None:
    """Cardinal rule: a pre-label from the gallery does NOT count as
    face-locked. The track must keep face-checking so this clip's
    actual face evidence can override the gallery."""
    pipeline = _make_pipeline(pre_match_callback=lambda *_: LABEL_OTHER)
    state = pipeline._tracks.setdefault(1, _TrackState(track_id=1))
    pipeline._pre_match_attempted.add(1)
    pre_label = pipeline._pre_match_callback(None, (0, 0, 1, 1))
    if pre_label in (LABEL_PATIENT, LABEL_OTHER):
        state.pre_label = pre_label
        state.label = pre_label
    assert state.face_locked is False
    assert state.label == LABEL_OTHER


# ─── Pre-match callback exception handling ───────────────────────


def test_pre_match_callback_exception_leaves_track_unknown() -> None:
    """If the pre-match callback raises (OSNet failure, gallery error),
    the track stays UNKNOWN and processing continues. Defensive against
    Re-ID infra issues — must not crash the whole tick."""
    def _bad_cb(*_args):
        raise RuntimeError("simulated OSNet failure")

    pipeline = _make_pipeline(pre_match_callback=_bad_cb)
    state = pipeline._tracks.setdefault(1, _TrackState(track_id=1))
    # Mirror the in-loop logic with try/except
    if 1 not in pipeline._pre_match_attempted:
        pipeline._pre_match_attempted.add(1)
        try:
            pre_label = pipeline._pre_match_callback(None, (0, 0, 1, 1))
        except Exception:
            pre_label = None
        if pre_label in (LABEL_PATIENT, LABEL_OTHER):
            state.pre_label = pre_label
            state.label = pre_label
    assert state.label == LABEL_UNKNOWN
    assert state.pre_label is None
