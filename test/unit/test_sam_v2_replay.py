"""SAM v2 unit coverage: the wire parser's hardening and the
RemoteReplayPipeline's core promise — pod observations replayed
through the UNCHANGED BodyReIDPipeline brain produce the same
locks/labels the legacy arm would (reviews G2/C-suite, 2026-07-20).

No network, no models: observations are hand-built, embeddings are
unit vectors, frames are tiny black images (the replay never looks at
pixels — perception came from the pod)."""

from __future__ import annotations

import numpy as np
import pytest

from src.services.reid.body_pipeline import (
    LABEL_OTHER,
    LABEL_PATIENT,
    PipelineConfig,
)
from src.services.reid.remote_replay import (
    RemoteFaceObs,
    RemoteObservations,
    RemoteReplayPipeline,
)
from src.services.reid.sam_client import _parse_observe

DIM = 512


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _obs(
    *,
    n_frames: int,
    tid: int = 1,
    face_frames: dict[int, np.ndarray] | None = None,
) -> RemoteObservations:
    """One track present every frame at a fixed bbox; optional face
    observations (det 0.9, 70px-tall bbox — above every quality
    floor) on the given frames."""
    rows = {
        i: np.asarray(
            [[5.0, 5.0, 95.0, 95.0, float(tid), 0.9, 0.0, 0.0]],
            dtype=np.float32,
        )
        for i in range(n_frames)
    }
    face_obs = {}
    for fi, emb in (face_frames or {}).items():
        face_obs[(tid, fi)] = [RemoteFaceObs(
            frame_idx=fi, embedding=emb, det_score=0.9,
            bbox=(10.0, 10.0, 80.0, 80.0),
        )]
    return RemoteObservations(
        rows_by_frame=rows, face_obs=face_obs,
        frames_seen=n_frames, det_count=n_frames,
        max_prob=0.9, mean_prob=0.9, person_thr=0.4,
    )


def _run(pipe: RemoteReplayPipeline, n_frames: int) -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(n_frames):
        pipe.process_frame(frame, i)


def test_replay_locks_patient_through_the_real_brain():
    """Two quality face matches (interval 5 apart) → hits_to_lock=2 →
    PATIENT lock, exactly the legacy state machine's rules."""
    gallery = np.stack([_unit(0)])
    obs = _obs(n_frames=12, face_frames={0: _unit(0), 5: _unit(0)})
    pipe = RemoteReplayPipeline(
        observations=obs, gallery=gallery,
        config=PipelineConfig(), pre_match_callback=None,
    )
    _run(pipe, 12)
    tracks = pipe.track_observations()
    assert len(tracks) == 1
    assert tracks[0].label == LABEL_PATIENT
    assert tracks[0].face_locked is True
    assert pipe.has_any_patient_track
    assert pipe.best_score == pytest.approx(1.0)
    assert pipe.latest_patient_bbox == (5.0, 5.0, 95.0, 95.0)


def test_replay_brands_other_after_verified_mismatches():
    """Five quality faces that don't match the gallery → OTHER lock
    (face_checks_for_other=5) → the LLM-skip signal fires."""
    gallery = np.stack([_unit(0)])
    stranger = _unit(1)  # orthogonal — cosine 0.0
    obs = _obs(
        n_frames=26,
        face_frames={f: stranger for f in (0, 5, 10, 15, 20)},
    )
    pipe = RemoteReplayPipeline(
        observations=obs, gallery=gallery,
        config=PipelineConfig(), pre_match_callback=None,
    )
    _run(pipe, 26)
    tracks = pipe.track_observations()
    assert tracks[0].label == LABEL_OTHER
    assert pipe.has_only_verified_other_tracks


def test_replay_no_faces_stays_unknown_and_presence_populates():
    """No face observations (face turned away / cap 0): the track must
    stay UNKNOWN — never OTHER — while presence still counts (the
    always-on-tracking invariant)."""
    gallery = np.stack([_unit(0)])
    obs = _obs(n_frames=10)
    pipe = RemoteReplayPipeline(
        observations=obs, gallery=gallery,
        config=PipelineConfig(), pre_match_callback=None,
    )
    _run(pipe, 10)
    tracks = pipe.track_observations()
    assert tracks[0].label not in (LABEL_PATIENT, LABEL_OTHER)
    assert pipe.num_tracks == 1
    assert pipe.max_concurrent_tracks == 1
    assert not pipe.has_only_verified_other_tracks


def _wire_payload() -> dict:
    import base64

    emb_b64 = base64.b64encode(_unit(0).tobytes()).decode("ascii")
    return {
        "contract": "observe-v2",
        "frames_seen": 2,
        "frame_indices": [0, 1],
        "tracks": [{
            "track_id": 3,
            "rows": [
                [0, 5.0, 5.0, 95.0, 95.0, 0.9],
                [1, 6.0, 6.0, 96.0, 96.0, float("nan")],  # dropped
            ],
            "face_obs": [[0, 0.9, 10.0, 10.0, 80.0, 80.0, emb_b64]],
        }],
        "stats": {"det_count": 2, "max_prob": 0.9, "mean_prob": 0.9,
                  "person_thr": 0.4},
        "timings_ms": {"decode": 1.0, "sam": 2.0, "face": 3.0,
                       "total": 6.0},
    }


def test_parse_observe_shapes_space_and_nonfinite_drop():
    obs = _parse_observe(_wire_payload())
    # Row space: keyed by ORIGINAL frame idx, BoT-SORT (N, 8) shape.
    assert set(obs.rows_by_frame.keys()) == {0}       # NaN row dropped
    rows = obs.rows_by_frame[0]
    assert rows.shape == (1, 8)
    assert rows[0, 4] == 3.0                          # tid in col 4
    assert rows[0, 5] == pytest.approx(0.9)           # conf in col 5
    hits = obs.face_obs[(3, 0)]
    assert hits[0].embedding.shape == (DIM,)
    assert hits[0].det_score == pytest.approx(0.9)


def test_parse_observe_malformed_row_raises():
    """Wrong arity must RAISE (→ label_clip fails open loudly), never
    half-parse."""
    bad = _wire_payload()
    bad["tracks"][0]["rows"] = [[0, 1.0, 2.0]]
    with pytest.raises(ValueError):
        _parse_observe(bad)


def test_parse_observe_caps_track_flood():
    """A hostile/buggy pod cannot allocate unbounded backend memory:
    tracks beyond the cap are dropped."""
    flood = _wire_payload()
    flood["tracks"] = [
        {"track_id": i, "rows": [[0, 1.0, 1.0, 2.0, 2.0, 0.5]],
         "face_obs": []}
        for i in range(300)
    ]
    obs = _parse_observe(flood)
    tids = {int(r[4]) for rows in obs.rows_by_frame.values() for r in rows}
    assert len(tids) <= 64


def test_oracle_refuted_claims_are_silence_not_testimony():
    """B3 incident regression: face claims the SAM-face concept
    refuted (oracle_face=False) must neither vote nor testify — five
    refuted mismatches may NOT brand OTHER (they're the back of a
    head), and the track stays UNKNOWN."""
    gallery = np.stack([_unit(0)])
    stranger = _unit(1)
    obs = _obs(
        n_frames=26,
        face_frames={f: stranger for f in (0, 5, 10, 15, 20)},
    )
    for hits in obs.face_obs.values():
        for i, h in enumerate(hits):
            hits[i] = RemoteFaceObs(
                frame_idx=h.frame_idx, embedding=h.embedding,
                det_score=h.det_score, bbox=h.bbox, oracle_face=False,
            )
    pipe = RemoteReplayPipeline(
        observations=obs, gallery=gallery,
        config=PipelineConfig(), pre_match_callback=None,
    )
    _run(pipe, 26)
    tracks = pipe.track_observations()
    assert tracks[0].label not in (LABEL_PATIENT, LABEL_OTHER)
    assert not pipe.has_only_verified_other_tracks
    assert pipe.best_score == 0.0          # refuted claims don't score


def _two_track_obs(n_frames: int = 8) -> RemoteObservations:
    """Two tracks every frame: tid=1 at x<100, tid=2 at x>=200."""
    rows = {
        i: np.asarray(
            [[5.0, 5.0, 95.0, 95.0, 1.0, 0.9, 0.0, 0.0],
             [200.0, 5.0, 290.0, 95.0, 2.0, 0.9, 0.0, 1.0]],
            dtype=np.float32,
        )
        for i in range(n_frames)
    }
    return RemoteObservations(
        rows_by_frame=rows, face_obs={}, frames_seen=n_frames,
        det_count=2 * n_frames, max_prob=0.9, mean_prob=0.9,
        person_thr=0.4,
    )


