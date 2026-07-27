"""/observe wire parsing — the surviving half of the old replay tests.

The RemoteReplayPipeline (pod observations replayed through the legacy
BodyReIDPipeline brain) died with the brain on 2026-07-27; identity now
lives in the GPU-host service. What survives here is the WIRE: sam_client
._parse_observe must keep shaping /observe payloads correctly for the
identity service's loop.
"""

import pytest

from identity.pod_client import _parse_observe

import numpy as np

DIM = 512


def _unit(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


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


