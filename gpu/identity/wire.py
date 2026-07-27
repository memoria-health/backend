"""RemoteReplayPipeline — the GPU port's backend half (2026-07-20).

The SAM pod is PERCEPTION ONLY: it detects + tracks people (SAM 3.1
concept propagation), cuts mask-clean crops, and face-detects + embeds
(SCRFD + ArcFace) — returning RAW observations with zero identity
decisions. This class replays those observations through the UNCHANGED
`BodyReIDPipeline` decision machinery via the `_perceive`/`_face_hits`
seam, so every label, lock bar, quality floor, single-patient
invariant, pre-match (carry + cross-clip pool), and harvest path is
the SAME CODE the legacy YOLO arm runs. The A/B is therefore exactly
one variable: whose eyes.

Index space: everything arrives keyed by ORIGINAL extracted-frame
index (the pod names frames by the index the backend sent), so the
replay aligns 1:1 with the frames `inference._run_pipeline_sync`
decodes — no remapping layer.

Face-check alignment: the pod schedules face attempts per track the
same way the brain consumes them (first frame seen, then every
`face_check_interval` frames, capped) — see sam_service. When the
brain asks `_face_hits` for a (track, frame) the pod didn't embed
(cap hit, or scheduling skew from frames only one side decoded), the
answer is "no face seen" — epistemically Path A, which never brands
OTHER, so skew degrades to fewer attempts, never to wrong labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from identity.matching import FaceHit

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteFaceObs:
    """One pod face observation: SCRFD found + ArcFace embedded a face
    inside a track's mask-clean crop on one frame. Bbox is in
    extracted-frame pixel coords (already translated by the server)."""

    frame_idx: int
    embedding: np.ndarray            # (512,) float32, L2-normalized
    det_score: float
    bbox: tuple[float, float, float, float]
    # SAM-face countersign: False = the face CONCEPT session found no
    # face overlapping this claim (counterfeit — e.g. back of a head).
    # The brain treats unconfirmed claims as silence (Path B).
    oracle_face: bool = True


@dataclass
class RemoteObservations:
    """The pod's whole-clip perception output, parsed by sam_client.

    `rows_by_frame`: frame_idx -> (N, 8) float32 array shaped exactly
    like BoT-SORT output rows (x1, y1, x2, y2, tid, conf, cls,
    det_ind) so `process_frame`'s row loop consumes them verbatim.
    `face_by_track_frame`: (tid, frame_idx) -> [FaceHit-compatible].
    """

    # Pod session epoch — track ids are only comparable within one
    # epoch; a pod restart recycles numbers (2026-07-25 T13 rebirth).
    epoch: str = ""
    rows_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    face_obs: dict[tuple[int, int], list[RemoteFaceObs]] = field(
        default_factory=dict,
    )
    frames_seen: int = 0             # frames the pod decoded + tracked
    det_count: int = 0
    max_prob: float = 0.0
    mean_prob: float = 0.0
    person_thr: float = 0.0
    timings_ms: dict[str, float] = field(default_factory=dict)
    # Display regions from the pod's screen-concept pass (only present
    # on ticks where the backend requested one): (x0, y0, x1, y1, prob)
    # in extracted-frame pixels. Feeds the per-camera ScreenMap.
    screen_regions: tuple[tuple[float, float, float, float, float], ...] = ()

    @staticmethod
    def empty() -> "RemoteObservations":
        return RemoteObservations()
