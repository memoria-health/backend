"""IdentityResult — the per-tick identity verdict every detection
consumer reads.

Extracted from inference.py (2026-07-26 productionization) so the
verdict OUTLIVES the legacy in-process reid brain that used to
produce it: since implementation A, the sidecar provider synthesizes
this same shape from the GPU-host identity service's /state, and
camera_ingest / TrackLedger / scheduler / executor / vlm_routing all
consume it unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.reid.observations import TrackObservation

if TYPE_CHECKING:
    from src.services.detection.frame_extraction import ExtractedFrame


@dataclass(frozen=True)
class IdentityResult:
    """Per-tick identity decision for a single patient.

    `reid_ran`: True iff the identity provider executed and produced a
    verdict. False on timeout or crash ONLY. The upstream gate in
    `camera_ingest` keys off this: when False, it bypasses
    identity-based suppression. Load-bearing safety invariant: an
    identity infrastructure failure must NEVER silence the detector
    pipeline.

    `attribution_ran`: True iff identity labels (PATIENT / OTHER /
    pre-labels) were even possible this tick. The suppression gate
    additionally requires this: a tick where attribution could not run
    must never suppress the LLM, no matter what the track labels claim.

    `patient_in_clip`: True iff at least one track was locked as
    PATIENT during this tick. Only meaningful when `reid_ran=True`.

    `confidence`: best identity-match score across the tick's checks.
    0.0 when no match.

    `face_count`: face checks that returned at least one face — lets
    the upstream gate distinguish "saw no faces" (ambiguous) from "saw
    faces, none matched" (confident not-patient).

    `patient_bbox`: pixel-space BODY bbox of the most-recently-seen
    locked PATIENT track; None when no lock.

    `people_present_reid` / `people_count_reid`: track-derived
    presence and peak concurrent count — presence holds even when no
    face is visible.
    """
    reid_ran: bool
    patient_in_clip: bool
    confidence: float
    best_frame_index: int | None
    face_count: int
    patient_bbox: tuple[float, float, float, float] | None = None
    people_present_reid: bool = False
    people_count_reid: int = 0
    attribution_ran: bool = False
    # One TrackObservation per live track — the signals layer scopes
    # motion per track with these. None iff the provider didn't run
    # (timeout/crash); empty tuple = ran, no people.
    tracks: tuple[TrackObservation, ...] | None = None
    # True iff num_tracks > 0 AND every track ended verified-OTHER.
    # Drives the "skip LLM when everyone is verified not-patient"
    # branch. The sidecar provider hard-sets this False (its one-seat
    # model has no verified-others), so suppression can never fire
    # from sidecar state.
    has_only_verified_other_tracks: bool = False
    # True iff attribution was ELIGIBLE this tick (a usable gallery
    # was in play). Independent of reid_ran.
    faces_enrolled: bool = False
    # Frames the provider already decoded (legacy brain only; the
    # sidecar leaves these None). vlm_routing reuses them for bbox
    # drawing ONLY on attribution ticks.
    extracted_frames: list[ExtractedFrame] | None = None
    # One body bbox per extracted frame (same pixel space); None
    # entries where no PATIENT lock existed at that instant.
    per_frame_bboxes: list[tuple[float, float, float, float] | None] | None = None
    # Sampling rate of `extracted_frames` (prompt honesty for the LLM).
    sample_fps: int | None = None
    # ── Input-quality admissibility (identity-persistence W2) ────
    # `quality_degraded` = this tick's pixels are inadmissible as
    # evidence of ABSENCE — camera_ingest freezes the state machine's
    # close countdown on it. It never blocks analysis.
    quality_width: int | None = None
    quality_sharpness: float | None = None
    quality_frames: int = 0
    quality_degraded: bool = False

    # Legacy alias: the semantic shifted from "face bbox" to "body
    # bbox" with the YOLO + BoT-SORT port; call sites still compile.
    @property
    def face_bbox(self) -> tuple[float, float, float, float] | None:
        return self.patient_bbox

    @classmethod
    def empty(cls, *, faces_enrolled: bool = False) -> "IdentityResult":
        return cls(
            reid_ran=False,
            patient_in_clip=False,
            confidence=0.0,
            best_frame_index=None,
            face_count=0,
            patient_bbox=None,
            extracted_frames=None,
            per_frame_bboxes=None,
            sample_fps=None,
            people_present_reid=False,
            people_count_reid=0,
            attribution_ran=False,
            tracks=None,
            has_only_verified_other_tracks=False,
            faces_enrolled=faces_enrolled,
        )
