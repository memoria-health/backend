"""Track observation vocabulary — the per-track unit of the v2
detection layers (per-track motion, subject filtering, crop
targeting, identity-lock lanes).

Extracted from body_pipeline.py (2026-07-26 productionization) so the
consumers that OUTLIVE the legacy in-process reid brain — TrackLedger,
tick signals, scheduler/executor, id-evidence dumps, and the sidecar
identity provider — depend on a module with zero pipeline machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

LABEL_PATIENT = "PATIENT"
LABEL_OTHER = "OTHER"
LABEL_UNKNOWN = "UNKNOWN"

# Which pixel grid `bbox_series` (and `face_px_max`) is measured in.
#
# EXTRACTED — the legacy in-process reid brain ran YOLO on frames
#   downscaled to `settings.reid_inference_sample_width`, so its boxes
#   need scaling up by native/extracted before they touch a buffer
#   frame.
# NATIVE — the identity sidecar (`gpu/identity/service.py`) pushes
#   frames at full resolution (`INTAKE_MAX_W = 0`, 2026-07-26) and
#   returns boxes already in native pixels. Scaling them AGAIN is the
#   2026-08-07 crop bug: a 1920-wide frame with sample_width=640 got a
#   3× multiply, and `person_crop_box` returned (520, 0, 1920, 1080)
#   for a person standing at x=333..666 — the crop contained their
#   legs and 1400px of empty room. Every person-bucket detector then
#   answered "no person is visible" on a clip where the scene call
#   could see the patient plainly.
#
# The tag exists because a bare 4-tuple of floats carries no space,
# so a consumer can only GUESS which grid it is on — and when the
# provider changed underneath, every consumer kept guessing the old
# answer in silence. This is the third bug of that exact shape (the
# EdgeTAM mask dimensions on 2026-08-06 were the second). Anything
# handing coordinates across the sidecar boundary must say which grid
# they are on.
BBOX_SPACE_EXTRACTED = "extracted"
BBOX_SPACE_NATIVE = "native"


@dataclass(frozen=True)
class TrackObservation:
    """Consumer-facing snapshot of one track after a clip run.

    `label` is LABEL_PATIENT / LABEL_OTHER / LABEL_UNKNOWN.
    `confidence` is the face-match score for PATIENT locks, 0.0
    otherwise. `pre_labeled` means the label came from a gallery
    rather than (only) this tick's face evidence.

    `bbox_series` coordinates live on the grid named by `bbox_space`.
    Resolve it with `bbox_extracted_width` and scale with
    `rapid_call.scale_bbox_to_frame` before touching native buffer
    frames — never assume the grid.
    """
    track_id: int
    label: str
    confidence: float
    # Keys: node wall-clock ts (sidecar, follow-crop series) or
    # legacy frame index (0 = single snapshot).
    bbox_series: tuple[tuple[float, tuple[float, float, float, float]], ...]
    face_locked: bool
    pre_labeled: bool
    # Tallest usable face on this track, px on the `bbox_space` grid
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
    # Seconds since perception LAST SAW this track (staleness, not
    # lifetime — conflating them degraded every crop, 2026-08-18).
    age_s: float = 0.0
    # Grid `bbox_series` / `face_px_max` are measured on. Defaults to
    # EXTRACTED so the legacy in-process path keeps its exact
    # behaviour; the sidecar provider sets NATIVE explicitly.
    bbox_space: str = BBOX_SPACE_EXTRACTED


def bbox_extracted_width(
    track: TrackObservation, native_width: float, *, sample_width: float,
) -> float:
    """The pixel width `track.bbox_series` is measured against.

    Feed this to `scale_bbox_to_frame(..., sample_width=...)` instead
    of the raw setting. For a NATIVE track it returns the native width,
    which makes the scale factor exactly 1.0 — the boxes are already
    where they belong. For an EXTRACTED track it reproduces the old
    `min(sample_width, native)` verbatim.
    """
    if track.bbox_space == BBOX_SPACE_NATIVE:
        return float(native_width)
    return min(float(sample_width), float(native_width))
