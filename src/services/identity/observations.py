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


@dataclass(frozen=True)
class TrackObservation:
    """Consumer-facing snapshot of one track after a clip run.

    `label` is LABEL_PATIENT / LABEL_OTHER / LABEL_UNKNOWN.
    `confidence` is the face-match score for PATIENT locks, 0.0
    otherwise. `pre_labeled` means the label came from a gallery
    rather than (only) this tick's face evidence. `bbox_series`
    coordinates are in extracted-frame pixel space — scale with
    `rapid_call.scale_bbox_to_frame` before touching native buffer
    frames.
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
