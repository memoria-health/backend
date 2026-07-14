"""RAPID recipe pieces (detection v2).

The validated RAPID recipe — overlay-tinted static person crop at native
fps, flash tier, describe-then-decide prompt. Measured: convulsion 8/10
recall, 1/30 FP; the lite tier is blind to this motion class at any
packaging (docs/architecture/detection-architecture-v2.md §0 law 2, §1.4).

The P0 standalone lane (`split_rapid` + `run_rapid_seizure` behind
`NEMARA_RAPID_SEIZURE_ENABLED`) is retired — Cleanup Ledger row 5,
Aryaman-directed 2026-07-14. RAPID-bucket detectors now flow through the
ordinary v2 scheduler → packer → executor path like every other bucket;
this module keeps only the recipe pieces the executor consumes:

  - `RAPID_SEIZURE_PROMPT` / `rapid_seizure_spec()` — the overlay-channel
    prompt for the seizure detector. The catalog row's prompt describes
    the clean main-call clip (and is what legacy flag-off still uses);
    it must never be used on a tinted crop, and vice versa.
  - `RAPID_MOTION_SPEC` — the describe-then-decide preamble that rides
    every overlay call (part of the validated recipe).
  - `scale_bbox_to_frame` — Re-ID extracted-frame → native-pixel box
    scaling, shared with the person-crop builder.
"""

from __future__ import annotations

from src.services.detection.prompt import DetectorSpec
from src.services.detection.rapid_artifact import Box

RAPID_SEIZURE_PROMPT = """\
SEIZURE (convulsive shaking) — this clip is a MOTION-OVERLAY channel: regions
that changed between frames are tinted bright red on the video. The tint shows
WHERE motion happened, never WHAT it is — red tint alone is NEVER sufficient
evidence of a seizure.

Judge only what the body is doing, in ANY position (lying, sitting, standing,
slumped on a couch or the floor):

- A convulsive seizure shows SUSTAINED, RHYTHMIC, INVOLUNTARY shaking with
  visible amplitude — limbs or torso visibly displacing back and forth,
  continuing across the clip.
- A person holding still whose outline carries only faint red wisps or fine
  trembling is shivering or a resting tremor — NOT a seizure.
- Steady red outlines from walking, gesturing, typing, stretching, exercise,
  screen flicker, or camera noise are ordinary motion — NOT a seizure.

Set seizure_detected=true only when amplitude AND rhythm AND involuntary
character are all visible in the motion itself."""


RAPID_MOTION_SPEC = DetectorSpec(
    identifier="rapid_motion",
    name="Motion description",
    prompt_fragment=(
        "FIRST, before any verdict, describe the dominant motion in this "
        "clip in one or two sentences: which body parts move, how fast, "
        "how far they displace, and whether the pattern is rhythmic or "
        "irregular. Describe the motion itself, not the red tint."
    ),
    schema_fields={
        "rapid_motion_description": {
            "type": "string",
            "description": "1-2 sentence description of the dominant motion",
        }
    },
    always_on=True,
)


def rapid_seizure_spec() -> DetectorSpec:
    """The seizure spec for overlay calls — same identifier (and therefore
    the same `seizure_*` field trio) as the catalog row, different prompt."""
    return DetectorSpec(
        identifier="seizure",
        name="Seizure",
        prompt_fragment=RAPID_SEIZURE_PROMPT,
        schema_fields={
            "seizure_detected": {"type": "boolean"},
            "seizure_confidence": {"type": "number"},
            "seizure_reasoning": {"type": "string"},
        },
    )


def scale_bbox_to_frame(
    bbox: Box, frame_w: float, frame_h: float, *, sample_width: float
) -> Box | None:
    """Re-ID boxes are computed on frames extracted at
    `max_width=sample_width` (frame_extraction scales to min(sample_width,
    source_width), aspect preserved) — NOT on the native buffer frames.
    Scale the box back into native pixel space; identical factor on both
    axes. A no-op when the source was already narrower than sample_width.
    Returns None on degenerate input."""
    _ = frame_h  # aspect preserved — one factor serves both axes
    ew = min(float(sample_width), float(frame_w))
    if ew <= 0 or frame_w <= 0:
        return None
    s = float(frame_w) / ew
    x0, y0, x1, y1 = bbox
    return (x0 * s, y0 * s, x1 * s, y1 * s)
