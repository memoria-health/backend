"""RAPID seizure call (detection v2, P0) — flag-gated by
`NEMARA_RAPID_SEIZURE_ENABLED`.

While the flag is on, the seizure detector leaves the main lite call
(`split_rapid`) and runs as its own dedicated call on the validated RAPID
recipe: churn pre-gate → overlay-tinted static person crops at native fps
(`rapid_artifact`) → flash-tier Gemini with a neutral, amplitude-forced
prompt. Measured: convulsion 8/10 recall, 1/30 FP; the lite tier is blind
to this motion class at any packaging (see
docs/architecture/detection-architecture-v2.md §0 law 2, §1.4).

The returned fields use the SAME `seizure_*` keys the main call used to
produce, so promotion / severity / dedup / clips are untouched downstream.
`filter_rapid_fields` strips everything else (people_present, speech, …)
so the RAPID call can never overwrite the main call's context fields.

The provider is injected (any callable with `gemini_provider`'s keyword
signature returning an object with `.detector_fields`) so the whole path
unit-tests with a fake — no network, no model.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Mapping, Sequence

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.prompt import DetectorSpec
from src.services.detection.rapid_artifact import (
    Box,
    build_rapid_artifacts,
    union_box,
)
from src.services.detection.signals.rapid_energy import (
    MIN_ACTIVE_FRACTION_DEFAULT,
    MIN_SCORE_DEFAULT,
    rapid_energy,
)
from src.settings import Settings

log = logging.getLogger("nemara.detection.rapid")

# Bound the per-tick spend: at most this many person artifacts get a call.
_MAX_ARTIFACTS = 2

# Only these field families may flow from the RAPID call into the tick's
# detector_fields — everything else belongs to the main call.
_KEEP_PREFIXES = ("seizure_", "rapid_")


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
    """The seizure spec for the RAPID call — same identifier (and therefore
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


def split_rapid(
    detectors: Sequence[DetectorSpec], settings: Settings
) -> tuple[list[DetectorSpec], bool]:
    """Remove seizure from the main call's detector list while the RAPID
    flag is on. Returns (main_detectors, rapid_active) — rapid_active is
    True only when the flag is on AND seizure is actually assigned here."""
    dets = list(detectors)
    if not settings.rapid_seizure_enabled:
        return dets, False
    main = [d for d in dets if d.identifier != "seizure"]
    return main, len(main) != len(dets)


def filter_rapid_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only the RAPID call's own field families."""
    return {k: v for k, v in fields.items() if k.startswith(_KEEP_PREFIXES)}


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


async def run_rapid_seizure(
    *,
    frames: Sequence[FrameSnapshot],
    per_person_boxes: Sequence[Sequence[Box]],
    settings: Settings,
    provider: Callable[..., Awaitable[Any]],
    log_tag: str,
) -> dict[str, Any] | None:
    """Pre-gate → artifacts → flash call(s). Returns the filtered seizure
    fields, or None when the gate stayed closed / nothing usable ran.
    Raises nothing on provider failure — logs and returns what it has."""
    if len(frames) < 4 or not per_person_boxes:
        return None
    fw, fh = frames[0].width, frames[0].height

    # Pre-gate = general person-scoped motion amount (churn score rides
    # along as a logged dial, floor disabled by default). Fail-open to the
    # whole frame if no box yields a usable region.
    regions = [union_box(b, fw, fh) for b in per_person_boxes]
    regions = [r for r in regions if r is not None]
    energies = (
        [rapid_energy(frames, region=r) for r in regions]
        if regions
        else [rapid_energy(frames)]
    )
    gate_af = max(e.active_fraction for e in energies)
    gate_score = max(e.score for e in energies)
    if not any(
        e.active_fraction >= MIN_ACTIVE_FRACTION_DEFAULT
        and e.score >= MIN_SCORE_DEFAULT
        for e in energies
    ):
        return None

    artifacts = await build_rapid_artifacts(
        frames, per_person_boxes, fps=settings.rapid_fps
    )
    if not artifacts:
        return None

    rapid_settings = settings.model_copy(update={
        "gemini_model": settings.rapid_model,
        "gemini_video_fps": str(settings.rapid_fps),
        "motion_overlay_enabled": True,   # the provider prepends the tint note
    })
    specs = [RAPID_MOTION_SPEC, rapid_seizure_spec()]
    best: dict[str, Any] | None = None
    in_tok = out_tok = 0
    attempted = succeeded = 0
    for art in artifacts[:_MAX_ARTIFACTS]:
        attempted += 1
        try:
            res = await provider(
                mp4=art, detectors=specs, log_tag=log_tag, settings=rapid_settings
            )
            succeeded += 1
        except Exception:  # noqa: BLE001 — one person's call failing must not
            log.exception("%s RAPID provider call failed", log_tag)
            continue
        usage = getattr(res, "usage", None)
        if usage:
            in_tok += int(usage.get("input", 0))
            out_tok += int(usage.get("output", 0))
        fields = filter_rapid_fields(res.detector_fields)
        if fields.get("seizure_detected"):
            best = fields
            break
        if best is None and fields:
            best = fields
    if attempted and not succeeded:
        # The gate OPENED — this tick genuinely wanted seizure coverage —
        # and every provider call failed (e.g. a model 503 storm). While
        # the flag is on, seizure lives ONLY here, so this must surface as
        # a degraded tick, not dissolve into "gate closed".
        raise RuntimeError(
            f"all {attempted} RAPID provider call(s) failed with the gate open"
        )
    if best is not None:
        # Gate readouts ride along for soak tuning (was the churn score
        # ever worth a floor? what does the activity distribution look
        # like on real ticks?).
        best["rapid_gate_active_fraction"] = round(gate_af, 4)
        best["rapid_gate_score"] = round(gate_score, 3)
        if in_tok or out_tok:
            best["rapid_input_tokens"] = in_tok
            best["rapid_output_tokens"] = out_tok
    return best
