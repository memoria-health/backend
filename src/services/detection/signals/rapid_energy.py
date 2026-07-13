"""RAPID pre-gate signal (detection v2, P0).

Decides — for free, no VLM — whether a frame window plausibly contains
sustained IN-PLACE rapid motion (convulsive shaking) and therefore whether
the expensive RAPID call (8 fps person-crop overlay on the flash-tier
model) should run this tick.

Discriminator: churn CONCENTRATION, not churn amount. Convulsion-like
motion makes the SAME pixels flip in most consecutive frame diffs;
translating motion (walking — the dominant motion tick) touches each
pixel in only a couple of diffs; stillness touches none. So:

    score           = |pixels active in >= persist_ratio of pair-diffs|
                      -------------------------------------------------
                      |pixels active in >= 1 pair-diff|
    active_fraction = |pixels active in >= 1 pair-diff| / |pixels|

The gate opens only when score AND active_fraction clear their floors —
the floor rejects specks (status LEDs, sensor noise) that concentrate
perfectly but cover nothing. This is a COST LEVER, not a classifier:
false-opens (e.g. a TV's in-place flicker until person-box scoping lands
in P2) just buy a RAPID call that answers "no"; a false-close is the only
real failure, so thresholds are tuned permissive and re-validated against
the labeled convulsion eval set before the flag defaults on.

Works on the Y plane of the buffered I420 frames only (no color
conversion), downscaled for speed; `region` scopes to a person box in
native pixel coordinates when the caller has one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from src.services.detection.frame_buffer import FrameSnapshot

# Per-pixel |ΔY| above this counts as "active" in a pair-diff (post-blur).
_DIFF_THRESHOLD = 20
# A pixel is "persistent churn" when active in at least this ratio of diffs.
_PERSIST_RATIO = 0.5
# Y planes wider than this are downscaled (INTER_AREA) before diffing.
_DOWNSCALE_WIDTH = 160
# Below this many frames the window can't establish rhythm at any fps.
_MIN_FRAMES = 4

# Gate floors — deliberately permissive (see module docstring); the
# settings wiring passes overrides so they can be tuned without a deploy.
MIN_SCORE_DEFAULT = 0.30
MIN_ACTIVE_FRACTION_DEFAULT = 0.004


@dataclass(frozen=True)
class RapidEnergy:
    """Churn concentration (0..1) and coverage (0..1) for one window."""

    score: float
    active_fraction: float


def _prepared_y(
    frame: FrameSnapshot, region: tuple[int, int, int, int] | None
) -> np.ndarray | None:
    """Y plane → optional region crop → downscale → denoise blur.
    Returns None when the clamped region is empty."""
    y = np.frombuffer(
        frame.data, dtype=np.uint8, count=frame.width * frame.height
    ).reshape(frame.height, frame.width)
    if region is not None:
        x0, y0, x1, y1 = region
        x0 = max(0, min(int(x0), frame.width))
        x1 = max(0, min(int(x1), frame.width))
        y0 = max(0, min(int(y0), frame.height))
        y1 = max(0, min(int(y1), frame.height))
        if x1 <= x0 or y1 <= y0:
            return None
        y = y[y0:y1, x0:x1]
    if y.shape[1] > _DOWNSCALE_WIDTH:
        scale = _DOWNSCALE_WIDTH / y.shape[1]
        y = cv2.resize(
            y,
            (_DOWNSCALE_WIDTH, max(1, int(round(y.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.GaussianBlur(y, (3, 3), 0)


def rapid_energy(
    frames: Sequence[FrameSnapshot],
    *,
    region: tuple[int, int, int, int] | None = None,
    diff_threshold: int = _DIFF_THRESHOLD,
    persist_ratio: float = _PERSIST_RATIO,
) -> RapidEnergy:
    """Measure churn concentration over a window of I420 frames."""
    if len(frames) < _MIN_FRAMES:
        return RapidEnergy(score=0.0, active_fraction=0.0)
    # Publisher renegotiation can change resolution mid-window; keep only
    # the dominant shape so pair-diffs always align.
    shapes: dict[tuple[int, int], int] = {}
    for f in frames:
        shapes[(f.width, f.height)] = shapes.get((f.width, f.height), 0) + 1
    keep = max(shapes, key=shapes.get)  # type: ignore[arg-type]
    planes = [
        p
        for f in frames
        if (f.width, f.height) == keep
        and (p := _prepared_y(f, region)) is not None
    ]
    if len(planes) < _MIN_FRAMES:
        return RapidEnergy(score=0.0, active_fraction=0.0)

    counts = np.zeros(planes[0].shape, dtype=np.int32)
    n_pairs = 0
    for a, b in zip(planes, planes[1:]):
        counts += cv2.absdiff(a, b) > diff_threshold
        n_pairs += 1
    active = counts > 0
    n_active = int(active.sum())
    active_fraction = n_active / counts.size
    if n_active == 0:
        return RapidEnergy(score=0.0, active_fraction=0.0)
    need = max(2, int(round(n_pairs * persist_ratio)))
    persistent = int((counts >= need).sum())
    return RapidEnergy(
        score=persistent / n_active, active_fraction=float(active_fraction)
    )


def rapid_gate(
    frames: Sequence[FrameSnapshot],
    *,
    region: tuple[int, int, int, int] | None = None,
    min_score: float = MIN_SCORE_DEFAULT,
    min_active_fraction: float = MIN_ACTIVE_FRACTION_DEFAULT,
) -> bool:
    """True when the window warrants spending a RAPID call."""
    e = rapid_energy(frames, region=region)
    return e.score >= min_score and e.active_fraction >= min_active_fraction
