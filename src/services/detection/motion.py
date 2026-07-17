"""ALL backend motion math in one module — the map of every motion
signal in the system, in plain English (consolidation, 2026-07-17).

The system has exactly FOUR motion signals. Who computes what, and who
is allowed to spend money because of it:

1. GATEWAY TRIPWIRE — on the box (nemara-gateway publisher.py).
   Samples scattered pixels, counts big per-pixel jumps, sends a
   boolean `in_motion` packet about once a second. Cheap and blunt —
   and proven to false-fire all night on IR sensor grain (Nagpur,
   2026-07-16: 97.4% of night ticks tripped it; ≈$31/day of Gemini
   calls confirming an empty office was empty). It is a HINT, never
   an authority. Hardening (delta 30→45) ships separately on the
   gateway repo; nothing here depends on it.

2. MOTION HINT CACHE — this backend (motion_cache.py). The gateway
   boolean, cached per camera. Consumers: the closed-state wake check
   (now verified by #3 before it can cost money), the data-collection
   gate, and `analyses.motion_in_clip` stamping.

3. BLOCK MOTION — this module. THE BACKEND AUTHORITY on "did the
   scene actually move." The frame is averaged into a coarse grid of
   blocks; real motion (person, pet, curtain, door) changes one or a
   few ADJACENT blocks a lot, while sensor grain — uncorrelated pixel
   noise — averages itself to ~zero inside a block (mean over ~3,600
   pixels shrinks noise ~60×). A per-block habituation baseline (EMA
   of each block's normal wiggle) additionally silences STATIONARY
   PERIODIC sources — a blinking PC LED, a flickering standby light —
   which are locally coherent but always in the same block: they
   raise their own baseline within seconds and stop counting, while a
   person entering hits FRESH blocks and always trips. Wake authority:
   camera_ingest's closed-state path calls `block_motion()` when the
   hint says "wake" and stays closed (free) when this says grain.

4. MOTION OVERLAY — motion_overlay.py. The red tint drawn on the
   VLM's clip so rapid-motion detectors can SEE small sustained
   movement. Same blur/threshold/blob family as #3 but tuned for
   pixel-accurate tinting; intentionally separate because its output
   is an image for a model, not a boolean for a gate.

(The scattered-sample `motion_stats` fields stamped on analyzed rows
since the 2026-07-16 hotfix are observability for tuning #1's
thresholds from the DB — same statistic as the box computes, kept for
continuity. Detection-v2's per-track/env signals live on their branch
and should consume #3 as their env source when merged, so there stays
exactly ONE definition of scene motion.)

Calibration (Nagpur 550724ce, night 2026-07-16 → dawn 07-17, 852+169
stamped ticks): night grain ≈ block delta <1; LED flicker / dawn
ambient ≈ 1–3; lights-on scene change ≈ 40+; a walking person fills
blocks at 15–80. The absolute trip floor of 12 sits ≥4× above every
ambient reading and ≥3× below every real-event reading we measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Grid of blocks the luma plane is averaged into. 32×18 keeps a block
# ≈60×60px at 1080p — big enough to crush grain (~60× noise
# suppression), small enough that a cat-sized object still owns most
# of a block.
GRID_W = 32
GRID_H = 18

# A block must move at least this much (mean luma levels) to count.
# See calibration table in the module docstring.
BLOCK_DELTA_TRIP = 12.0

# ...and at least this multiple of its OWN recent baseline, so a
# stationary blinker (PC LED) that trips every second habituates
# itself out within ~half a minute while fresh motion always counts.
BASELINE_MULT = 4.0
# EMA weight per observation for the per-block baseline.
BASELINE_ALPHA = 0.08


@dataclass(frozen=True)
class BlockMotion:
    """One window's verdict."""

    coherent: bool          # something locally-real moved
    max_block_delta: float  # strongest single-block change seen
    active_blocks: int      # blocks over their trip line (max per pair)


class BlockMotionTracker:
    """Per-camera block-motion authority with habituation memory.

    Owned by the camera's ingest instance; `assess(window)` is pure
    numpy over the already-buffered frames (sub-millisecond for a
    dozen frames). Baselines live in memory only — a restart just
    means one slightly trigger-happy first window, which fails toward
    analysis (safe direction).
    """

    def __init__(self) -> None:
        self._baseline: np.ndarray | None = None  # (GRID_H, GRID_W) EMA of |Δ|

    @staticmethod
    def _grid(frame) -> np.ndarray | None:
        n = frame.width * frame.height
        if n <= 0 or len(frame.data) < n:
            return None
        bh, bw = frame.height // GRID_H, frame.width // GRID_W
        if bh == 0 or bw == 0:
            return None
        y = np.frombuffer(frame.data, dtype=np.uint8, count=n).reshape(
            frame.height, frame.width,
        )
        return (
            y[: bh * GRID_H, : bw * GRID_W]
            .reshape(GRID_H, bh, GRID_W, bw)
            .mean(axis=(1, 3))
        )

    def assess(self, window: Sequence) -> BlockMotion:  # noqa: D102 — class doc covers it
        max_d = 0.0
        active = 0
        prev: np.ndarray | None = None
        for f in window:
            g = self._grid(f)
            if g is None:
                prev = None
                continue
            if prev is not None and g.shape == prev.shape:
                d = np.abs(g - prev)
                if self._baseline is None or self._baseline.shape != d.shape:
                    # Cold start: zero baseline ⇒ the absolute floor
                    # alone judges the first window (trigger-happy,
                    # i.e. fails TOWARD analysis — never veto a real
                    # wake because we just restarted).
                    self._baseline = np.zeros_like(d)
                # A block trips only when it moves BOTH beyond the
                # absolute floor AND well beyond its own recent normal.
                trip = (d >= BLOCK_DELTA_TRIP) & (
                    d >= BASELINE_MULT * self._baseline
                )
                active = max(active, int(trip.sum()))
                max_d = max(max_d, float(d.max()))
                # Habituate AFTER judging, so a novel source counts
                # before it starts raising its own bar.
                self._baseline += BASELINE_ALPHA * (d - self._baseline)
            prev = g
        return BlockMotion(
            coherent=active > 0,
            max_block_delta=round(max_d, 2),
            active_blocks=active,
        )


def sample_stats(window: Sequence) -> dict:
    """Observability twin of the GATEWAY tripwire (#1 in the module
    doc): the fraction of scattered luma samples (every 397th pixel,
    matching the box) whose frame-to-frame delta exceeds 30/45/60,
    plus the mean |delta|. Stamped on analyzed ticks so gateway
    thresholds keep being tunable from the DB (originally the
    2026-07-16 night-noise hotfix). Observation only — nothing gates
    on these; the gate is BlockMotionTracker."""
    f30 = f45 = f60 = 0.0
    means: list[float] = []
    prev = None
    for f in window:
        n = f.width * f.height
        if n <= 0 or len(f.data) < n:
            prev = None
            continue
        y = np.frombuffer(f.data, dtype=np.uint8, count=n)[::397].astype(np.int16)
        if prev is not None and y.shape == prev.shape:
            d = np.abs(y - prev)
            size = d.size or 1
            f30 = max(f30, float((d > 30).sum()) / size)
            f45 = max(f45, float((d > 45).sum()) / size)
            f60 = max(f60, float((d > 60).sum()) / size)
            means.append(float(d.mean()))
        prev = y
    return {
        "motion_frac30_max": round(f30, 5),
        "motion_frac45_max": round(f45, 5),
        "motion_frac60_max": round(f60, 5),
        "motion_mean_abs": round(sum(means) / len(means), 3) if means else 0.0,
    }
