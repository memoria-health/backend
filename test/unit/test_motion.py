"""Unit: consolidated backend motion authority (motion.py).

Pins the four behaviors the wake veto depends on:
  - uncorrelated sensor grain (the Nagpur night leak) reads STILL
  - a locally-coherent blob (person/pet/curtain scale) trips
  - a stationary blinker (PC LED) trips at first, then habituates out
  - a MOVING object keeps tripping (fresh blocks have low baselines)
plus the observability twin (sample_stats) and cold-start direction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.services.detection.motion import (
    BLOCK_DELTA_TRIP,
    BlockMotionTracker,
    sample_stats,
)

W, H = 640, 360


def _frame(y: np.ndarray) -> SimpleNamespace:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return SimpleNamespace(width=W, height=H, data=y.tobytes() + uv.tobytes())


def _grain(seed: int, sigma: float = 12.0) -> np.ndarray:
    """IR-night-style uncorrelated noise around a flat scene."""
    rng = np.random.default_rng(seed)
    base = np.full(W * H, 60.0)
    return np.clip(base + rng.normal(0, sigma, W * H), 0, 255).astype(np.uint8)


def _with_blob(y: np.ndarray, x0: int, bright: int = 200) -> np.ndarray:
    """Paint a ~40x40px coherent object at column x0."""
    y2 = y.reshape(H, W).copy()
    y2[150:190, x0:x0 + 40] = bright
    return y2.reshape(-1)


def test_night_grain_reads_still() -> None:
    t = BlockMotionTracker()
    out = t.assess([_frame(_grain(i)) for i in range(8)])
    assert out.coherent is False
    assert out.max_block_delta < BLOCK_DELTA_TRIP / 2   # huge margin


def test_coherent_blob_trips_even_over_grain() -> None:
    t = BlockMotionTracker()
    frames = [
        _frame(_grain(0)),
        _frame(_with_blob(_grain(1), 300)),              # object appears
    ]
    out = t.assess(frames)
    assert out.coherent is True
    assert out.active_blocks >= 1


def _with_led(y: np.ndarray, x0: int = 300) -> np.ndarray:
    """An LED-scale blinker: ~8x8px inside one 20x20 block (block-mean
    delta ≈ 22 — over the trip floor, under the habituation cap's
    guaranteed-trip line)."""
    y2 = y.reshape(H, W).copy()
    y2[160:168, x0:x0 + 8] = 200
    return y2.reshape(-1)


def test_stationary_blinker_habituates_out() -> None:
    t = BlockMotionTracker()
    on = _frame(_with_led(_grain(0, 2.0)))
    off = _frame(_grain(1, 2.0))
    # First look: novel blinker counts (wake is correct — we don't know
    # it's a lamp yet).
    assert t.assess([off, on]).coherent is True
    # A minute of blinking later, its block's baseline has risen to the
    # blink amplitude and it stops counting.
    for _ in range(40):
        t.assess([off, on, off])
    assert t.assess([off, on, off]).coherent is False


def test_habituation_cap_never_blinds_a_block() -> None:
    # Saturate one block's baseline with a strong repeating source,
    # then hit the SAME block with person-scale motion (delta ≥
    # BASELINE_MULT × BASELINE_CAP = 80): must ALWAYS trip — the cap
    # guarantees habituation quiets, never immunizes (review fix
    # 2026-07-17).
    t = BlockMotionTracker()
    on = _frame(_with_led(_grain(0, 2.0)))
    off = _frame(_grain(1, 2.0))
    for _ in range(60):
        t.assess([off, on, off])
    big = _grain(2, 2.0).reshape(H, W).copy()
    big[160:180, 300:320] = 250                     # full block, delta ~190
    assert t.assess([off, _frame(big.reshape(-1))]).coherent is True


def test_moving_object_keeps_tripping_after_habituation() -> None:
    t = BlockMotionTracker()
    on = _frame(_with_blob(_grain(0, 2.0), 300))
    off = _frame(_grain(1, 2.0))
    for _ in range(40):
        t.assess([off, on, off])                        # habituate col 300
    # Same object now moves across FRESH blocks — must trip.
    moving = [_frame(_with_blob(_grain(2, 2.0), x)) for x in (80, 140, 200)]
    assert t.assess([off, *moving]).coherent is True


def test_cold_start_fails_toward_analysis() -> None:
    # First-ever window with real motion must trip (zero baseline —
    # never veto a real wake because the worker just restarted).
    t = BlockMotionTracker()
    out = t.assess([_frame(_grain(0, 2.0)), _frame(_with_blob(_grain(1, 2.0), 300))])
    assert out.coherent is True


def test_sample_stats_shape_and_grain_vs_blob() -> None:
    still = [_frame(_grain(i, 2.0)) for i in range(4)]
    s = sample_stats(still)
    assert set(s) == {
        "motion_frac30_max", "motion_frac45_max",
        "motion_frac60_max", "motion_mean_abs",
    }
    assert s["motion_frac45_max"] < 0.0025
    blob = sample_stats([_frame(_grain(0, 2.0)), _frame(_with_blob(_grain(1, 2.0), 300))])
    assert blob["motion_frac45_max"] > s["motion_frac45_max"]
