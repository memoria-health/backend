"""Unit tests for `services/ingest/pi/motion_detector.py`.

The motion detector is pure numerical code over numpy arrays. Tests
construct synthetic Y-plane frames (uint8 2D arrays) directly without
involving any JPEG decoder, so the suite has no image-library
dependency.

Two layers of assertions:

  1. Algorithmic correctness over synthetic data: a still scene
     produces no motion; a moving rectangle does; the rising-edge
     event fires only after `consecutive_frames_required` frames;
     the history ring uses the correct baseline lag.

  2. Numerical invariants: max-tile mean SAD >= whole-frame mean
     SAD; values are clipped to [0, 255]; max-tile catches
     concentrated motion that whole-frame mean misses.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.ingest.pi.motion_detector import (
    MotionConfig,
    MotionDetector,
    MotionResult,
    _bilinear_downsample,
    _compute_sad_metrics,
)

# ─── helpers ─────────────────────────────────────────────────────


def _still_frame(side: int = 128, *, fill: int = 128) -> np.ndarray:
    """Uniform-grey Y plane at the given side length."""
    return np.full((side, side), fill, dtype=np.uint8)


def _with_block(
    base: np.ndarray, *, top: int, left: int, size: int, fill: int
) -> np.ndarray:
    """Return a copy of `base` with a square block painted in."""
    out = base.copy()
    out[top : top + size, left : left + size] = fill
    return out


# ─── MotionConfig validation ─────────────────────────────────────


def test_config_defaults_match_ios_values() -> None:
    """The tuned defaults are the load-bearing contract with the iOS
    side. A future edit that bumps these without coordinating both
    paths should require an explicit test change.
    """
    c = MotionConfig()
    assert c.threshold == 1.5
    assert c.consecutive_frames_required == 2
    assert c.baseline_frame_lag == 9
    assert c.tile_side == 8
    assert c.ds_side == 64


def test_config_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be positive"):
        MotionConfig(threshold=0)
    with pytest.raises(ValueError, match="threshold must be positive"):
        MotionConfig(threshold=-1.0)


def test_config_rejects_invalid_consecutive() -> None:
    with pytest.raises(ValueError, match="consecutive_frames_required"):
        MotionConfig(consecutive_frames_required=0)


def test_config_rejects_invalid_baseline_lag() -> None:
    with pytest.raises(ValueError, match="baseline_frame_lag"):
        MotionConfig(baseline_frame_lag=0)


def test_config_rejects_non_divisible_tile() -> None:
    """`tile_side` must evenly divide `ds_side` or the reshape-based
    tile sum would silently truncate.
    """
    with pytest.raises(ValueError, match="must evenly divide"):
        MotionConfig(ds_side=64, tile_side=7)


def test_config_rejects_tile_equal_to_ds() -> None:
    """`tile_side == ds_side` would technically pass the modulo
    check but collapse the tile grid to 1x1, making max-tile equal
    to mean-frame and defeating the tile-discrimination algorithm.
    """
    with pytest.raises(ValueError, match="strictly less than"):
        MotionConfig(ds_side=64, tile_side=64)


def test_config_rejects_threshold_above_255() -> None:
    """Per-tile mean SAD on uint8 data is bounded to [0, 255]. A
    threshold above 255 can never be reached and would silently
    disable detection.
    """
    with pytest.raises(ValueError, match="cannot exceed 255"):
        MotionConfig(threshold=256)
    with pytest.raises(ValueError, match="cannot exceed 255"):
        MotionConfig(threshold=1000)


# ─── input validation on process() ───────────────────────────────


def test_process_rejects_non_2d_input() -> None:
    det = MotionDetector()
    with pytest.raises(ValueError, match="must be a 2D array"):
        det.process(np.zeros((64, 64, 3), dtype=np.uint8))


def test_process_rejects_wrong_dtype() -> None:
    det = MotionDetector()
    with pytest.raises(ValueError, match="must be dtype uint8"):
        det.process(np.zeros((128, 128), dtype=np.float32))


def test_process_rejects_input_smaller_than_ds_side() -> None:
    det = MotionDetector()
    with pytest.raises(ValueError, match="smaller than ds_side"):
        det.process(np.zeros((32, 32), dtype=np.uint8))


# ─── history ring behavior ───────────────────────────────────────


def test_no_motion_during_history_warmup() -> None:
    """The first `baseline_frame_lag` frames have no baseline to
    compare against. Motion must not be reported during this period
    even with synthetic "motion" frames.
    """
    config = MotionConfig()  # baseline_frame_lag=9
    det = MotionDetector(config)
    still = _still_frame()
    moving = _with_block(still, top=10, left=10, size=20, fill=255)

    for i in range(config.baseline_frame_lag):
        # Alternate to deliberately introduce visible "motion" if the
        # detector were broken and compared against the immediately
        # previous frame.
        frame = moving if i % 2 == 0 else still
        result = det.process(frame)
        assert result.in_motion is False, f"warmup frame {i} fired motion"
        assert result.frames_ingested == i + 1


def test_still_scene_reports_no_motion_after_warmup() -> None:
    config = MotionConfig()
    det = MotionDetector(config)
    still = _still_frame()
    # Run for warmup + 10 more frames.
    for _ in range(config.baseline_frame_lag + 10):
        result = det.process(still)
    assert result.in_motion is False
    assert result.max_tile_sad == 0.0
    assert result.mean_sad == 0.0
    assert result.consecutive_count == 0


def test_consecutive_motion_fires_event_on_exact_frame() -> None:
    """A sustained over-threshold scene must fire `in_motion=True`
    on exactly the consecutive-frames-required-th frame after warmup,
    and return False on subsequent frames (rising-edge semantics).
    """
    config = MotionConfig(
        consecutive_frames_required=2, baseline_frame_lag=3, threshold=1.5
    )
    det = MotionDetector(config)

    # Fill the history with stills.
    still = _still_frame()
    for _ in range(config.baseline_frame_lag):
        det.process(still)

    # Switch to a clearly moving scene. The block is large enough
    # that the max-tile mean SAD is well above threshold.
    moving = _with_block(still, top=10, left=10, size=40, fill=255)

    first = det.process(moving)
    second = det.process(moving)
    third = det.process(moving)

    assert first.in_motion is False
    assert first.consecutive_count == 1
    assert first.max_tile_sad > config.threshold

    # Second frame crosses the consecutive threshold.
    assert second.in_motion is True
    assert second.consecutive_count == 2

    # Sustained motion is not re-fired on every subsequent frame.
    # (Rising-edge: only the exact threshold-crossing frame fires.)
    assert third.in_motion is False
    assert third.consecutive_count == 3


def test_streak_resets_when_motion_falls_below_threshold() -> None:
    config = MotionConfig(
        consecutive_frames_required=3, baseline_frame_lag=3, threshold=5.0
    )
    det = MotionDetector(config)
    still = _still_frame()
    for _ in range(config.baseline_frame_lag):
        det.process(still)

    moving = _with_block(still, top=10, left=10, size=40, fill=255)
    det.process(moving)  # streak=1
    result = det.process(moving)  # streak=2
    assert result.consecutive_count == 2
    # Return to still; streak resets to 0.
    result = det.process(still)
    assert result.consecutive_count == 0
    assert result.in_motion is False


# ─── tile vs whole-frame discrimination (the load-bearing insight) ──


def test_max_tile_catches_localized_motion_when_whole_frame_mean_does_not() -> None:
    """The key insight from the iOS port: a single tile of motion
    must light up max-tile SAD even when the whole-frame mean SAD
    is too low to trigger anything. Without this, distant subjects
    would never be detected.

    Build a 128x128 frame with a single bright pixel changing. The
    whole-frame mean SAD over 16384 pixels is tiny (~0.02), but the
    pixel falls in one downsampled tile where it dominates that
    tile's mean SAD.
    """
    config = MotionConfig(
        consecutive_frames_required=1, baseline_frame_lag=2, threshold=0.5
    )
    det = MotionDetector(config)
    still = _still_frame(side=128, fill=128)

    # Warm up.
    for _ in range(config.baseline_frame_lag):
        det.process(still)

    # Add a 4x4 bright patch in one quadrant. After downsample to
    # 64x64, this is ~2x2 bright pixels concentrated in one tile.
    moving = _with_block(still, top=40, left=40, size=4, fill=255)
    result = det.process(moving)

    # max_tile_sad must be > mean_sad (the inequality is the whole
    # point of the algorithm).
    assert result.max_tile_sad > result.mean_sad
    # And specifically, max_tile_sad must clear the threshold even
    # though the whole-frame mean is well below it.
    assert result.mean_sad < config.threshold
    assert result.max_tile_sad >= config.threshold


def test_full_frame_brightness_change_triggers_both_metrics() -> None:
    """Uniform brightness shift must light up BOTH whole-frame and
    max-tile SAD (every tile sees the same delta).
    """
    config = MotionConfig(
        consecutive_frames_required=1, baseline_frame_lag=2, threshold=5.0
    )
    det = MotionDetector(config)
    dark = _still_frame(side=128, fill=64)
    for _ in range(config.baseline_frame_lag):
        det.process(dark)
    bright = _still_frame(side=128, fill=192)
    result = det.process(bright)
    assert result.max_tile_sad == pytest.approx(result.mean_sad, abs=2.0)
    # Both should be near the brightness delta (192 - 64 = 128).
    assert result.mean_sad > 120
    assert result.in_motion is True


# ─── baseline lag wiring ─────────────────────────────────────────


def test_baseline_lag_compares_against_correct_history_slot() -> None:
    """With baseline_frame_lag=3, frame N is compared against frame N-3,
    NOT frame N-1. We exploit this by feeding a sequence that produces
    different SADs depending on which baseline is used.
    """
    config = MotionConfig(
        consecutive_frames_required=1, baseline_frame_lag=3, threshold=0.5
    )
    det = MotionDetector(config)

    dark = _still_frame(side=128, fill=64)
    bright = _still_frame(side=128, fill=192)

    # History sequence: bright, bright, bright (warmup), then bright again.
    # If the detector compared against frame N-1 (lag=1), every
    # comparison would be 0 (bright vs bright). With lag=3, the
    # FIRST post-warmup frame compares against the FIRST warmup
    # frame, also bright → no motion. Then introduce a single
    # changed frame and verify it only triggers motion 3 frames
    # later (when it BECOMES the baseline).
    for _ in range(config.baseline_frame_lag):
        det.process(bright)

    # Frame at t=lag: bright again. Compared against frame at t=0
    # which was also bright → no motion.
    r0 = det.process(bright)
    assert r0.in_motion is False

    # Switch to dark. This frame at t=lag+1 compared against bright
    # frame at t=1 → high SAD, motion.
    r1 = det.process(dark)
    assert r1.in_motion is True
    assert r1.max_tile_sad > config.threshold


def test_history_ring_does_not_grow_unboundedly() -> None:
    """The history ring is a fixed-size numpy array. After processing
    many frames, its shape must still be (baseline_frame_lag+1,
    ds_side, ds_side).
    """
    config = MotionConfig()
    det = MotionDetector(config)
    still = _still_frame()
    for _ in range(1000):
        det.process(still)
    # Access private state for the invariant check.
    assert det._history.shape == (
        config.baseline_frame_lag + 1,
        config.ds_side,
        config.ds_side,
    )


# ─── reset behavior ──────────────────────────────────────────────


def test_reset_clears_history_and_streak() -> None:
    config = MotionConfig(
        consecutive_frames_required=1, baseline_frame_lag=2, threshold=5.0
    )
    det = MotionDetector(config)
    dark = _still_frame(side=128, fill=64)
    bright = _still_frame(side=128, fill=192)
    for _ in range(config.baseline_frame_lag):
        det.process(dark)
    r = det.process(bright)
    assert r.in_motion is True

    det.reset()
    # First frame after reset is the start of a new warmup; no motion
    # possible until baseline_frame_lag more frames are ingested.
    r = det.process(bright)
    assert r.in_motion is False
    assert r.frames_ingested == 1
    assert r.consecutive_count == 0


def test_reset_zeros_history_array_in_place() -> None:
    """After `reset()`, the internal history ring must be zeroed.
    The warmup gate prevents observing this via the public API
    (any post-reset comparison can only happen after the ring is
    re-filled with new frames), so we assert on the private state
    directly. This catches a regression where `reset()` cleared
    counters but forgot to zero the buffer; a subsequent session
    that did not fully re-warm before any comparison could read
    stale data.
    """
    config = MotionConfig()
    det = MotionDetector(config)
    bright = _still_frame(side=128, fill=200)
    for _ in range(config.baseline_frame_lag + 3):
        det.process(bright)
    # Sanity: history has been populated.
    assert det._history.any(), "history should be non-zero before reset"

    det.reset()

    # Every byte of the history ring must be zero after reset.
    assert not det._history.any(), (
        "reset() did not zero the history ring; stale frames could "
        "be reused as baselines on the next session"
    )
    # Counters too.
    assert det._history_head == 0
    assert det._frames_ingested == 0
    assert det._consecutive_count == 0


def test_process_works_with_source_exactly_ds_side() -> None:
    """End-to-end behavior when the source frame size equals the
    internal downsample target. The bilinear path becomes an
    effective identity; SAD math must still produce the correct
    metrics.
    """
    config = MotionConfig(
        consecutive_frames_required=1, baseline_frame_lag=2, threshold=5.0
    )
    det = MotionDetector(config)
    dark = _still_frame(side=64, fill=64)
    for _ in range(config.baseline_frame_lag):
        det.process(dark)
    bright = _still_frame(side=64, fill=192)
    r = det.process(bright)
    # Bright vs dark, uniform across the frame, gives ~128 SAD
    # (192 - 64) on both mean and max-tile.
    assert r.mean_sad == pytest.approx(128.0, abs=1.0)
    assert r.max_tile_sad == pytest.approx(128.0, abs=1.0)
    assert r.in_motion is True


def test_process_works_with_non_square_source() -> None:
    """720p style: 720 rows by 1280 cols, the actual Pi resolution.
    Aspect ratio is squashed into the square 64x64 downsample (same
    as iOS); SAD math is still valid because the squashing is
    consistent across frames.
    """
    config = MotionConfig(
        consecutive_frames_required=1, baseline_frame_lag=2, threshold=5.0
    )
    det = MotionDetector(config)
    dark = np.full((720, 1280), 64, dtype=np.uint8)
    for _ in range(config.baseline_frame_lag):
        det.process(dark)
    bright = np.full((720, 1280), 192, dtype=np.uint8)
    r = det.process(bright)
    assert r.mean_sad == pytest.approx(128.0, abs=2.0)
    assert r.max_tile_sad == pytest.approx(128.0, abs=2.0)
    assert r.in_motion is True


# ─── _compute_sad_metrics direct ─────────────────────────────────


def test_compute_sad_metrics_zero_diff_returns_zero() -> None:
    a = np.full((64, 64), 128, dtype=np.uint8)
    mean, mx = _compute_sad_metrics(a, a, tile_side=8)
    assert mean == 0.0
    assert mx == 0.0


def test_compute_sad_metrics_uniform_diff_equals_difference() -> None:
    """When EVERY pixel differs by a constant `delta`, both mean_sad
    and max_tile_sad must equal `delta`.
    """
    a = np.full((64, 64), 100, dtype=np.uint8)
    b = np.full((64, 64), 150, dtype=np.uint8)
    mean, mx = _compute_sad_metrics(a, b, tile_side=8)
    assert mean == pytest.approx(50.0)
    assert mx == pytest.approx(50.0)


def test_compute_sad_metrics_handles_uint8_wraparound() -> None:
    """abs(255 - 0) must be 255, not 1 (which would happen if the
    implementation accidentally used uint8 subtraction).
    """
    a = np.full((64, 64), 255, dtype=np.uint8)
    b = np.full((64, 64), 0, dtype=np.uint8)
    mean, mx = _compute_sad_metrics(a, b, tile_side=8)
    assert mean == pytest.approx(255.0)
    assert mx == pytest.approx(255.0)


def test_compute_sad_metrics_localized_diff_amplified_by_max_tile() -> None:
    """One bright tile against an all-dark baseline. The whole-frame
    mean SAD divides 255*64 by 4096 = ~3.98. The max-tile mean SAD
    over the bright tile is 255 (all of its pixels differ by 255).
    """
    a = np.full((64, 64), 0, dtype=np.uint8)
    b = a.copy()
    b[0:8, 0:8] = 255  # one tile fully changed
    mean, mx = _compute_sad_metrics(a, b, tile_side=8)
    assert mean == pytest.approx(255.0 * 64 / 4096)
    assert mx == pytest.approx(255.0)


def test_max_tile_sad_never_less_than_mean_sad() -> None:
    """Invariant: the max over tiles cannot be below the mean across
    all tiles. Property test with random uint8 frames.
    """
    rng = np.random.default_rng(seed=42)
    for _ in range(20):
        a = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        b = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        mean, mx = _compute_sad_metrics(a, b, tile_side=8)
        assert mx >= mean - 1e-9  # tolerate float rounding


# ─── _bilinear_downsample direct ─────────────────────────────────


def test_bilinear_downsample_returns_target_shape() -> None:
    src = np.zeros((100, 200), dtype=np.uint8)
    out = _bilinear_downsample(src, target_size=64)
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_bilinear_downsample_preserves_constant_input() -> None:
    src = np.full((100, 200), 137, dtype=np.uint8)
    out = _bilinear_downsample(src, target_size=64)
    # Bilinear of a constant is the same constant everywhere.
    assert (out == 137).all()


def test_bilinear_downsample_handles_size_equal_to_target() -> None:
    """When source size already equals target, every output pixel
    should match the input exactly (within rounding).
    """
    rng = np.random.default_rng(seed=0)
    src = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    out = _bilinear_downsample(src, target_size=64)
    # Within ±1 of source due to clip+astype rounding.
    diff = np.abs(out.astype(np.int16) - src.astype(np.int16))
    assert diff.max() <= 1


def test_bilinear_downsample_output_values_stay_in_uint8_range() -> None:
    """Floating-point interpolation must not produce out-of-range
    uint8 values after the clip+cast.
    """
    rng = np.random.default_rng(seed=1)
    src = rng.integers(0, 256, size=(720, 1280), dtype=np.uint8)
    out = _bilinear_downsample(src, target_size=64)
    assert out.min() >= 0
    assert out.max() <= 255


# ─── MotionResult shape ──────────────────────────────────────────


def test_motion_result_is_frozen_dataclass() -> None:
    """MotionResult is intended to be a value type. A mutable result
    would be a source of action-at-a-distance bugs in the relay.
    """
    r = MotionResult(
        in_motion=True,
        mean_sad=1.0,
        max_tile_sad=2.0,
        consecutive_count=3,
        frames_ingested=4,
    )
    with pytest.raises(AttributeError):
        # FrozenInstanceError is a subclass of AttributeError in
        # Python 3.13+. Catching the parent keeps the assertion
        # stable across point releases.
        r.in_motion = False  # type: ignore[misc]
