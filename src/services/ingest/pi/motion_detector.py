"""Python port of the iOS `MotionDetector` (NemaraCam app).

Original lives at `nemara-camera-app/NemaraCam/Camera/MotionDetector.swift`.
This port preserves the SAD-tile algorithm and the tuned defaults so
both the iPhone and Pi paths produce comparable motion signals.

Algorithm

  1. Extract the Y (luminance) plane of the incoming frame.
  2. Downsample to `ds_side` x `ds_side` (default 64x64) using
     bilinear interpolation. The 64x64 working size is small enough
     for the SAD pass to be cheap and large enough that a distant
     subject still covers a few pixels.
  3. Split the 64x64 frame into a `tiles_per_side` x `tiles_per_side`
     grid of tiles (default 8x8 grid = 64 tiles of 8x8 px each).
  4. Compare the current frame against the frame `baseline_frame_lag`
     positions back in a small ring buffer. Per-tile SAD is the sum
     of absolute pixel differences. Per-tile mean SAD is that sum
     divided by tile pixel count.
  5. The motion decision metric is `max(per_tile_mean_sad)`. The
     whole-frame mean SAD is also returned for diagnostics.
  6. If the max-tile mean SAD exceeds `threshold` for
     `consecutive_frames_required` consecutive frames, the detector
     transitions to in_motion=True.

Why max-tile not mean-frame

A distant person who only affects a couple of pixels in the 64x64
grid still lights up one 8x8 tile clearly. Averaging across all
4096 pixels would bury their signal under the noise of the other
4094 still pixels. Max-tile catches concentrated motion at any
distance without lowering the threshold into sensor-noise territory.

Differences from the iOS implementation

  - The iOS version uses `vImageScale_Planar8` (Lanczos by default).
    This port uses pure-NumPy bilinear interpolation. The aggregate
    SAD statistics differ by at most a few percent on natural
    imagery; the calibrated threshold (1.5) is robust to that
    variance.
  - The iOS version maintains a timestamp ring of recorded motion
    events for UI telemetry. The Python path signals motion via
    LiveKit data packets from the relay layer (Phase 5), so this
    port returns per-frame results and does not own a timestamp
    history.

Threading

Not thread-safe. Expected to be driven by a single asyncio task
inside the Pi ingest worker. The relay decodes each JPEG once and
hands the Y-plane to `process`. Re-encoding for LiveKit happens
on the same task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionConfig:
    """Tunables for the SAD-tile motion detector.

    Defaults match the iOS implementation's tuned values:
    threshold=1.5 (per-tile mean SAD on the 0-255 scale),
    consecutive_frames_required=2 (confirm motion within ~133ms at
    15 fps), baseline_frame_lag=9 (~600ms baseline at 15 fps; long
    enough to catch slow walkers far from the camera).
    """

    threshold: float = 1.5
    consecutive_frames_required: int = 2
    baseline_frame_lag: int = 9
    tile_side: int = 8
    ds_side: int = 64

    def __post_init__(self) -> None:
        # threshold is a per-tile MEAN sad on the 0-255 scale, so any
        # value above 255 can never be exceeded and would permanently
        # silence the detector.
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")
        if self.threshold > 255:
            raise ValueError(
                "threshold cannot exceed 255 (max possible per-tile mean SAD)"
            )
        if self.consecutive_frames_required < 1:
            raise ValueError("consecutive_frames_required must be >= 1")
        if self.baseline_frame_lag < 1:
            raise ValueError("baseline_frame_lag must be >= 1")
        if self.tile_side < 1:
            raise ValueError("tile_side must be >= 1")
        if self.ds_side < 2:
            raise ValueError("ds_side must be >= 2")
        if self.ds_side % self.tile_side != 0:
            raise ValueError(
                f"tile_side ({self.tile_side}) must evenly divide "
                f"ds_side ({self.ds_side})"
            )
        # `tile_side == ds_side` would technically pass the modulo
        # check (the tile grid becomes 1x1) but it silently degrades
        # the algorithm to a whole-frame detector, defeating the
        # tile-based discrimination that catches localized motion.
        if self.tile_side >= self.ds_side:
            raise ValueError(
                f"tile_side ({self.tile_side}) must be strictly less than "
                f"ds_side ({self.ds_side}); otherwise the tile grid "
                f"collapses to 1x1 and max-tile becomes mean-frame"
            )


@dataclass(frozen=True)
class MotionResult:
    """Per-frame motion detector output.

    `in_motion` is True only on the exact frame the consecutive-frame
    threshold is crossed (rising edge). For "is the camera currently
    in motion" the relay should track the most recent True result
    and a TTL, matching how MotionCache consumes iOS data packets.
    """

    in_motion: bool
    mean_sad: float
    max_tile_sad: float
    consecutive_count: int
    frames_ingested: int


class MotionDetector:
    """Stateful per-camera motion detector.

    Holds a ring of recent downsampled frames so each new frame can
    be compared against the one `baseline_frame_lag` positions back.

    Usage:

        detector = MotionDetector()
        for frame in frames:                  # uint8 2D array, any size
            result = detector.process(frame)
            if result.in_motion:
                publish_motion_heartbeat()
    """

    def __init__(self, config: MotionConfig | None = None) -> None:
        self.config = config or MotionConfig()
        self._ring_capacity = self.config.baseline_frame_lag + 1
        self._history: np.ndarray = np.zeros(
            (self._ring_capacity, self.config.ds_side, self.config.ds_side),
            dtype=np.uint8,
        )
        self._history_head: int = 0
        self._frames_ingested: int = 0
        self._consecutive_count: int = 0

    def reset(self) -> None:
        """Drop all state. Used when a camera session restarts."""
        self._history_head = 0
        self._frames_ingested = 0
        self._consecutive_count = 0
        self._history.fill(0)

    def process(self, luminance: np.ndarray) -> MotionResult:
        """Process one frame. `luminance` is a 2D uint8 array (the Y
        plane of the source frame). Any resolution >= ds_side is
        accepted; the detector downsamples internally.

        Returns a `MotionResult` for the frame. While the history
        ring is still filling (the first `baseline_frame_lag`
        frames), motion is never reported, but the frame is still
        ingested and the result records its metrics.
        """
        if luminance.ndim != 2:
            raise ValueError(
                f"luminance must be a 2D array, got shape {luminance.shape!r}"
            )
        if luminance.dtype != np.uint8:
            raise ValueError(
                f"luminance must be dtype uint8, got {luminance.dtype}"
            )
        if luminance.shape[0] < self.config.ds_side or luminance.shape[1] < self.config.ds_side:
            raise ValueError(
                f"luminance shape {luminance.shape} is smaller than "
                f"ds_side {self.config.ds_side}"
            )

        ds_side = self.config.ds_side
        current = _bilinear_downsample(luminance, ds_side)

        # Insufficient history: ingest and return no-motion result.
        if self._frames_ingested < self.config.baseline_frame_lag:
            self._history[self._history_head] = current
            self._history_head = (self._history_head + 1) % self._ring_capacity
            self._frames_ingested += 1
            return MotionResult(
                in_motion=False,
                mean_sad=0.0,
                max_tile_sad=0.0,
                consecutive_count=0,
                frames_ingested=self._frames_ingested,
            )

        baseline_idx = (
            self._history_head - self.config.baseline_frame_lag
        ) % self._ring_capacity
        baseline = self._history[baseline_idx]

        mean_sad, max_tile_sad = _compute_sad_metrics(
            current, baseline, tile_side=self.config.tile_side
        )

        # Rising-edge detection. The result's in_motion is True only
        # on the frame the streak reaches consecutive_frames_required;
        # later frames in a sustained event return False so callers
        # can treat motion as an event stream, not a continuous flag.
        if max_tile_sad >= self.config.threshold:
            self._consecutive_count += 1
            in_motion = (
                self._consecutive_count
                == self.config.consecutive_frames_required
            )
        else:
            self._consecutive_count = 0
            in_motion = False

        # Push current into history ring AFTER comparison.
        self._history[self._history_head] = current
        self._history_head = (self._history_head + 1) % self._ring_capacity
        self._frames_ingested += 1

        return MotionResult(
            in_motion=in_motion,
            mean_sad=float(mean_sad),
            max_tile_sad=float(max_tile_sad),
            consecutive_count=self._consecutive_count,
            frames_ingested=self._frames_ingested,
        )


def _bilinear_downsample(src: np.ndarray, target_size: int) -> np.ndarray:
    """Pure-NumPy bilinear downsample of a 2D uint8 array.

    Returns a `(target_size, target_size)` uint8 array. The four
    neighboring source pixels of each target pixel are averaged by
    fractional offsets. Equivalent to OpenCV's `cv2.INTER_LINEAR` and
    close enough to Apple's vImage Lanczos default that the aggregate
    SAD statistics are within a few percent of the iOS path.
    """
    h, w = src.shape
    # Map target coordinates [0, target-1] back to source space
    # [0, h-1] / [0, w-1]. Float32 throughout to avoid uint8 overflow
    # in the interpolation arithmetic.
    y = np.linspace(0, h - 1, target_size, dtype=np.float32)
    x = np.linspace(0, w - 1, target_size, dtype=np.float32)
    y_int = y.astype(np.int32)
    x_int = x.astype(np.int32)
    # Cap so y_int+1 / x_int+1 never index out of bounds when the
    # target coordinate lands exactly on the last source row/col.
    y_clip = np.minimum(y_int, h - 2)
    x_clip = np.minimum(x_int, w - 2)
    dy = (y - y_clip).reshape(-1, 1)
    dx = (x - x_clip).reshape(1, -1)

    src_f = src.astype(np.float32)
    # Broadcast the 4 neighbor lookups against the dy/dx coefficients.
    yi = y_clip.reshape(-1, 1)
    xi = x_clip.reshape(1, -1)
    tl = src_f[yi, xi]
    tr = src_f[yi, xi + 1]
    bl = src_f[yi + 1, xi]
    br = src_f[yi + 1, xi + 1]

    interpolated = (
        tl * (1.0 - dy) * (1.0 - dx)
        + tr * (1.0 - dy) * dx
        + bl * dy * (1.0 - dx)
        + br * dy * dx
    )
    # Pixel values are bilinearly interpolated; clip to be safe
    # against floating-point rounding pushing the result to 255.5.
    return np.clip(interpolated, 0, 255).astype(np.uint8)


def _compute_sad_metrics(
    current: np.ndarray,
    baseline: np.ndarray,
    *,
    tile_side: int,
) -> tuple[float, float]:
    """Compute (whole-frame mean SAD, max-tile mean SAD) between
    two same-shape 2D uint8 arrays.

    The reshape-to-tiles trick: a `(N, N)` array reshapes to
    `(N // tile, tile, N // tile, tile)`, and summing over axes (1, 3)
    collapses each tile to its tile-wide aggregate.
    """
    # Cast to int16 BEFORE subtracting so the abs-diff cannot wrap
    # under uint8 modular arithmetic.
    diff = np.abs(
        current.astype(np.int16) - baseline.astype(np.int16)
    ).astype(np.uint32)

    n = current.shape[0]
    pixel_count = n * n
    total_sad = int(diff.sum())
    mean_sad = total_sad / pixel_count

    tiles_per_side = n // tile_side
    tile_pixel_count = tile_side * tile_side
    # Reshape to (tiles_y, tile, tiles_x, tile), sum within each tile.
    per_tile_sad = (
        diff.reshape(tiles_per_side, tile_side, tiles_per_side, tile_side)
        .sum(axis=(1, 3))
    )
    per_tile_mean = per_tile_sad.astype(np.float64) / tile_pixel_count
    max_tile_sad = float(per_tile_mean.max())

    return mean_sad, max_tile_sad
