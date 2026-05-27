"""JPEG bytes to I420 frame conversion.

The Pi's MJPEG stream gives us JPEG bytes per frame. Two downstream
consumers need different views of those bytes:

  - `motion_detector.MotionDetector.process(luminance)` wants a 2D
    uint8 Y plane (any resolution).
  - The LiveKit publisher wants an I420 buffer (Y plane + half-res
    Cb plane + half-res Cr plane, concatenated). LiveKit re-encodes
    this as H.264 internally; we just need to hand it well-formed
    planar YUV 4:2:0.

`decode_jpeg(jpeg_bytes)` does both decodes in one pass: Pillow
decodes the JPEG to a YCbCr image (4:4:4), NumPy extracts the Y
plane verbatim, and the Cb/Cr planes are box-averaged 2x2 to get
the 4:2:0 subsampling LiveKit expects.

Returns a `DecodedFrame` value object so the publisher and motion
detector get the same single source of truth per frame and we never
decode the same JPEG twice.

Failure modes: malformed JPEGs from a glitching Pi (rare but real
on a noisy 2.4 GHz WiFi link) raise `JpegDecodeError`. The caller
should drop the frame and continue; protocol-level reconnection is
handled by `mjpeg_reader.read_mjpeg`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError


class JpegDecodeError(Exception):
    """Raised when a JPEG cannot be decoded.

    The relay's per-frame loop should catch this and drop the frame,
    not tear down the LiveKit publisher.
    """


@dataclass(frozen=True)
class DecodedFrame:
    """One decoded JPEG frame in two shapes.

    `y_plane` is the (height, width) uint8 Y channel, suitable for
    direct hand-off to MotionDetector.process.

    `i420` is a contiguous bytes buffer in I420 layout:
        Y plane (height * width bytes), then
        Cb plane (height//2 * width//2 bytes), then
        Cr plane (height//2 * width//2 bytes).
    Suitable for direct hand-off to `livekit.rtc.VideoFrame` with
    buffer type `I420`.
    """

    width: int
    height: int
    y_plane: np.ndarray
    i420: bytes


def decode_jpeg(jpeg_bytes: bytes) -> DecodedFrame:
    """Decode one JPEG frame.

    Raises `JpegDecodeError` on malformed input. The returned
    `DecodedFrame.y_plane` is a fresh numpy array (no aliasing of
    Pillow's internal buffer), and `.i420` is a fresh bytes object
    safe to pass to LiveKit which expects to own it.
    """
    if not jpeg_bytes:
        raise JpegDecodeError("Empty JPEG payload")

    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        # `convert("YCbCr")` gives a 4:4:4 image with three uint8
        # channels in band order Y, Cb, Cr. Pillow handles all the
        # JPEG-specific colorspace fixups (BT.601 vs JFIF range).
        ycbcr = img.convert("YCbCr")
    except UnidentifiedImageError as exc:
        raise JpegDecodeError(f"Not a JPEG: {exc}") from exc
    except OSError as exc:
        # Pillow raises OSError for truncated / corrupt JPEGs.
        raise JpegDecodeError(f"Corrupt JPEG: {exc}") from exc

    width, height = ycbcr.size

    # LiveKit's I420 frame format requires even dimensions because
    # the chroma planes are half-resolution. JPEG encoders should
    # always produce even-dimensioned outputs at sane resolutions
    # (everything we care about: 480p, 720p, 1080p), but we guard
    # against a pathological Pi config that picked an odd width.
    if width % 2 != 0 or height % 2 != 0:
        raise JpegDecodeError(
            f"Frame dimensions ({width}x{height}) must be even for I420"
        )

    arr = np.asarray(ycbcr, dtype=np.uint8)
    # `arr` is (H, W, 3) with axes Y, Cb, Cr at full resolution.
    y = np.ascontiguousarray(arr[:, :, 0])
    cb_full = arr[:, :, 1]
    cr_full = arr[:, :, 2]

    cb = _box_downsample_2x(cb_full)
    cr = _box_downsample_2x(cr_full)

    # I420 layout: Y bytes + Cb bytes + Cr bytes, contiguous.
    i420 = y.tobytes() + cb.tobytes() + cr.tobytes()

    return DecodedFrame(
        width=width,
        height=height,
        y_plane=y,
        i420=i420,
    )


def _box_downsample_2x(plane: np.ndarray) -> np.ndarray:
    """Box-average a (H, W) uint8 array down by 2x in each dimension.

    Used to convert Pillow's 4:4:4 YCbCr output into the 4:2:0
    chroma subsampling that I420 expects. Box averaging is the
    correct downsample for chroma: it preserves the local color
    statistics that a downstream H.264 encoder uses.

    Assumes even input dimensions; caller has validated this.
    """
    h, w = plane.shape
    # Reshape to (H/2, 2, W/2, 2), take mean over the 2x2 blocks.
    # int16 intermediate avoids uint8 overflow during the sum.
    block_sum = (
        plane.reshape(h // 2, 2, w // 2, 2)
        .astype(np.int16)
        .sum(axis=(1, 3))
    )
    return (block_sum // 4).astype(np.uint8)
