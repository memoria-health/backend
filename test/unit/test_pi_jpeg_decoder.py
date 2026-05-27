"""Unit tests for `services/ingest/pi/jpeg_decoder.py`.

The decoder is pure: JPEG bytes in, DecodedFrame out. Tests
construct synthetic JPEGs in-process via Pillow so the suite has
no external image fixtures.

Verifies: decode shape, byte count, Y-plane fidelity, I420 layout
(Y + Cb/2 + Cr/2 contiguous), error paths for malformed and
odd-dimension input.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.services.ingest.pi.jpeg_decoder import (
    DecodedFrame,
    JpegDecodeError,
    _box_downsample_2x,
    decode_jpeg,
)

# ─── helpers ─────────────────────────────────────────────────────


def _make_jpeg(width: int, height: int, color: tuple[int, int, int] = (128, 64, 200)) -> bytes:
    """Build a JPEG bytes blob with a uniform color, used as a
    deterministic per-pixel test input.
    """
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_gradient_jpeg(width: int, height: int) -> bytes:
    """A JPEG with a horizontal brightness gradient. Useful for
    checking that the Y plane preserves spatial structure (not
    accidentally transposed or mirrored).
    """
    arr = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ─── happy path ──────────────────────────────────────────────────


def test_decode_returns_decoded_frame_with_correct_dimensions() -> None:
    jpeg = _make_jpeg(640, 480)
    frame = decode_jpeg(jpeg)
    assert isinstance(frame, DecodedFrame)
    assert frame.width == 640
    assert frame.height == 480


def test_y_plane_has_expected_shape_and_dtype() -> None:
    jpeg = _make_jpeg(640, 480)
    frame = decode_jpeg(jpeg)
    assert frame.y_plane.shape == (480, 640)
    assert frame.y_plane.dtype == np.uint8


def test_y_plane_is_contiguous_numpy_array() -> None:
    """`MotionDetector.process` reshapes the Y plane to extract
    tiles via reshape, which requires a C-contiguous array.
    """
    jpeg = _make_jpeg(640, 480)
    frame = decode_jpeg(jpeg)
    assert frame.y_plane.flags["C_CONTIGUOUS"]


def test_i420_buffer_has_expected_byte_count() -> None:
    """I420 = Y (W*H) + Cb (W/2 * H/2) + Cr (W/2 * H/2) = 1.5 * W * H."""
    jpeg = _make_jpeg(640, 480)
    frame = decode_jpeg(jpeg)
    expected_bytes = 640 * 480 * 3 // 2
    assert len(frame.i420) == expected_bytes


def test_i420_layout_y_plane_at_offset_zero() -> None:
    """The first W*H bytes of the I420 buffer must match the
    Y-plane array byte-for-byte. Catches a reordering bug that would
    silently scramble colors when LiveKit interprets the buffer.
    """
    jpeg = _make_jpeg(64, 48)
    frame = decode_jpeg(jpeg)
    y_bytes = frame.y_plane.tobytes()
    assert frame.i420[: len(y_bytes)] == y_bytes


def test_decoded_frame_is_immutable() -> None:
    """`DecodedFrame` is a value type. A mutable result is a footgun
    in the relay's per-frame loop.
    """
    jpeg = _make_jpeg(64, 48)
    frame = decode_jpeg(jpeg)
    with pytest.raises(AttributeError):
        frame.width = 999  # type: ignore[misc]


# ─── content fidelity ────────────────────────────────────────────


def test_gradient_preserves_horizontal_structure() -> None:
    """A left-to-right brightness gradient must come back with the
    Y plane brighter on the right than the left. Catches axis
    transposes (height vs width swap) and mirroring bugs.
    """
    jpeg = _make_gradient_jpeg(640, 480)
    frame = decode_jpeg(jpeg)
    left_avg = float(frame.y_plane[:, :100].mean())
    right_avg = float(frame.y_plane[:, -100:].mean())
    assert right_avg > left_avg + 50, (
        f"Gradient lost: left={left_avg:.1f} right={right_avg:.1f}"
    )


def test_uniform_color_produces_uniform_y_plane() -> None:
    """A solid-color JPEG decodes to a Y plane with near-uniform
    values (some lossy compression jitter is expected).
    """
    jpeg = _make_jpeg(64, 48, color=(180, 80, 40))
    frame = decode_jpeg(jpeg)
    # Tolerate JPEG quantization noise around the constant Y value.
    assert frame.y_plane.std() < 5.0


# ─── error paths ─────────────────────────────────────────────────


def test_empty_bytes_raises() -> None:
    with pytest.raises(JpegDecodeError, match="Empty JPEG"):
        decode_jpeg(b"")


def test_non_jpeg_bytes_raises() -> None:
    with pytest.raises(JpegDecodeError):
        decode_jpeg(b"this is not a JPEG payload")


def test_truncated_jpeg_raises() -> None:
    jpeg = _make_jpeg(640, 480)
    truncated = jpeg[: len(jpeg) // 2]
    with pytest.raises(JpegDecodeError):
        decode_jpeg(truncated)


def test_odd_dimension_jpeg_raises() -> None:
    """LiveKit's I420 format requires even dimensions because chroma
    is half-resolution. An odd-dimensioned source frame would corrupt
    the chroma plane shape; reject loudly.
    """
    jpeg = _make_jpeg(641, 480)  # 641 is odd
    with pytest.raises(JpegDecodeError, match="must be even for I420"):
        decode_jpeg(jpeg)


# ─── _box_downsample_2x ──────────────────────────────────────────


def test_box_downsample_halves_dimensions() -> None:
    src = np.zeros((64, 64), dtype=np.uint8)
    out = _box_downsample_2x(src)
    assert out.shape == (32, 32)
    assert out.dtype == np.uint8


def test_box_downsample_averages_constant_block() -> None:
    src = np.full((64, 64), 100, dtype=np.uint8)
    out = _box_downsample_2x(src)
    assert (out == 100).all()


def test_box_downsample_averages_mixed_block() -> None:
    """A 2x2 block of (0, 100, 200, 100) should average to (400/4)=100."""
    src = np.zeros((2, 2), dtype=np.uint8)
    src[0, 0] = 0
    src[0, 1] = 100
    src[1, 0] = 200
    src[1, 1] = 100
    out = _box_downsample_2x(src)
    assert out.shape == (1, 1)
    assert out[0, 0] == 100  # (0+100+200+100) // 4 = 100


def test_box_downsample_handles_uint8_overflow_safely() -> None:
    """Naive uint8 sum of four 255s would overflow; the int16
    intermediate cast must prevent this.
    """
    src = np.full((2, 2), 255, dtype=np.uint8)
    out = _box_downsample_2x(src)
    assert out[0, 0] == 255
