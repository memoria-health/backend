"""JPEG-in-buffer tests (2026-07-19): the frame codec itself plus the
format-sniffing paths of every consumer that used to assume raw I420 —
the mp4 builder (MJPEG input), the motion/wake math (luma thumbnails),
the rapid overlay + crop (decode → same math → re-encode), and the
buffer's accepts() pre-check.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from src.services.detection import frame_codec
from src.services.detection.frame_buffer import FrameBuffer, FrameSnapshot
from src.services.detection.motion import BlockMotionTracker
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.mp4_remux import frames_to_mp4
from src.services.detection.rapid_artifact import crop_i420


def _solid_i420(w: int, h: int, y_val: int) -> bytes:
    y = np.full((h, w), y_val, dtype=np.uint8)
    uv = np.full((h // 2, w), 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _jpeg_frame(
    w: int, h: int, y_val: int, at: float, *, with_luma: bool = True,
) -> FrameSnapshot:
    raw = _solid_i420(w, h, y_val)
    jpeg, luma = frame_codec.compress_for_buffer(w, h, raw, quality=88)
    return FrameSnapshot(
        width=w, height=h, data=jpeg, captured_at=at,
        luma=luma if with_luma else b"",
    )


# ── codec ──────────────────────────────────────────────────────


def test_jpeg_roundtrip_preserves_dims_and_brightness():
    raw = _solid_i420(640, 360, 180)
    jpeg = frame_codec.i420_to_jpeg(640, 360, raw, quality=88)
    assert frame_codec.is_jpeg(jpeg)
    assert len(jpeg) < len(raw) // 4          # actually compressed
    w, h, back = frame_codec.jpeg_to_i420(jpeg)
    assert (w, h) == (640, 360)
    y = np.frombuffer(back, dtype=np.uint8, count=640 * 360)
    assert abs(float(y.mean()) - 180) < 6     # brightness survives


def test_luma_thumbnail_dims():
    raw = _solid_i420(640, 360, 90)
    luma = frame_codec.luma_thumbnail(640, 360, raw)
    assert len(luma) == frame_codec.LUMA_W * frame_codec.LUMA_H
    arr = np.frombuffer(luma, dtype=np.uint8)
    assert abs(float(arr.mean()) - 90) < 3


# ── odd frame dimensions ───────────────────────────────────────
# A congested publisher downscales to sizes like 213x120, where I420
# rounds each colour plane UP to ceil(w/2) x ceil(h/2). The buffer is
# then bigger than the packed (h*3/2, w) block, which used to raise
# "cannot reshape array of size 38400 into shape (180,213)" once per
# frame in prod.


def _true_i420(w: int, h: int, y_val: int) -> bytes:
    """Correctly sized I420 for ANY dims (chroma planes rounded up)."""
    y = np.full((h, w), y_val, dtype=np.uint8)
    uv = np.full(((h + 1) // 2, (w + 1) // 2), 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes() * 2


@pytest.mark.parametrize("w,h", [(213, 120), (200, 113), (213, 113)])
def test_i420_to_bgr_handles_odd_dimensions(w, h):
    bgr = frame_codec.i420_to_bgr(w, h, _true_i420(w, h, 150))
    assert bgr.shape == (h, w, 3)          # pad is trimmed back off


def test_odd_width_uses_same_colour_math_as_even():
    # The odd path pads Y and reuses cv2's I420 conversion, so dropping
    # one column must not shift the colours.
    odd = frame_codec.i420_to_bgr(213, 120, _true_i420(213, 120, 150))
    even = frame_codec.i420_to_bgr(212, 120, _true_i420(212, 120, 150))
    assert abs(float(odd.mean()) - float(even.mean())) < 1.0


def test_compress_for_buffer_odd_width_prod_repro():
    raw = _true_i420(213, 120, 150)
    assert len(raw) == 38400               # the size from the traceback
    jpeg, luma = frame_codec.compress_for_buffer(213, 120, raw, quality=88)
    assert frame_codec.is_jpeg(jpeg)
    assert len(luma) == frame_codec.LUMA_W * frame_codec.LUMA_H


# ── mp4 builder: MJPEG input ───────────────────────────────────


@pytest.mark.asyncio
async def test_frames_to_mp4_accepts_jpeg_window():
    frames = [
        _jpeg_frame(320, 180, 60 + i * 20, at=float(i)) for i in range(4)
    ]
    mp4 = await frames_to_mp4(frames, fps=4)
    assert mp4[4:8] == b"ftyp"                # a real MP4 came out


@pytest.mark.asyncio
async def test_frames_to_mp4_rejects_mixed_format_window():
    frames = [
        _jpeg_frame(320, 180, 90, at=0.0),
        FrameSnapshot(320, 180, _solid_i420(320, 180, 90), 1.0),
    ]
    with pytest.raises(ValueError):
        await frames_to_mp4(frames, fps=4)


# ── motion math on luma thumbnails ─────────────────────────────


def test_block_motion_uses_luma_thumbnail():
    tracker = BlockMotionTracker()
    still = [_jpeg_frame(640, 360, 100, at=float(i)) for i in range(3)]
    verdict = tracker.assess(still)
    assert verdict.coherent is False

    # A big brightness jump on the whole frame trips blocks.
    moving = [
        _jpeg_frame(640, 360, 100, at=0.0),
        _jpeg_frame(640, 360, 220, at=1.0),
    ]
    verdict = BlockMotionTracker().assess(moving)
    assert verdict.coherent is True
    assert verdict.max_block_delta > 50


def test_block_motion_skips_jpeg_frames_without_luma():
    tracker = BlockMotionTracker()
    frames = [
        _jpeg_frame(640, 360, 100, at=0.0, with_luma=False),
        _jpeg_frame(640, 360, 220, at=1.0, with_luma=False),
    ]
    # JPEG bytes are not a luma plane; without the thumbnail the frame
    # is unreadable to this math — it must skip, never misread.
    verdict = tracker.assess(frames)
    assert verdict.coherent is False


# ── rapid overlay + crop keep the input format ─────────────────


def test_overlay_motion_jpeg_in_jpeg_out():
    frames = [
        _jpeg_frame(320, 180, 60, at=0.0),
        _jpeg_frame(320, 180, 200, at=1.0),   # whole-frame change
    ]
    out = overlay_motion(frames)
    assert len(out) == 2
    assert all(frame_codec.is_jpeg(f.data) for f in out)


def test_crop_i420_jpeg_path():
    frames = [_jpeg_frame(320, 180, 120, at=float(i)) for i in range(2)]
    out = crop_i420(frames, (10, 10, 110, 110))
    assert len(out) == 2
    assert out[0].width == 100 and out[0].height == 100
    assert frame_codec.is_jpeg(out[0].data)


# ── buffer accepts() pre-check ─────────────────────────────────


def test_buffer_accepts_mirrors_push_gap():
    buf = FrameBuffer(window_seconds=10, target_fps=4)   # gap 0.25s
    assert buf.accepts(1.0) is True
    assert buf.push(width=8, height=8, data=b"x", captured_at=1.0)
    assert buf.accepts(1.1) is False                     # inside gap
    assert buf.accepts(1.30) is True
    # push re-applies the rule even if accepts() was consulted earlier
    assert buf.push(width=8, height=8, data=b"y", captured_at=1.1) is False
    assert buf.push(width=8, height=8, data=b"z", captured_at=1.3) is True


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)
