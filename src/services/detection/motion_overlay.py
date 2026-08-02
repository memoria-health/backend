"""Frame-difference MOTION OVERLAY for the detector's VLM clip.

For each frame we mark the pixels that CHANGED since the previous frame (frame
differencing) and paint them with a bright red tint. The point is to make
otherwise-invisible small motion — a vibrating hand, a tremor — VISIBLE to the
VLM, so it can reason about localized, sustained, non-displacing motion (the
signature of shaking / a seizure) instead of a still-looking limb.

Works directly on I420 (YUV 4:2:0) to avoid two colour conversions per frame:
brighten Y where it moved (so the tint shows even on dark areas), and push the
half-res chroma (U/V) toward red on the motion mask.

Used ONLY by the RAPID/overlay channel, which tints its own crop in the
executor (`tint_window` -> `build_overlay_crop_mp4`), and ONLY on the clip
the VLM sees (Re-ID runs on the clean clip). The matching prompt note is
requested per-call via `gemini_provider(motion_overlay=True)` — the tint
and the note must travel together, or detectors misread red as blood or
objects.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.services.detection import frame_codec
from src.services.detection.frame_buffer import FrameSnapshot

# Naive frame-differencing lights up EVERY high-contrast edge on a noisy /
# compressed camera (the Tapo's sensor + codec noise makes static edges flicker).
# So we: (1) Gaussian-blur first to kill that high-frequency edge noise, (2) use
# a higher threshold, (3) morphological-open to drop speckle, and (4) keep only
# large connected motion blobs (a moving limb is one big region; noise is
# scattered specks). Validated on a real living-room clip: only the moving arms
# tint, the static scene stays clean.
_THRESH = 16          # blurred-luma change (0-255) that counts as motion
_BLUR = 5             # Gaussian kernel (odd) — smooths out sensor/edge noise
_MIN_AREA = 60        # drop motion blobs smaller than this (noise speckle)
_Y_BOOST = 45         # brighten moved pixels so red shows even on dark areas
_ALPHA = 0.6          # red-tint strength on the chroma planes
_RED_U = 100          # Cb for red
_RED_V = 240          # Cr for red
_MORPH_K = np.ones((3, 3), np.uint8)


def overlay_motion(frames: list[FrameSnapshot]) -> list[FrameSnapshot]:
    """Return copies of `frames` with moved regions tinted red. The first frame
    is unchanged (no predecessor); any malformed frame is passed through as-is.
    Pure CPU/numpy — call via `asyncio.to_thread` off the event loop.

    JPEG-in-buffer (2026-07-19): JPEG frames are decoded to I420, run
    through the SAME tint math, and re-encoded so the output window
    keeps the input's format (mp4_remux expects one format per window).
    This path runs only on rapid-gated ticks — the decode cost lands
    on the rare windows that earn it."""
    if len(frames) < 2:
        return list(frames)
    out: list[FrameSnapshot] = []
    prev_y: np.ndarray | None = None
    for f in frames:
        was_jpeg = frame_codec.is_jpeg(f.data)
        if was_jpeg:
            try:
                w, h, raw = frame_codec.jpeg_to_i420(f.data)
            except ValueError:
                out.append(f)
                prev_y = None
                continue
        else:
            w, h, raw = f.width, f.height, f.data
        ysize = w * h
        csize = ysize // 4
        if w <= 0 or h <= 0 or w % 2 or h % 2 or len(raw) != ysize + 2 * csize:
            out.append(f)
            prev_y = None
            continue
        arr = np.frombuffer(raw, dtype=np.uint8).copy()   # mutable
        y = arr[:ysize].reshape(h, w)
        y_blur = cv2.GaussianBlur(y, (_BLUR, _BLUR), 0)       # noise-suppressed luma
        if prev_y is not None and prev_y.shape == y_blur.shape:
            mask = (cv2.absdiff(y_blur, prev_y) > _THRESH).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_K, iterations=1)
            # keep only large connected motion regions (a limb) — drops speckle.
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            big = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= _MIN_AREA)[0]
            big = big[big != 0]                              # exclude background
            mask = (np.isin(labels, big) if big.size
                    else np.zeros((h, w), dtype=bool))
            mask = cv2.dilate(mask.astype(np.uint8), _MORPH_K, iterations=2).astype(bool)
            if mask.any():
                y[mask] = np.minimum(255, y[mask].astype(np.int16) + _Y_BOOST).astype(np.uint8)
                cmask = mask[::2, ::2]                        # -> chroma resolution
                u = arr[ysize:ysize + csize].reshape(h // 2, w // 2)
                v = arr[ysize + csize:].reshape(h // 2, w // 2)
                u[cmask] = ((1 - _ALPHA) * u[cmask] + _ALPHA * _RED_U).astype(np.uint8)
                v[cmask] = ((1 - _ALPHA) * v[cmask] + _ALPHA * _RED_V).astype(np.uint8)
        prev_y = y_blur                                      # diff against clean blurred luma
        data = (
            frame_codec.i420_to_jpeg(
                w, h, bytes(arr), quality=frame_codec.DEFAULT_JPEG_QUALITY,
            )
            if was_jpeg else bytes(arr)
        )
        out.append(FrameSnapshot(w, h, data, f.captured_at, luma=f.luma))
    return out
