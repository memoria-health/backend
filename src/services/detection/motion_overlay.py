"""Frame-difference MOTION OVERLAY for the detector's VLM clip.

For each frame we mark the pixels that CHANGED since the previous frame (frame
differencing) and paint them with a bright red tint. The point is to make
otherwise-invisible small motion — a vibrating hand, a tremor — VISIBLE to the
VLM, so it can reason about localized, sustained, non-displacing motion (the
signature of shaking / a seizure) instead of a still-looking limb.

Works directly on I420 (YUV 4:2:0) to avoid two colour conversions per frame:
brighten Y where it moved (so the tint shows even on dark areas), and push the
half-res chroma (U/V) toward red on the motion mask.

Gated by `settings.motion_overlay_enabled`; applied ONLY to the clip the VLM
sees (Re-ID runs on the clean clip). Whenever this is on, the prompt MUST
explain the tint, or detectors will misread red as blood / objects.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.services.detection.frame_buffer import FrameSnapshot

# Naive frame-differencing lights up EVERY high-contrast edge on a noisy /
# compressed camera (the Tapo's sensor + codec noise makes static edges flicker).
# So we: (1) Gaussian-blur first to kill that high-frequency edge noise, (2) use
# a higher threshold, (3) morphological-open to drop speckle, and (4) keep only
# large connected motion blobs (a moving limb is one big region; noise is
# scattered specks). Validated on a real living-room clip: only the moving arms
# tint, the static scene stays clean.
_THRESH = 28          # blurred-luma change (0-255) that counts as motion
_BLUR = 7             # Gaussian kernel (odd) — smooths out sensor/edge noise
_MIN_AREA = 250       # drop motion blobs smaller than this (noise speckle)
_Y_BOOST = 45         # brighten moved pixels so red shows even on dark areas
_ALPHA = 0.6          # red-tint strength on the chroma planes
_RED_U = 100          # Cb for red
_RED_V = 240          # Cr for red
_MORPH_K = np.ones((3, 3), np.uint8)


def overlay_motion(frames: list[FrameSnapshot]) -> list[FrameSnapshot]:
    """Return copies of `frames` with moved regions tinted red. The first frame
    is unchanged (no predecessor); any malformed frame is passed through as-is.
    Pure CPU/numpy — call via `asyncio.to_thread` off the event loop."""
    if len(frames) < 2:
        return list(frames)
    out: list[FrameSnapshot] = []
    prev_y: np.ndarray | None = None
    for f in frames:
        w, h = f.width, f.height
        ysize = w * h
        csize = ysize // 4
        if w <= 0 or h <= 0 or w % 2 or h % 2 or len(f.data) != ysize + 2 * csize:
            out.append(f)
            prev_y = None
            continue
        arr = np.frombuffer(f.data, dtype=np.uint8).copy()   # mutable
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
        out.append(FrameSnapshot(w, h, bytes(arr), f.captured_at))
    return out
