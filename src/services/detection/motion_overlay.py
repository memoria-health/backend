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

_THRESH = 18          # per-pixel Y change (0-255) that counts as motion
_Y_BOOST = 45         # brighten moved pixels so red shows even on dark areas
_ALPHA = 0.6          # red-tint strength on the chroma planes
_RED_U = 100          # Cb for red
_RED_V = 240          # Cr for red
_DILATE = np.ones((5, 5), np.uint8)


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
        cur_y = y.astype(np.int16)                           # original luma (copy)
        if prev_y is not None and prev_y.shape == cur_y.shape:
            mask = (np.abs(cur_y - prev_y) > _THRESH).astype(np.uint8)
            mask = cv2.dilate(mask, _DILATE, iterations=1).astype(bool)
            if mask.any():
                y[mask] = np.minimum(255, cur_y[mask] + _Y_BOOST).astype(np.uint8)
                cmask = mask[::2, ::2]                        # -> chroma resolution
                u = arr[ysize:ysize + csize].reshape(h // 2, w // 2)
                v = arr[ysize + csize:].reshape(h // 2, w // 2)
                u[cmask] = ((1 - _ALPHA) * u[cmask] + _ALPHA * _RED_U).astype(np.uint8)
                v[cmask] = ((1 - _ALPHA) * v[cmask] + _ALPHA * _RED_V).astype(np.uint8)
        prev_y = cur_y
        out.append(FrameSnapshot(w, h, bytes(arr), f.captured_at))
    return out
