"""RAPID artifact builder (detection v2, P0).

Turns one tick's NATIVE-rate frames into the validated RAPID input — the
recipe that measured convulsion at 8/10 recall / 1/30 FP
(docs/architecture/detection-architecture-v2.md §1.4):

    motion overlay on FULL native frames  →  static per-person crop
    →  encode at native fps, video-only.

Order is load-bearing: the tint is frame-differencing, so it must run on
the full stationary frame. A per-frame moving crop window would diff
stationary scenery against itself shifted and paint phantom red. The crop
is therefore STATIC per person — the union of that person's boxes across
the window, padded, clamped, and even-aligned for I420.

Pure functions + one async entry point; person boxes are an input (from
Re-ID's per-frame bboxes or a one-shot tracker pass — the caller decides),
so this module stays testable without loading any model.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from src.services.detection import frame_codec
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4

log = logging.getLogger("nemara.detection.rapid")

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels

# Padding around the union box, per side, as a fraction of the box size.
# Matches the head-room the validated experiment crops carried; re-checked
# against the labeled eval set before the flag defaults on.
_PAD_RATIO = 0.20
# Never crop tighter than this — tiny crops starve the encoder of context.
_MIN_SIDE = 64


def _even_floor(v: float) -> int:
    return (int(v) // 2) * 2


def _even_ceil(v: float) -> int:
    n = int(v) if float(v).is_integer() else int(v) + 1
    return n if n % 2 == 0 else n + 1


def union_box(
    boxes: Sequence[Box],
    frame_w: int,
    frame_h: int,
    *,
    pad_ratio: float = _PAD_RATIO,
    min_side: int = _MIN_SIDE,
) -> tuple[int, int, int, int] | None:
    """One person's boxes over the window → a single STATIC crop window:
    union → pad → grow to min_side → clamp → even-align. None if no boxes
    or the result degenerates."""
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad_x = (x1 - x0) * pad_ratio
    pad_y = (y1 - y0) * pad_ratio
    x0, y0, x1, y1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
    # Grow to min_side around the center (before clamping, so an edge box
    # slides inward instead of staying undersized).
    if x1 - x0 < min_side:
        cx = (x0 + x1) / 2
        x0, x1 = cx - min_side / 2, cx + min_side / 2
    if y1 - y0 < min_side:
        cy = (y0 + y1) / 2
        y0, y1 = cy - min_side / 2, cy + min_side / 2
    # Slide fully inside the frame, then clamp (frame dims are I420-even).
    if x0 < 0:
        x1, x0 = x1 - x0, 0
    if y0 < 0:
        y1, y0 = y1 - y0, 0
    if x1 > frame_w:
        x0, x1 = max(0.0, x0 - (x1 - frame_w)), frame_w
    if y1 > frame_h:
        y0, y1 = max(0.0, y0 - (y1 - frame_h)), frame_h
    xi0, yi0 = _even_floor(max(0.0, x0)), _even_floor(max(0.0, y0))
    xi1, yi1 = min(frame_w, _even_ceil(x1)), min(frame_h, _even_ceil(y1))
    if xi1 - xi0 < 2 or yi1 - yi0 < 2:
        return None
    return (xi0, yi0, xi1, yi1)


def crop_i420(
    frames: Sequence[FrameSnapshot], box: tuple[int, int, int, int]
) -> list[FrameSnapshot]:
    """Crop each I420 frame to `box` (even-aligned) via plane slicing.
    Frames whose shape mismatches the first frame's, or whose data is
    truncated, are skipped (renegotiation mid-window / malformed input)."""
    import numpy as np

    if not frames:
        return []
    x0, y0, x1, y1 = box
    # Reference = the MOST RECENT frame's shape (mp4_remux convention):
    # WebRTC renegotiation can change resolution mid-window, and the last
    # frame is the current publish state — anchoring on frame 0 would keep
    # the stale minority and drop the live majority.
    ref_w, ref_h = frames[-1].width, frames[-1].height
    cw, ch = x1 - x0, y1 - y0
    out: list[FrameSnapshot] = []
    for f in frames:
        if (f.width, f.height) != (ref_w, ref_h):
            continue
        # JPEG-in-buffer (2026-07-19): decode → crop → re-encode so
        # the cropped window keeps the input's format. Rapid-gated
        # path only — the decode cost lands on windows that earn it.
        if frame_codec.is_jpeg(f.data):
            import cv2
            try:
                bgr = frame_codec.jpeg_to_bgr(f.data)
            except ValueError:
                continue
            if bgr.shape[0] < y1 or bgr.shape[1] < x1:
                continue
            ok, buf = cv2.imencode(
                ".jpg", bgr[y0:y1, x0:x1],
                [int(cv2.IMWRITE_JPEG_QUALITY),
                 frame_codec.DEFAULT_JPEG_QUALITY],
            )
            if not ok:
                continue
            out.append(FrameSnapshot(
                width=cw, height=ch, data=bytes(buf),
                captured_at=f.captured_at,
            ))
            continue
        ysize = f.width * f.height
        csize = ysize // 4
        if len(f.data) < ysize + 2 * csize:
            continue
        arr = np.frombuffer(f.data, dtype=np.uint8)
        y = arr[:ysize].reshape(f.height, f.width)[y0:y1, x0:x1]
        u = arr[ysize:ysize + csize].reshape(f.height // 2, f.width // 2)[
            y0 // 2: y1 // 2, x0 // 2: x1 // 2]
        v = arr[ysize + csize:ysize + 2 * csize].reshape(f.height // 2, f.width // 2)[
            y0 // 2: y1 // 2, x0 // 2: x1 // 2]
        out.append(FrameSnapshot(
            width=cw, height=ch,
            data=y.tobytes() + u.tobytes() + v.tobytes(),
            captured_at=f.captured_at,
        ))
    return out


async def build_rapid_artifacts(
    frames: Sequence[FrameSnapshot],
    per_person_boxes: Sequence[Sequence[Box]],
    *,
    fps: int,
) -> list[bytes]:
    """One video-only MP4 per person: tint native frames once, crop per
    person, encode at native fps. Persons with no usable box are skipped."""
    if len(frames) < 2 or not per_person_boxes:
        return []
    tinted = await asyncio.to_thread(overlay_motion, list(frames))
    frame_w, frame_h = frames[-1].width, frames[-1].height
    artifacts: list[bytes] = []
    for boxes in per_person_boxes:
        box = union_box(boxes, frame_w, frame_h)
        if box is None:
            continue
        cropped = await asyncio.to_thread(crop_i420, tinted, box)
        if len(cropped) < 2:
            continue
        try:
            artifacts.append(await frames_to_mp4(cropped, fps=fps))
        except (Mp4RemuxError, ValueError) as exc:
            # One person's encode failing must not drop the others'.
            log.warning("rapid_artifact: encode failed, skipping person: %s", exc)
            continue
    return artifacts
