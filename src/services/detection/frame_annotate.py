"""Draw a green bounding box on JPEG-encoded frames.

Used by the OpenAI and Gemini providers to visually mark the patient
on every frame sent to the LLM when Re-ID has locked the patient's
face. The LLM then knows which person in a multi-person frame the
detectors should score against — pure text guidance like "focus on
the patient" leaves the LLM guessing when more than one person is
visible.

The bbox comes from `IdentityResult.face_bbox` and is in the same
pixel space as the extracted frames (both Re-ID inference and the
provider sample at `max_width=480` via `extract_frames`).

We draw the rectangle on every frame, not just the best-match frame.
Patients tend to stay in roughly the same area across a 15s window,
and a single annotated frame would let the LLM forget the
identification halfway through its sequence reasoning. Consistent
marking on all frames is a clearer signal.

Style choices come from the plan ("Only the patient is visually
marked, and only with a thin green rectangle — no label text inside,
no track ID, no confidence number"). A bare box is what production
inference should look like; labels inside the box are debug clutter,
distract the VLM, and look ugly on the archived clips the caregiver
reviews later.
"""

from __future__ import annotations

import asyncio
import base64
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Pure green in BGR (OpenCV's color order), 2px line. Visible against
# both light walls and dim rooms; thin enough to not obscure facial
# features the LLM might still want to read.
_BOX_COLOR_BGR: tuple[int, int, int] = (0, 200, 0)
_BOX_THICKNESS: int = 2


def annotate_jpeg(jpeg_bytes: bytes, bbox: tuple[float, float, float, float]) -> bytes:
    """Synchronous worker. Decodes the JPEG, draws a green rectangle
    at `bbox` (x0, y0, x1, y1), re-encodes. Returns the new JPEG bytes.

    On any error (corrupt JPEG, cv2 throws, bbox out of range), logs
    and returns the original bytes — the LLM sees an un-annotated
    frame rather than dropping the tick.
    """
    try:
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.warning("frame_annotate: imdecode returned None, %d bytes", len(jpeg_bytes))
            return jpeg_bytes
        h, w = img.shape[:2]
        x0, y0, x1, y1 = bbox
        # Clamp to image bounds defensively. Re-ID's coords should
        # already be inside the frame, but a malformed bbox from a
        # cropping bug shouldn't crash the tick.
        ix0 = max(0, min(w - 1, int(x0)))
        iy0 = max(0, min(h - 1, int(y0)))
        ix1 = max(0, min(w - 1, int(x1)))
        iy1 = max(0, min(h - 1, int(y1)))
        if ix1 <= ix0 or iy1 <= iy0:
            log.warning(
                "frame_annotate: degenerate bbox %s in %dx%d frame",
                bbox, w, h,
            )
            return jpeg_bytes
        cv2.rectangle(img, (ix0, iy0), (ix1, iy1), _BOX_COLOR_BGR, _BOX_THICKNESS)
        ok, encoded = cv2.imencode(".jpg", img)
        if not ok:
            log.warning("frame_annotate: imencode failed")
            return jpeg_bytes
        return encoded.tobytes()
    except Exception:
        log.exception("frame_annotate: unexpected failure")
        return jpeg_bytes


async def annotate_jpeg_async(
    jpeg_bytes: bytes,
    bbox: tuple[float, float, float, float],
) -> bytes:
    """Run the cv2 work in a thread so we don't block the event loop.
    OpenAI/Gemini providers process ~44 frames per tick at the
    defaults (11s window × 4fps), and each draw is ~5ms, so this
    stays well inside the tick budget.
    """
    return await asyncio.to_thread(annotate_jpeg, jpeg_bytes, bbox)


async def annotate_base64_async(
    data_base64: str,
    bbox: tuple[float, float, float, float],
) -> str:
    """Convenience wrapper for callers holding b64-encoded JPEGs (the
    shape `ExtractedFrame.data_base64` returns)."""
    raw = base64.b64decode(data_base64)
    annotated = await annotate_jpeg_async(raw, bbox)
    return base64.b64encode(annotated).decode("ascii")
