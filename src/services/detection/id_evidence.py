"""Identity evidence frames (soak observability, Aryaman 2026-07-14).

One annotated JPEG per analyzed tick with people in it: the Re-ID
extraction frame with every track's box drawn on it — PATIENT green,
OTHER red, UNKNOWN amber, ★ when the label came from a face lock —
plus a heavy green outline for the patient box. The visual audit trail
for "is tracking / face-ID / person-ID actually working?", written
where the host can read it (`out/id_frames/<cam8>/HHMMSS.jpg`,
ring-pruned) so the soak dashboard can surface the latest ones.

Flag-gated (`soak_id_frames_enabled`, default off) and wrapped by the
caller in try/except — this module must never be able to break a tick.
Boxes are drawn on the SAME extracted frame Re-ID computed them on, so
no coordinate scaling is involved.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.services.reid.inference import IdentityResult
    from src.settings import Settings

log = logging.getLogger(__name__)

_RING = 240  # newest frames kept per camera (~40 min at one per 10s tick)
_COLORS = {"PATIENT": (46, 160, 90), "OTHER": (205, 70, 60)}
_UNKNOWN = (222, 168, 40)


def _frame_bytes(frame: Any) -> bytes:
    """Extracted frames are JPEGs — raw bytes, a dataclass carrying
    base64 (`data_base64`, the reid pipeline's shape), or raw `.data`."""
    if isinstance(frame, (bytes, bytearray)):
        return bytes(frame)
    b64 = getattr(frame, "data_base64", None)
    if b64 is not None:
        return base64.b64decode(b64)
    return frame.data


def _box_at(track: Any, idx: int):
    """The track's box at (or nearest before) extraction frame `idx`."""
    box = None
    for fi, bb in track.bbox_series:
        if box is None or fi <= idx:
            box = bb
        if fi >= idx:
            break
    return box


def _draw(jpeg: bytes, identity: "IdentityResult") -> bytes | None:
    from PIL import Image, ImageDraw

    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    d = ImageDraw.Draw(im)
    frames_n = len(identity.extracted_frames or ())
    idx = identity.best_frame_index
    if idx is None or not (0 <= idx < frames_n):
        idx = max(0, frames_n - 1)
    drew = False
    for t in identity.tracks or ():
        box = _box_at(t, idx)
        if box is None:
            continue
        color = _COLORS.get(t.label, _UNKNOWN)
        d.rectangle(tuple(box), outline=color, width=3)
        tag = f"{t.label[:1]}{t.track_id} {t.confidence:.2f}"
        if getattr(t, "face_locked", False):
            tag += " *face"
        d.text((box[0] + 4, max(2, box[1] - 14)), tag, fill=color)
        drew = True
    if identity.patient_bbox is not None:
        d.rectangle(tuple(identity.patient_bbox),
                    outline=_COLORS["PATIENT"], width=5)
        drew = True
    if not drew:
        return None
    out = io.BytesIO()
    im.save(out, "JPEG", quality=80)
    return out.getvalue()


async def dump_id_evidence(
    identity: "IdentityResult", *, camera_id, settings: "Settings",
) -> None:
    """Annotate + write one evidence frame; prune the ring. No-ops
    when the flag is off, nothing was extracted, or nobody was seen."""
    if not getattr(settings, "soak_id_frames_enabled", False):
        return
    frames = identity.extracted_frames or ()
    if not frames:
        return
    if not (identity.tracks or identity.patient_bbox is not None):
        return
    idx = identity.best_frame_index
    pick = frames[idx] if (idx is not None and 0 <= idx < len(frames)) else frames[-1]
    annotated = await asyncio.to_thread(_draw, _frame_bytes(pick), identity)
    if annotated is None:
        return
    cam_dir = Path(settings.soak_id_frames_dir) / str(camera_id)[:8]
    cam_dir.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%dT%H%M%S") + ".jpg"
    (cam_dir / name).write_bytes(annotated)
    stale = sorted(cam_dir.glob("*.jpg"))[:-_RING]
    for f in stale:
        f.unlink(missing_ok=True)
