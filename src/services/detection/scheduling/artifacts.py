"""Per-bucket artifact builders (detection v2, P3 T2).

The packer names a recipe; this module builds the actual media. P3
ships the builders the CURRENT catalog needs when the flag turns on:

  scene / full-frame  → no builder here: the tick's existing window
                        mp4 (frames_to_mp4 over the buffered window)
                        IS the scene artifact.
  person crop         → build_person_crop_mp4(): the track's swept
                        union box (STATIC crop — the tint-then-crop /
                        static-box law), cropped via I420 plane
                        slicing, remuxed at the recipe fps. Carries
                        audio when the call needs it.
  face crop           → DEFERRED to P6 with the FACE detectors: the
                        pipeline does not retain face boxes yet (only
                        face_px_max), and nothing in the default 24
                        lands in the FACE bucket. Do not build ahead
                        of the data it needs.
  snapshot / zone     → DEFERRED to P6 with per-camera zone configs
                        (the packer routes snapshot members to the
                        wide call until then — full-frame stills
                        measurably fail).
  overlay (RAPID)     → owned by the validated P0 lane
                        (rapid_artifact.build_rapid_artifacts).

Reuses the P0 helpers directly — `union_box` (pad, min-side, slide-
inside, even-align) and `crop_i420` (plane slicing, renegotiation +
truncation hardening) — so the crop geometry stays identical to the
eval-validated RAPID path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Sequence

from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.rapid_artifact import crop_i420, union_box
from src.services.detection.rapid_call import scale_bbox_to_frame

if TYPE_CHECKING:
    from src.services.detection.frame_buffer import FrameSnapshot
    from src.services.reid.body_pipeline import TrackObservation

log = logging.getLogger(__name__)


def person_crop_box(
    window: Sequence["FrameSnapshot"],
    *,
    track: "TrackObservation",
    sample_width: int,
) -> tuple[int, int, int, int] | None:
    """The static native-pixel crop window for one track: every box in
    its series scaled extracted→native, then union+pad+clamp. None when
    the track has no usable boxes or the window is empty."""
    if not window or not track.bbox_series:
        return None
    # mp4_remux convention: the LAST frame's shape is the live publish
    # state; crop_i420 anchors on it too, so scale against the same.
    ref = window[-1]
    scaled = []
    for _, bbox in track.bbox_series:
        b = scale_bbox_to_frame(
            bbox, ref.width, ref.height, sample_width=sample_width,
        )
        if b is not None:
            scaled.append(b)
    if not scaled:
        return None
    return union_box(scaled, ref.width, ref.height)


async def build_person_crop_mp4(
    window: Sequence["FrameSnapshot"],
    *,
    track: "TrackObservation",
    sample_width: int,
    fps: int,
    pcm: bytes | None = None,
    pcm_sample_rate: int | None = None,
    pcm_num_channels: int | None = None,
) -> bytes | None:
    """One track → one cropped MP4 at the recipe fps. None when no
    usable crop could be built — the executor then degrades that call
    (the detectors are not silently dropped)."""
    box = person_crop_box(window, track=track, sample_width=sample_width)
    if box is None:
        return None
    cropped = await asyncio.to_thread(crop_i420, window, box)
    if not cropped:
        return None
    try:
        return await frames_to_mp4(
            cropped,
            fps=fps,
            pcm=pcm if pcm else None,
            pcm_sample_rate=pcm_sample_rate,
            pcm_num_channels=pcm_num_channels,
        )
    except (Mp4RemuxError, ValueError) as exc:
        log.warning(
            "scheduling.person_crop remux failed track=%d box=%s err=%s",
            track.track_id, box, exc,
        )
        return None
