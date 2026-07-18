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
  snapshot / zone     → RETIRED (2026-07-18). Stills are dead per the
                        experiment record (full-frame stills fail
                        object checks; action-over-time + crops win).
                        Environment checks ride the scene call.
  overlay (RAPID)     → tint_window() + build_overlay_crop_mp4():
                        the validated P0 recipe (tint the FULL native
                        frames first, then static per-person crop,
                        video-only) run as a regular bucket call since
                        the standalone lane retired (Ledger row 5).

Reuses the P0 helpers directly — `union_box` (pad, min-side, slide-
inside, even-align) and `crop_i420` (plane slicing, renegotiation +
truncation hardening) — so the crop geometry stays identical to the
eval-validated RAPID path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Sequence

import cv2
import numpy as np

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.rapid_artifact import crop_i420, union_box
from src.services.detection.rapid_call import scale_bbox_to_frame

if TYPE_CHECKING:
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


# Tint at bounded resolution. Native-res tint on a 40-60 frame window
# (x2 simultaneously-tracked cameras + their ffmpeg children + event
# archival) OOM-killed the worker twice on soak day 2026-07-14. Motion
# blobs the overlay prompt reads are limb-scale — they survive 640w
# untouched, and every downstream copy (crop, pipe, mp4) shrinks with
# the frames. Tint-before-crop order is unchanged.
_TINT_MAX_WIDTH = 640


def _shrink_i420(f: "FrameSnapshot", max_width: int) -> "FrameSnapshot":
    """Downscale one I420 frame to ≤max_width (even-aligned), plane by
    plane. Malformed frames pass through untouched — overlay_motion
    already skips them."""
    w, h = f.width, f.height
    ysize = w * h
    if (
        w <= max_width or w <= 0 or h <= 0 or w % 2 or h % 2
        or len(f.data) != ysize + 2 * (ysize // 4)
    ):
        return f
    tw = max_width - (max_width % 2)
    th = max(2, round(h * tw / w))
    th -= th % 2
    arr = np.frombuffer(f.data, dtype=np.uint8)
    y = arr[:ysize].reshape(h, w)
    u = arr[ysize:ysize + ysize // 4].reshape(h // 2, w // 2)
    v = arr[ysize + ysize // 4:].reshape(h // 2, w // 2)
    y2 = cv2.resize(y, (tw, th), interpolation=cv2.INTER_AREA)
    u2 = cv2.resize(u, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
    v2 = cv2.resize(v, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
    data = y2.tobytes() + u2.tobytes() + v2.tobytes()
    return FrameSnapshot(
        width=tw, height=th, data=data, captured_at=f.captured_at,
    )


async def tint_window(
    window: Sequence["FrameSnapshot"],
) -> list["FrameSnapshot"]:
    """Motion-overlay the native-rate frames (thread-offloaded), each
    first bounded to _TINT_MAX_WIDTH. Tint before crop is load-bearing
    (rapid_artifact module doc): diffing a moving crop window paints
    phantom red on stationary scenery. The executor tints once per tick
    and crops per overlay plan."""
    def _shrink_then_tint() -> list["FrameSnapshot"]:
        return overlay_motion(
            [_shrink_i420(f, _TINT_MAX_WIDTH) for f in window]
        )

    return await asyncio.to_thread(_shrink_then_tint)


async def build_overlay_crop_mp4(
    tinted: Sequence["FrameSnapshot"],
    *,
    track: "TrackObservation | None",
    sample_width: int,
    fps: int,
) -> bytes | None:
    """Tinted frames → one RAPID artifact: static per-track crop when a
    track is given (identical geometry to the person-crop builder), the
    full tinted frame when not — the P0 lane's no-box fallback. Video-
    only by recipe. None when nothing encodable came out; the executor
    then records the call as degraded rather than silently running clean
    footage under an overlay prompt."""
    if len(tinted) < 2:
        return None
    frames: Sequence["FrameSnapshot"] = tinted
    if track is not None:
        box = person_crop_box(tinted, track=track, sample_width=sample_width)
        if box is not None:
            frames = await asyncio.to_thread(crop_i420, tinted, box)
    if len(frames) < 2:
        return None
    try:
        return await frames_to_mp4(list(frames), fps=fps)
    except (Mp4RemuxError, ValueError) as exc:
        log.warning("scheduling.overlay_crop remux failed err=%s", exc)
        return None
