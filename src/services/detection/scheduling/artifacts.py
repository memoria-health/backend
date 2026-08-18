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

from src.services.detection import frame_codec
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.mp4_remux import Mp4RemuxError, frames_to_mp4
from src.services.detection.rapid_artifact import crop_i420, union_box
from src.services.detection.rapid_call import scale_bbox_to_frame
from src.services.identity.observations import bbox_extracted_width

if TYPE_CHECKING:
    from src.services.identity.observations import TrackObservation

log = logging.getLogger(__name__)

_FOLLOW_COVERAGE_S = 3.0   # max series-to-frame time gap before degrade
_FOLLOW_EMA = 0.5          # window pan smoothing (1.0 = raw recenter)


def person_crop_box(
    window: Sequence["FrameSnapshot"],
    *,
    track: "TrackObservation",
    sample_width: int,
) -> tuple[int, int, int, int] | None:
    """The static native-pixel crop window for one track: every box in
    its series scaled onto the native grid, then union+pad+clamp. None
    when the track has no usable boxes or the window is empty.

    `sample_width` is only a HINT — the real divisor comes from the
    track's own `bbox_space` (see `bbox_extracted_width`). A sidecar
    track is already native and must not be scaled; multiplying it
    anyway is the 2026-08-07 bug that cropped the patient's legs and
    1400px of empty room."""
    if not window or not track.bbox_series:
        return None
    # mp4_remux convention: the LAST frame's shape is the live publish
    # state; crop_i420 anchors on it too, so scale against the same.
    ref = window[-1]
    extracted_w = bbox_extracted_width(
        track, ref.width, sample_width=sample_width,
    )
    scaled = []
    for _, bbox in track.bbox_series:
        b = scale_bbox_to_frame(
            bbox, ref.width, ref.height, sample_width=extracted_w,
        )
        if b is not None:
            scaled.append(b)
    if not scaled:
        return None
    return union_box(scaled, ref.width, ref.height)


def plan_follow_rects(
    window: Sequence["FrameSnapshot"],
    track: "TrackObservation",
    *,
    sample_width: int,
) -> "list[tuple[int, int, int, int]] | None":
    """One rect PER FRAME sized to that frame's own (EMA-smoothed) box
    — TRUE digital zoom (2026-08-18, Aryaman: a fixed-size viewport
    left the person tiny whenever the window also held a close-up
    moment). Every rect keeps the OUTPUT aspect so the encoder-constant
    resize is pure scale, never distortion.

    Cross-machine clocks differ (series = node wall clock, frames =
    backend monotonic), so alignment is by OFFSET FROM THE NEWEST entry
    on each side. Returns (rects, (out_w, out_h)); frames are resized
    to out dims after cropping."""
    if not window or not track.bbox_series:
        return None
    ref = window[-1]
    extracted_w = bbox_extracted_width(
        track, ref.width, sample_width=sample_width,
    )
    series = []
    for ts, bbox in track.bbox_series:
        b = scale_bbox_to_frame(
            bbox, ref.width, ref.height, sample_width=extracted_w,
        )
        if b is not None:
            series.append((float(ts), b))
    if not series:
        return None
    s_end = series[-1][0]
    w_end = window[-1].captured_at
    # Nearest-in-time box per frame; a frame with no box within the
    # coverage threshold degrades the WHOLE call (mixed framing inside
    # one clip is worse than full-frame). Padded via the eval-validated
    # union math on a single box (pad + min_side + clamp + even-align).
    picked = []
    for f in window:
        off = w_end - f.captured_at
        gap, best = min(
            ((abs((s_end - e[0]) - off), e[1]) for e in series),
            key=lambda g: g[0],
        )
        if gap > _FOLLOW_COVERAGE_S:
            return None
        r = union_box([best], ref.width, ref.height)
        if r is None:
            return None
        picked.append(r)
    ow = min(max(r[2] - r[0] for r in picked), ref.width) & ~1
    oh = min(max(r[3] - r[1] for r in picked), ref.height) & ~1
    if ow < 2 or oh < 2:
        return None
    aspect = ow / oh
    rects = []
    scx = scy = sw = sh = None
    for r in picked:
        cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        # EMA pan + zoom: the box cadence (~3fps) is coarser than the
        # frame rate; raw recentering/resizing teleports the viewport
        # every few frames and paints global motion into the clip.
        scx = cx if scx is None else scx + _FOLLOW_EMA * (cx - scx)
        scy = cy if scy is None else scy + _FOLLOW_EMA * (cy - scy)
        sw = (r[2] - r[0]) if sw is None else sw + _FOLLOW_EMA * ((r[2] - r[0]) - sw)
        sh = (r[3] - r[1]) if sh is None else sh + _FOLLOW_EMA * ((r[3] - r[1]) - sh)
        # Expand this frame's box to the OUTPUT aspect (grow, never
        # shrink) so the final resize is distortion-free zoom.
        cw, ch = sw, sh
        if cw / ch < aspect:
            cw = ch * aspect
        else:
            ch = cw / aspect
        cw = int(min(cw, ref.width)) & ~1
        ch = int(min(ch, ref.height)) & ~1
        cw, ch = max(cw, 2), max(ch, 2)
        x0 = int(max(0, min(scx - cw / 2, ref.width - cw))) & ~1
        y0 = int(max(0, min(scy - ch / 2, ref.height - ch))) & ~1
        rects.append((x0, y0, x0 + cw, y0 + ch))
    return rects, (ow, oh)


def _resize_frame(f: "FrameSnapshot", tw: int, th: int) -> "FrameSnapshot":
    """Scale one cropped frame (JPEG or raw I420) to exactly tw×th —
    the zoom half of crop-and-zoom. Same aspect by construction."""
    if (f.width, f.height) == (tw, th):
        return f
    interp = cv2.INTER_LINEAR if tw > f.width else cv2.INTER_AREA
    if frame_codec.is_jpeg(f.data):
        try:
            bgr = frame_codec.jpeg_to_bgr(f.data)
        except ValueError:
            return f
        ok, buf = cv2.imencode(
            ".jpg", cv2.resize(bgr, (tw, th), interpolation=interp))
        if not ok:
            return f
        return FrameSnapshot(width=tw, height=th, data=buf.tobytes(),
                             captured_at=f.captured_at)
    w, h = f.width, f.height
    ysize = w * h
    if len(f.data) != ysize + 2 * (ysize // 4):
        return f
    arr = np.frombuffer(f.data, dtype=np.uint8)
    y = cv2.resize(arr[:ysize].reshape(h, w), (tw, th),
                   interpolation=interp)
    u = cv2.resize(arr[ysize:ysize + ysize // 4].reshape(h // 2, w // 2),
                   (tw // 2, th // 2), interpolation=interp)
    v = cv2.resize(arr[ysize + ysize // 4:].reshape(h // 2, w // 2),
                   (tw // 2, th // 2), interpolation=interp)
    return FrameSnapshot(
        width=tw, height=th,
        data=y.tobytes() + u.tobytes() + v.tobytes(),
        captured_at=f.captured_at,
    )


def _crop_follow(window, rects, out_dims):
    ow, oh = out_dims
    out = []
    for f, r in zip(window, rects):
        for c in crop_i420([f], r):
            out.append(_resize_frame(c, ow, oh))
    return out


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
    """One track → one FOLLOWING cropped MP4 at the recipe fps. None
    when no usable crop could be built — the executor then degrades
    that call (the detectors are not silently dropped)."""
    plan = plan_follow_rects(window, track, sample_width=sample_width)
    if plan is None:
        return None
    rects, out_dims = plan
    cropped = await asyncio.to_thread(_crop_follow, window, rects, out_dims)
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
            "scheduling.person_crop remux failed track=%d rects=%d err=%s",
            track.track_id, len(rects), exc,
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
