"""Native SAM 3.1 multiplex session wrapper — perception core of the
v2 pod service. Self-contained: no `src` imports, so the pod serves
with only sam3 + torch + cv2 + numpy on the path.

Ported from the validated v1 sam_pipeline internals (2026-07-20):
  - build_sam3_multiplex_video_predictor + init_state kwargs shim
  - bbox derived from the MASK's bounding rect (out_boxes_xywh is
    normalized [0,1] and silently truncates to zeros under int())
  - session flow: start_session(file) -> add_prompt(frame 0, text)
    -> propagate_in_video stream -> close_session, temp file removed

The propagation is exposed as a GENERATOR of per-frame parses so the
server can act on each frame (schedule face attempts, cut mask-clean
crops) and drop the pixel masks immediately — flat memory at 88-frame
worst-case ticks instead of holding ~200MB of masks.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Iterator

import numpy as np

log = logging.getLogger("perception.track")

_SAM = None


def _np(x):
    if x is None:
        return None
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def get_sam(ckpt_path: str):
    """Process-singleton native SAM 3.1 multiplex video predictor."""
    global _SAM
    if _SAM is None:
        import inspect

        from sam3.model_builder import build_sam3_multiplex_video_predictor

        pred = build_sam3_multiplex_video_predictor(
            checkpoint_path=ckpt_path, use_fa3=False,
        )
        orig = pred.model.init_state
        sig = inspect.signature(orig)
        pred.model.init_state = lambda **kw: orig(
            **{k: v for k, v in kw.items() if k in sig.parameters}
        )
        _SAM = pred
        log.info("sam_track.loaded ckpt=%s", ckpt_path)
    return _SAM


def mask_bbox(m: np.ndarray) -> tuple[float, float, float, float] | None:
    """Bounding rect (x0, y0, x1, y1) of a binary mask, or None."""
    rows = np.any(m > 0.5, axis=1)
    cols = np.any(m > 0.5, axis=0)
    if not rows.any() or not cols.any():
        return None
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    return (float(xs[0]), float(ys[0]), float(xs[-1]) + 1.0, float(ys[-1]) + 1.0)


def parse_frame(outputs: dict, thr: float) -> list[tuple[int, tuple, np.ndarray]]:
    """One frame's SAM outputs -> [(track_id, bbox_xyxy_px, mask), ...]
    for detections at/above `thr`. Mask is the full-res binary mask."""
    ids = _np(outputs.get("out_obj_ids"))
    probs = _np(outputs.get("out_probs"))
    masks = outputs.get("out_binary_masks")
    n = 0 if ids is None else len(ids)
    dets = []
    for i in range(n):
        p = float(probs[i]) if probs is not None else 1.0
        if p < thr:
            continue
        m = _np(masks[i]) if masks is not None and i < len(masks) else None
        if m is not None and m.ndim == 3:
            m = m[0]
        if m is None:
            continue
        bbox = mask_bbox(m)
        if bbox is None:
            continue
        dets.append((int(ids[i]), bbox, m, p))
    return dets


def ground_frame0(
    pred, frames_bgr: list[np.ndarray], *, concept: str, thr: float,
) -> list:
    """Detections on frame 0 only, WITHOUT propagating the rest.

    `continuous._ground` wants one thing: the masks on frame 0. It has
    always got them by starting a propagation stream and breaking out of
    it at the first yield — but the stream had already computed the
    other five frames by then, and their results were dropped.

    Measured on the pod, 6 real 1280x720 frames with two people in shot:

        with propagate     2101 ms   2 detections
        without propagate   694 ms   2 detections   (3.03x)
        mask IoU vs the propagate path: min 0.982, mean 0.984

    The temporal context SAM3 multiplex needs is not lost. The session
    is still opened over all six frames, so `add_prompt` grounds with
    that context available — we simply stop asking for per-frame results
    we throw away. The residual ~2% mask difference is propagation
    refining frame 0 through its memory bank; downstream that mask is
    matched at an IoU floor of 0.30 and used to seed EdgeTAM, so 2% is
    far below anything that can change an identity decision.

    `track_stream` is untouched and still used by `_track_native`
    (NEMARA_TRACKER=sam3), which genuinely consumes every frame.
    """
    import torch

    if not frames_bgr:
        return []
    path = _write_clip(frames_bgr)
    session_id = None
    try:
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
        ):
            started = pred.handle_request(request=dict(
                type="start_session", resource_path=path,
            ))
            session_id = started["session_id"]
            seeded = pred.handle_request(request=dict(
                type="add_prompt", session_id=session_id,
                frame_index=0, text=concept,
            ))
        outputs = seeded.get("outputs")
        return parse_frame(outputs, thr) if outputs is not None else []
    finally:
        if session_id is not None:
            try:
                pred.handle_request(request=dict(
                    type="close_session", session_id=session_id,
                ))
            except Exception:  # noqa: BLE001
                log.warning("sam_track.close_failed", exc_info=True)
        try:
            os.unlink(path)
        except OSError:
            pass


def _write_clip(frames_bgr: list[np.ndarray]) -> str:
    """Encode frames to an mp4 SAM3 can open, in RAM-backed /dev/shm.

    Measured at 21ms for 6 frames — 1.2% of a ground tick, so this is
    not worth removing even though it looks like the JPEG round trip we
    deleted from the EdgeTAM path. SAM3's session API takes a
    resource_path, not a tensor.
    """
    import cv2

    h, w = frames_bgr[0].shape[:2]
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=tmp_dir)
    os.close(fd)
    writer = None
    try:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (w, h),
        )
        for frame in frames_bgr:
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return path


def track_stream(
    pred, frames_bgr: list[np.ndarray], *, concept: str, thr: float,
) -> Iterator[tuple[int, list]]:
    """Yield (position, [(tid, bbox, mask, prob), ...]) per frame of
    `frames_bgr`, in order. Position indexes into frames_bgr; the
    caller owns any orig-index mapping. Masks are valid only during
    the yield (dropped by GC as the caller releases them)."""
    import cv2

    if not frames_bgr:
        return
    h, w = frames_bgr[0].shape[:2]
    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    fd, path = tempfile.mkstemp(suffix=".mp4", dir=tmp_dir)
    os.close(fd)
    writer = None
    session_id = None
    emitted: set[int] = set()
    try:
        # Writer INSIDE the try so an encode failure never leaks a
        # patient-video mp4 in RAM-backed /dev/shm (review S7).
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (w, h),
        )
        for i, frame in enumerate(frames_bgr):
            # cv2.VideoWriter SILENTLY DROPS frames whose dims differ
            # from the writer's — one dropped write shifts SAM's
            # positions off pos_to_orig for the rest of the clip
            # (reviews C2/E4). The server filters to uniform dims
            # upstream; this resize is the belt if anything slips.
            if frame.shape[:2] != (h, w):
                log.warning(
                    "sam_track.dim_mismatch pos=%d %s != %s — resizing",
                    i, frame.shape[:2], (h, w),
                )
                frame = cv2.resize(frame, (w, h))
            writer.write(frame)
        writer.release()
        writer = None

        import torch

        # EXPLICIT autocast: torch autocast state is THREAD-LOCAL, and
        # sam3 relies on it being active (bf16 weights, f32 inputs).
        # v1 inherited it implicitly on the main thread; the server now
        # runs perception in a worker thread (to_thread), where the
        # implicit state doesn't exist — without this, every forward
        # dies with "mat1 and mat2 must have the same dtype".
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available(),
        ):
            started = pred.handle_request(request=dict(
                type="start_session", resource_path=path,
            ))
            session_id = started["session_id"]
            seeded = pred.handle_request(request=dict(
                type="add_prompt", session_id=session_id,
                frame_index=0, text=concept,
            ))
            stream = pred.handle_stream_request(request=dict(
                type="propagate_in_video", session_id=session_id,
                propagation_direction="forward", start_frame_index=0,
            ))
            # Frame 0 must be yielded FIRST: consumers run the face
            # schedule in arrival order, and a late frame-0 fallback
            # would phase-shift every per-track attempt off the brain's
            # cadence (review C3). If propagation starts past 0, emit
            # the seeded frame-0 parse before the stream.
            first = next(stream, None)
            if first is not None:
                pos0 = int(first.get("frame_index", first.get("frame_idx")))
                if pos0 != 0 and seeded.get("outputs") is not None:
                    emitted.add(0)
                    yield 0, parse_frame(seeded["outputs"], thr)
                emitted.add(pos0)
                yield pos0, parse_frame(first.get("outputs", first), thr)
                for res in stream:
                    pos = int(res.get("frame_index", res.get("frame_idx")))
                    emitted.add(pos)
                    yield pos, parse_frame(res.get("outputs", res), thr)
            if 0 not in emitted and seeded.get("outputs") is not None:
                yield 0, parse_frame(seeded["outputs"], thr)
    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:  # noqa: BLE001
                pass
        if session_id is not None:
            try:
                pred.handle_request(request=dict(
                    type="close_session", session_id=session_id,
                ))
            except Exception:  # noqa: BLE001
                log.exception("sam_track.close_session_failed")
        try:
            os.unlink(path)
        except OSError:
            pass


def mask_crop(
    img: np.ndarray,
    bbox: tuple[float, float, float, float],
    mask: np.ndarray | None,
) -> np.ndarray | None:
    """Crop bbox and zero every pixel outside the person mask — the
    core SAM advantage: only this person's pixels reach the face
    embedder (kills couch/neighbor contamination)."""
    import cv2

    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = img[y0:y1, x0:x1].copy()
    if mask is not None:
        try:
            m = mask
            if m.shape[:2] != img.shape[:2]:
                m = cv2.resize(
                    m.astype(np.uint8), (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            m = m[y0:y1, x0:x1]
            crop[m == 0] = 0
        except Exception:  # noqa: BLE001
            log.exception("sam_track.mask_apply_failed")
    return crop
