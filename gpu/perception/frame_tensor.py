"""Feed EdgeTAM frames from memory instead of round-tripping through disk.

WHY (measured 2026-08-01, on the A40, 12 frames @ 640x360, 1 object):

    JPEG encode + write      15.8 ms    0.9%
    init_state (load)      1184.7 ms   68.4%   <- 98.7 ms/frame
    seed mask                26.1 ms    1.5%
    propagate_in_video      506.1 ms   29.2%   <- 42.2 ms/frame
    ------------------------------------------
    total                  1732.8 ms          144.4 ms/frame

`propagate_in_video` — the actual tracking — runs at 42.2 ms/frame,
right on EdgeTAM's published ~36 ms/frame. The model was never slow.
Two thirds of every call was `init_state`'s loader: PIL-opening each
JPEG we had just written, decoding it, resizing to image_size^2 and
normalising, one frame at a time on the CPU.

The frames are already decoded in the pod's memory when this runs. The
same preprocessing as one batched GPU op measures **4.2 ms** for the
whole batch, so the call drops 1732.8 -> 536.5 ms (3.23x) with the
model doing identical work.

HOW: `sam2_video_predictor` binds `load_video_frames` into its own
module namespace (`from sam2.utils.misc import ... load_video_frames`),
and `init_state` is the only caller. We patch that name once with a
shim that returns a prebuilt tensor when one is staged for this thread
and otherwise defers to the original. Everything else in `init_state`
runs upstream code untouched, so none of the two dozen other state keys
can drift from the version we vendored.

The staging slot is a ContextVar, not a global, because item 3 on the
scaling plan removes the pod's single-flight lock and lets several
cameras into the predictor at once. A module global would tear under
that; a ContextVar is per-thread, which is the granularity the pod's
`asyncio.to_thread` hop actually uses.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging

import numpy as np

log = logging.getLogger(__name__)

# SAM2's own constants (sam2/utils/misc.py load_video_frames defaults).
# Kept here rather than imported so a vendored-code change shows up as a
# mask mismatch in the equivalence test instead of silently altering
# what the model sees.
_IMG_MEAN = (0.485, 0.456, 0.406)
_IMG_STD = (0.229, 0.224, 0.225)

# Prebuilt (images, video_height, video_width) staged for the next
# init_state call on THIS thread. None = fall through to the real loader.
_STAGED: contextvars.ContextVar = contextvars.ContextVar(
    "sam2_staged_frames", default=None,
)

_installed = False


def frames_to_model_tensor(frames_bgr, image_size, device, *, dtype=None):
    """BGR uint8 frames -> the exact tensor `load_video_frames` returns.

    Mirrors `_load_img_as_tensor` + the normalise step:
      PIL open -> convert("RGB") -> resize((S, S)) -> /255 -> CHW
      -> subtract mean, divide std

    Two details that decide whether the masks come out the same:

    * PIL's `Image.resize` defaults to BICUBIC, so we use bicubic here.
      Antialiasing is irrelevant — the pod tracks at 640 wide and the
      model wants 1024, so every axis is an UPSCALE.
    * PIL resamples in uint8 and therefore clamps; bicubic overshoots
      on high-contrast edges, so float bicubic would keep values
      outside [0, 1] that PIL would have clipped. We clamp to match.
    """
    import torch

    if not frames_bgr:
        raise ValueError("frames_to_model_tensor: no frames")

    h, w = frames_bgr[0].shape[:2]
    arr = np.ascontiguousarray(np.stack(frames_bgr))          # N,H,W,3 BGR
    t = torch.from_numpy(arr).to(device, non_blocking=True)
    t = t[..., [2, 1, 0]]                                      # BGR -> RGB
    t = t.permute(0, 3, 1, 2).float().div_(255.0)              # N,3,H,W in [0,1]
    t = torch.nn.functional.interpolate(
        t, size=(image_size, image_size), mode="bicubic", align_corners=False,
    ).clamp_(0.0, 1.0)

    mean = torch.tensor(_IMG_MEAN, device=t.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMG_STD, device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std
    if dtype is not None:
        t = t.to(dtype)
    # video_height/width are the ORIGINAL frame dims — the predictor
    # uses them to size masks back up on output, so they must be the
    # incoming frame's size, not image_size.
    return t, h, w


def install():
    """Patch `load_video_frames` in the predictor's namespace. Idempotent."""
    global _installed
    if _installed:
        return
    from sam2 import sam2_video_predictor as svp

    original = svp.load_video_frames

    def _shim(**kwargs):
        staged = _STAGED.get()
        if staged is not None:
            return staged
        return original(**kwargs)

    # Positional calls would bypass the kwargs-only shim silently. The
    # vendored predictor calls with kwargs; assert it so a future
    # version that switches to positional fails loudly here rather than
    # by quietly reading frames off disk again at 98.7 ms each.
    def _shim_any(*args, **kwargs):
        if args:
            raise TypeError(
                "frame_tensor shim expects keyword args; the vendored "
                "sam2_video_predictor changed its call convention"
            )
        return _shim(**kwargs)

    svp.load_video_frames = _shim_any
    _installed = True
    log.info("frame_tensor.installed — init_state now reads staged tensors")


def prime_feature_cache(predictor, state, chunk=16):
    """Encode every frame in ONE batched pass instead of N passes of 1.

    `SAM2VideoPredictor._get_image_feature` encodes a single unsqueezed
    frame per call and stores it in a cache that holds exactly one entry
    — it REASSIGNS `cached_features`, it does not add to it. So
    propagating N frames runs N batch-of-1 forwards, which is why the
    A40 measures ~17% SM utilisation while we wait on it.

    Priming every frame up front means every lookup is a hit and the
    encoder never runs inside propagation.

    Measured, 12 frames at 640x360 on the A40:

        plain    430.6 ms
        primed   369.4 ms      1.166x
        accuracy vs ground truth  -0.0002 (unchanged)
        mask agreement with the unprimed path  mean 0.9945

    Modest on purpose — the encoder is only ~31% of propagation
    (13 of 42 ms/frame); memory attention and mask decoding are the
    rest, and those are sequential per session. Do NOT expect batching
    to solve cross-camera concurrency: it cannot overlap that 69%.

    Fails SAFE. Any problem building the cache leaves it empty, and
    upstream simply encodes per frame as before — slower, never wrong.
    """
    import torch

    imgs = state.get("images")
    if imgs is None or not hasattr(imgs, "shape"):
        return False
    try:
        cache = {}
        device = state.get("device", imgs.device)
        for start in range(0, int(imgs.shape[0]), chunk):
            block = imgs[start:start + chunk].to(device).float()
            out = predictor.forward_image(block)
            if "backbone_fpn" not in out or "vision_pos_enc" not in out:
                return False          # upstream shape changed — bail
            for i in range(int(block.shape[0])):
                per = dict(out)
                per["backbone_fpn"] = [t[i:i + 1] for t in out["backbone_fpn"]]
                per["vision_pos_enc"] = [t[i:i + 1] for t in out["vision_pos_enc"]]
                cache[start + i] = (block[i:i + 1], per)
        state["cached_features"] = cache
        return True
    except Exception:                 # noqa: BLE001
        log.warning(
            "frame_tensor.prime_failed — falling back to per-frame "
            "encoding (correct, just slower)", exc_info=True,
        )
        state["cached_features"] = {}
        return False


@contextlib.contextmanager
def staged_frames(frames_bgr, predictor, device):
    """Stage `frames_bgr` so the enclosed `init_state()` uses them.

    The path argument still passed to init_state is ignored while
    staged, but callers should pass something harmless — see
    `continuous._propagate`, which no longer creates a temp dir at all.
    """
    install()
    image_size = getattr(predictor, "image_size", 1024)
    tensor = frames_to_model_tensor(frames_bgr, image_size, device)
    token = _STAGED.set(tensor)
    try:
        yield
    finally:
        # Cleared even on error: a leaked staging slot would make the
        # NEXT unrelated init_state on this thread silently reuse these
        # frames, which is the worst failure this file could have.
        _STAGED.reset(token)
