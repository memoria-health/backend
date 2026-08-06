"""Clip censoring: person-mask blur + audio strip (2026-07-27).

When the sensitivity screen (`sensitivity.py`) flags a clip, this
module produces the ONLY version of it that may leave the box: every
person pixelated beyond recognition (all persons, not patient-only —
plan decision ⚑1: a caregiver mid-bathing-assist is a dignity subject
too) and the audio track fully stripped. Macro positions and "what
happened" stay readable; identity and exposure do not.

Two tiers, fail-closed:

  1. Person-mask censor — the person masks EdgeTAM already computed on
     the perception pod, read back from `GET /masks` (see
     `pod_masks.py`), generously dilated, and the masked region
     pixelated via OpenCV. A frame of a SENSITIVE clip for which no
     mask exists is pixelated FULL-FRAME — the mask source must never
     be the weak link (a missed body is exactly the failure mode we
     cannot afford). Re-encoded via ffmpeg with `-an`.
     The pipeline STREAMS: decode one frame → mask → censor → write to
     the piped encoder → drop it, so peak memory is O(1 frame)
     — decoding a whole 1080p clip up front (~1.2 GB at 200 frames)
     got the child OOM-killed (SIGKILL -9) in the local Docker VM on
     2026-07-28. Runs in an ISOLATED CHILD PROCESS (`python -m
     src.services.detection.censor`, see `_cli_main`) under a hard
     `censor_timeout_seconds` deadline: on 2026-07-28 a tier-1 torch
     inference hit a native OpenMP deadlock and froze the ENTIRE
     backend process, event loop included — a hang below Python that
     `asyncio.to_thread` cannot contain, but `kill()` on a child
     process always can.
  2. Full-frame blur — pure ffmpeg (boxblur + `-an`), no Python CV
     stack in the path. Used when the caller has no trustworthy verdict
     (VLM screen failed → we don't know WHAT is sensitive) and as the
     fallback when tier 1 raises for any reason (pod unreachable,
     decode error, ...).

If tier 2 also fails, `CensorError` propagates and the caller DROPS
the clip entirely (`events.clip_suppressed`) — raw footage never
reaches storage on any path.

WHY THE MASKS COME OFF THE POD (2026-08-06). Tier 1 used to run
yolo11n-seg locally. At normal seated distance it scored ~0.05, and
because every failure path here treats "no mask" as "cover everything",
the operator's 60-second gate page arrived as a solid green rectangle
for its whole duration: unjudgeable, so the alert auto-released
unreviewed and the gate was not doing its job. A bigger model was
measured and rejected — yolo11x-seg managed 7 of 21 frames on the
incident clip for far more compute. Meanwhile EdgeTAM on the pod was
already computing a full mask for every frame of every tick and tracking
the same person reliably, then throwing the masks away. So the segmenter
is GONE rather than demoted to a fallback: a weaker second segmenter
would just preserve the failure being removed, one clip in N.

Cost: one HTTP read per clip (a few hundred KB, run-length encoded) in
place of per-frame inference, so tier 1 is now mask arithmetic only.
The pod must be reachable for it to work at all; when it is not, tier 2
covers the frame. No weights, no torch, no ultralytics in this path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import TYPE_CHECKING, Any

from src.services.detection import pod_masks
from src.settings import get_settings

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class CensorError(RuntimeError):
    """Both censor tiers failed. The caller must DROP the clip —
    uploading the raw bytes is never an acceptable recovery."""


# Pixelation factor: the masked region is downsampled by this per axis
# and re-upsampled with nearest-neighbour. 24 → a 720p body becomes a
# handful of flat blocks — posture readable, identity/exposure not.
_PIXELATE_FACTOR = 24
# Heavy full-frame blur for the ffmpeg tier. luma_power=3 stacks the
# box kernel into a near-gaussian; radius 20 at clip resolutions leaves
# only macro shapes.
# radius 28 (was 20): field feedback 2026-07-28 — 20 left too much
# readable at clip resolutions.
# radius 14 (was 28), 2026-08-03: halved — a full-frame smear left the
# caregiver unable to tell whether the alert was real.
_FFMPEG_BLUR_FILTER = (
    "boxblur=luma_radius=14:luma_power=3:chroma_radius=7:chroma_power=3"
)

# Max consecutive detection-dropout frames bridged with the previous
# mask (extra-dilated) before failing safe to full-frame pixelation.
# At 8fps, 3 frames = 375ms of plausible movement — covered by the
# widened dilation. Longer gaps mean the person may genuinely be
# elsewhere in frame → full-frame.
_MASK_GAP_BRIDGE_FRAMES = 3

# Upper bound on how much wall clock one censored clip can span, used to
# size the pod-mask fetch window before the clip is decoded. Event clips
# are ~26s and the pod retains 45s, so this is slack, not a limit — it
# exists so a bad `clip_start` cannot ask the pod for an unbounded range.
_MAX_CLIP_SECONDS = 120.0

# How far back the door-frame still looks for a mask. The still is live,
# so this only has to cover the pipeline lag between a frame being
# captured and its mask being retained (`lag_s` ≈ 2.6s live) plus a
# tick's worth of slack.
_STILL_LOOKBACK_SECONDS = 10.0

# Ceiling on the motion allowance in `_motion_margin_px`, in seconds.
# Past this the margin would swallow the room and stop being a redaction
# at all; the bounded gap bridge and the full-frame cover take over
# instead. 0.5s at a body-height per second is already half a body of
# slack in every direction.
_MOTION_MARGIN_MAX_S = 0.5

# ── research mask mode (2026-08-04) ─────────────────────────────────
#
# A SECOND, separate treatment used only for the B2 research corpus.
# It is not reachable from the caregiver-facing path and does not change
# the event-clip treatment above in any way.
#
# It fills the person with a flat colour instead of pixelating them, and
# hugs the body far more closely. That is a DELIBERATE REVERSAL of the
# block-quantisation above, and the tradeoff has to be stated plainly:
#
#   The event-clip mask snaps to a block grid precisely so the boundary
#   reads as crude rectangles rather than a figure, because a smooth
#   silhouette leaks the body OUTLINE even at infinite pixelation. This
#   mode leaks that outline ON PURPOSE. Pose and position are exactly
#   what a research corpus needs; appearance and skin are exactly what
#   it must not carry.
#
# So this is not "a better censor" — it is a different exposure trade,
# only defensible because the corpus is opt-in under explicit, separate
# consent and never reaches a caregiver's phone. Anything that would put
# it in front of a caregiver is a bug.
#
# Flat green: it is not a colour a room produces, so a masked region is
# unambiguous to a human reviewer and trivially separable to a model.
_RESEARCH_FILL_BGR = (0, 255, 0)
# Margin around the masked person. 2% of the short side is ~21px on a
# 1080p frame, which rounds off fingers and closes the gaps between limbs
# — the silhouette reads as a blob rather than a person. That size was
# compensating for a mask that arrived at ~160x160 and was upscaled with
# nearest-neighbour; `pod_masks` interpolates the soft mask before
# thresholding, so the mask itself is accurate to a few pixels and the
# margin no longer has to cover for it.
#
# 0.5% (~5px at 1080p) still covers the under-coverage the masks show
# at body edges, especially loose clothing and hair. It is widened at
# runtime by `_motion_margin_px` to cover the time gap between the
# masked instant and the frame being censored (2026-08-06). Raise this
# base if any real footage shows skin outside the fill; that is the
# failure that matters, and it is the reason this is a named constant
# rather than a literal.
_RESEARCH_DILATE_FRACTION = 0.005
_RESEARCH_DILATE_MIN_PX = 3

CENSOR_MODE_PIXELATE = "pixelate"  # event clips (default, unchanged)
CENSOR_MODE_RESEARCH = "research"  # B2 corpus only


def _person_mask(
    window: "pod_masks.MaskWindow",
    frame_time: float,
    shape: tuple[int, int],
) -> np.ndarray | None:
    """Union of every tracked person's pod mask at `frame_time`, as a
    uint8 {0,1} array at `shape` (height, width). None when the pod
    retained nothing near that time.

    All the real work — decoding at the RECORDED dims, unioning in
    float, upscaling before thresholding so the outline stays smooth —
    lives in `pod_masks`, because the live recorder path needs exactly
    the same arithmetic against a different lookup.
    """
    return window.mask_at(frame_time, shape)


def _pixelate(frame: np.ndarray, block: int = _PIXELATE_FACTOR) -> np.ndarray:
    """Whole-frame pixelation (down/upsample with nearest-neighbour).
    ``block`` is the output block size in pixels."""
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(
        frame,
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def _dilate_mask(mask: np.ndarray, *, extra: bool = False) -> np.ndarray:
    """Dilate a {0,1} person mask. ``extra`` doubles the kernel — used
    when bridging a detection-dropout frame with the PREVIOUS frame's
    mask, where the person may have moved since."""
    import cv2

    h, w = mask.shape[:2]
    k = max(15, int(min(h, w) * (0.12 if extra else 0.06))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(mask, kernel) > 0).astype(mask.dtype)


def _motion_margin_px(gap_seconds: float | None, mask: np.ndarray) -> int:
    """Extra dilation to cover how far the subject may have MOVED between
    the masked instant and this frame.

    Needed since 2026-08-06. The old segmenter ran on the frame being
    censored, so its mask was exact and a few pixels of margin were
    enough. Pod masks are retained at the identity service's tracking
    cadence (~3fps) while clips encode at `clip_fps` (8), so a frame is
    routinely covered by a silhouette from up to ~170ms away — about
    63px of walking at 1080p, against a 5px research margin. That is
    uncovered skin at the trailing edge, which is the one failure this
    module exists to prevent.

    Scaled by the subject's OWN height in pixels rather than the frame's,
    so it tracks distance from the camera for free: someone across the
    room is smaller in frame and also moves fewer pixels per second. A
    fast indoor walk is roughly one body height per second, which is the
    bound used here.
    """
    import numpy as np

    if not gap_seconds:
        return 0
    ys = np.nonzero(mask.any(axis=1))[0]
    if ys.size == 0:
        return 0
    body_px = int(ys[-1] - ys[0] + 1)
    return int(min(gap_seconds, _MOTION_MARGIN_MAX_S) * body_px)


def _research_dilate(mask: np.ndarray, *, extra_px: int = 0) -> np.ndarray:
    """Tight dilation for the research mode — a real but small margin,
    widened by `extra_px` to cover subject motion since the mask was
    computed (see `_motion_margin_px`)."""
    import cv2

    h, w = mask.shape[:2]
    k = max(
        _RESEARCH_DILATE_MIN_PX,
        int(min(h, w) * _RESEARCH_DILATE_FRACTION),
    ) + max(0, int(extra_px))
    k |= 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return (cv2.dilate(mask, kernel) > 0).astype(mask.dtype)


def _research_frame(
    frame: np.ndarray, mask: np.ndarray | None, *, motion_px: int = 0,
) -> np.ndarray:
    """Fill the person with flat colour, keeping the room intact.

    No mask means the pod retained none, not that the room is empty — the
    gate upstream already established someone is present. So the frame
    is filled entirely rather than left as-is. One rule, no special
    cases: whether we mask at all is decided upstream by the sensitivity
    verdict, and once we are masking, every frame gets covered one way or
    another.
    """
    out = frame.copy()
    if mask is None or not mask.any():
        out[:] = _RESEARCH_FILL_BGR
        return out
    out[_research_dilate(mask, extra_px=motion_px) > 0] = _RESEARCH_FILL_BGR
    return out


def _censor_frame(
    frame: np.ndarray, mask: np.ndarray | None,
) -> np.ndarray:
    """Censor one frame, pure for unit tests. ``mask`` is the {0,1}
    person union (or None). No person → FULL-FRAME pixelation (the
    sensitive verdict stands even where no mask exists).
    With a mask: dilate generously — limbs, held items, and mask
    under-coverage at body edges must land inside the censored region —
    then pixelate only that region."""
    import cv2
    import numpy as np

    if mask is None or not mask.any():
        return _pixelate(frame)
    h, w = frame.shape[:2]
    # ~6% of the short side, odd-sized; ≥15 px so tiny/distant persons
    # still get real margin.
    k = max(15, int(min(h, w) * 0.06)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask, kernel) > 0
    # PERSON-RELATIVE block size (2026-07-28 field feedback): fixed
    # 24px blocks gave a close-up body dozens of blocks — more detail
    # exactly when most exposed. Scale so the person renders as ~8
    # blocks across their bounding box's short side, regardless of
    # distance.
    ys, xs = np.where(dilated)
    bbox_short = min(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)
    block = max(_PIXELATE_FACTOR, int(bbox_short) // 8)
    pixelated = _pixelate(frame, block)
    # BLOCK-QUANTIZED region: censoring the smooth silhouette leaks the
    # body OUTLINE even at infinite pixelation. Snap the region to the
    # block grid (any block touching the mask censors fully) so the
    # boundary reads as crude rectangles, not a figure.
    small = cv2.resize(
        dilated.astype(np.uint8),
        (max(1, w // block), max(1, h // block)),
        interpolation=cv2.INTER_AREA,
    )
    blocky = cv2.resize(
        (small > 0).astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    out = frame.copy()
    out[blocky] = pixelated[blocky]
    return out


class _FrameEncoder:
    """Incremental raw-BGR → H.264 MP4 writer over a piped ffmpeg.

    Opened once (dimensions + fps fixed up front), fed one frame at a
    time via `write`, finalized via `close` — so the tier-1 pipeline
    never holds more than one frame in memory (a 200-frame 1080p clip
    fully decoded is ~1.2 GB of BGR; that OOM-killed the censor child
    in the local Docker VM on 2026-07-28).

    Pipe-deadlock note: we block writing ffmpeg's stdin, so its stdout/
    stderr must never back up against a full, undrained pipe. stdout is
    unused (the MP4 goes to `out_path`) → DEVNULL; stderr goes to an
    unlinked temp FILE (kernel drains it, no reader thread needed) and
    is read back after exit for the error message.

    `-an` is redundant (raw video input carries no audio) but stays as
    the explicit statement of the audio-strip contract."""

    def __init__(
        self, width: int, height: int, fps: float, out_path: str,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", f"{fps:g}",
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
        ]
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr,
            )
        except FileNotFoundError as exc:
            self._stderr.close()
            raise CensorError("ffmpeg binary not found on PATH") from exc

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass  # ffmpeg died early; close() surfaces the exit code

    def close(self) -> None:
        """Finish the encode. Nonzero ffmpeg exit → CensorError."""
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        returncode = self._proc.wait()
        try:
            self._stderr.seek(0)
            err = self._stderr.read().decode("utf-8", errors="replace").strip()
        finally:
            self._stderr.close()
        if returncode != 0:
            raise CensorError(
                f"ffmpeg encode exited {returncode}: {err or '(no stderr)'}"
            )

    def abort(self) -> None:
        """Kill + reap ffmpeg (mid-stream pipeline failure — the
        partial output is garbage anyway). Never raises."""
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        self._proc.wait()
        self._stderr.close()


def _censor_person_mask_sync(
    in_path: str, out_path: str, *, mode: str = CENSOR_MODE_PIXELATE,
    camera_id: str, clip_start: float, pod_url: str, pod_token: str = "",
) -> None:
    """Tier 1, synchronous (runs inside the CLI child process — see
    `_cli_main`): STREAM one frame at a time — decode → person-mask
    pixelation → straight into the piped ffmpeg encoder — so peak
    memory is O(1 frame), never O(clip). Audio never enters the path
    (raw-video pipe).

    `clip_start` is the EPOCH capture time of the clip's first frame and
    `camera_id` the backend camera UUID. The pod's masks are fetched ONCE
    for the whole window before the decode loop: this runs per frame, and
    an HTTP read per frame would turn a seconds-long pass into a
    minutes-long one against the gate's 60s budget.
    """
    import cv2

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise CensorError(f"censor: cannot decode {in_path}")
    encoder: _FrameEncoder | None = None
    try:
        # Trust our OWN encoder, not the container's self-report. This
        # function is only ever called on a clip this system just
        # encoded (single caller: clip_archival), and we encode at
        # clip_fps — so the rate is known, not something to discover.
        #
        # Reading it from the file was the weaker choice twice over.
        # The old fallback was a hardcoded 4.0, the ANALYSIS rate,
        # which is half the real clip rate — so any failed read wrote a
        # half-speed censored copy with the audio running twice as long
        # as the video. And the guard around it only rejected zero,
        # negative, or absurd values: a wrong-but-plausible reading
        # sailed straight through (a variable-frame-rate file reports
        # 0.25 fps — measured), which would have written a clip 32x too
        # slow with nothing flagged anywhere.
        fps = float(get_settings().clip_fps)
        probed = cap.get(cv2.CAP_PROP_FPS)
        if probed and abs(probed - fps) > 0.5:
            log.warning(
                "censor: clip reports %.2f fps but clips are encoded at "
                "%.2f — using the encoder's rate", probed, fps,
            )
        ok, frame = cap.read()
        if not ok:
            raise CensorError("censor: clip decoded to 0 frames")
        h, w = frame.shape[:2]
        encoder = _FrameEncoder(w, h, float(fps), out_path)
        # ONE fetch for the whole clip. `frames_to_vfr_mp4` materializes
        # the window onto a uniform fps grid whose playback duration
        # equals wall clock, so output frame i sits at
        # clip_start + i/fps — which is what makes a time lookup legal
        # here at all. A PodMaskError propagates: tier 2 covers the clip
        # rather than tier 1 shipping it unmasked.
        window = pod_masks.fetch_window(
            camera_id, clip_start, clip_start + _MAX_CLIP_SECONDS,
            base_url=pod_url, token=pod_token,
            timeout=float(get_settings().perception_mask_timeout_s),
        )
        if window.empty:
            log.warning(
                "censor.no_pod_masks cam=%s start=%.3f — every frame will "
                "be covered", camera_id, clip_start,
            )
        # Gap bridging (2026-07-28 field feedback): a 1-2 frame detection
        # dropout mid-motion flashed the whole frame to pixelation, which
        # reads as flicker. For gaps up to _MASK_GAP_BRIDGE_FRAMES, reuse
        # the last mask dilated wider (at 8fps a 3-frame gap is 375ms —
        # the extra dilation covers plausible movement). Longer losses
        # still fail safe to full-frame.
        last_mask = None
        gap = 0
        index = 0
        while ok:
            frame_time = clip_start + index / fps
            mask = _person_mask(window, frame_time, (h, w))
            # How stale the mask we got actually is. Pod masks arrive at
            # the tracking cadence (~3fps) and clips encode at clip_fps,
            # so a matched mask is routinely up to ~170ms old; the margin
            # below covers the movement in between. None → the bridge
            # supplied the mask, which is already extra-dilated.
            match_gap = window.match_gap(frame_time)
            index += 1
            if mask is not None:
                last_mask, gap = mask, 0
            elif last_mask is not None and gap < _MASK_GAP_BRIDGE_FRAMES:
                gap += 1
                mask = _dilate_mask(last_mask, extra=True)
                match_gap = None
            if mode == CENSOR_MODE_RESEARCH:
                censored = _research_frame(
                    frame, mask,
                    motion_px=(
                        _motion_margin_px(match_gap, mask)
                        if mask is not None else 0
                    ),
                )
                if censored is None:
                    # No subject found and the gap bridge is exhausted.
                    # Research mode has no safe fallback (see
                    # `_research_frame`), so the whole clip goes.
                    raise CensorError(
                        "censor: research mode could not locate the "
                        "subject; clip must be dropped"
                    )
            else:
                censored = _censor_frame(frame, mask)
            encoder.write(censored)
            ok, frame = cap.read()
    except BaseException:
        if encoder is not None:
            encoder.abort()
        raise
    finally:
        cap.release()
    encoder.close()


def mask_people_in_still_sync(
    jpeg: bytes, *, camera_id: str, pod_url: str, pod_token: str = "",
    frame_time: float | None = None,
) -> bytes:
    """Pixelate PEOPLE in one still, leaving the room itself visible.

    For the exit-door picker on a PRIVATE camera. The caregiver has to
    see the room to tap its doors, but privacy mode exists so they
    cannot watch that room — and a still they can re-fetch is a
    slideshow of it. Masking the people resolves both: doors are static
    architecture, people are the thing being protected.

    DIFFERENT FAIL RULE from `_censor_frame`, deliberately. There, a
    frame with no mask is pixelated FULL-FRAME, because the clip already
    carries a sensitive verdict and a missed body is the unaffordable
    failure. Here the room IS the subject, so full-frame pixelation would
    defeat the feature entirely. No person found → return the frame
    unchanged, which is correct: an empty bedroom is architecture.

    THE CALLER OWNS "is the room empty?" (2026-08-06). This function used
    to segment the still itself, so it could answer that question from
    the pixels. Pod masks cannot: the pod records nothing at all for a
    tick with no people, so "no masks" reads identically whether the room
    is empty or the sidecar has stopped watching this camera — and those
    two must not produce the same picture. `api/cameras.door_frame`
    settles it against the identity service's /state before calling here,
    and only calls when someone is actually being tracked. Reaching this
    function therefore means "mask a person who IS there", and failing to
    find one raises so the caller serves nothing.

    Being unable to reach the pod raises too, by design.
    """
    import cv2
    import numpy as np

    arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise CensorError("door frame: could not decode JPEG")

    # The still is whatever is in the ingest buffer right now, so "now"
    # is its time. The tolerance is the fetch window's, not a frame
    # match: the newest retained mask is the right one for a live still.
    now = time.time() if frame_time is None else frame_time
    window = pod_masks.fetch_window(
        camera_id, now - _STILL_LOOKBACK_SECONDS, now,
        base_url=pod_url, token=pod_token,
        timeout=float(get_settings().perception_mask_timeout_s),
    )
    mask = window.newest(arr.shape[:2])
    if mask is None or not mask.any():
        # The caller established someone is tracked here, so no mask
        # means the pod's window is stale or the person left between the
        # two reads — either way we cannot promise this still is masked.
        raise CensorError("door frame: no pod mask for a tracked person")

    # Same treatment as the event path: generous dilation, person-relative
    # block size, block-quantised region so the silhouette does not leak.
    out = _censor_frame(arr, mask)
    ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise CensorError("door frame: could not re-encode JPEG")
    return bytes(buf)


def mask_stream_sync(
    width: int, height: int, fps: float, out_path: str, stdin: Any = None,
    *, camera_id: str, pod_url: str, pod_token: str = "",
    feed: Any = None,
) -> None:
    """Mask raw I420 frames arriving on stdin, straight into an MP4.

    The inline half of the research censor. The recorder already holds
    decoded frames, so piping them here skips the decode AND the
    re-encode a post-pass over a finished clip would need — and raw
    footage never reaches disk.

    A ROLLING window, not one prefetch (2026-08-06). The file censor
    knows its clip's bounds and reads the pod once; this does not. A
    recorder session runs up to 900s, the pod retains 45s, and — the part
    that decides the design — the mask for the frame being written right
    now does not exist yet. It is computed on the pod a couple of seconds
    after capture (`lag_s` ≈ 2.6s live). So `pod_masks.MaskFeed` re-reads
    a short trailing window on a background thread and this loop masks
    with the freshest mask available, EXTRA-DILATED because it is by
    construction that much stale. Reading per frame is not an option:
    this runs inline at 15fps.

    `censor_stream_segment_every` now paces the feed's polling rather
    than local inference — same knob, same meaning (how often the mask
    is refreshed), one HTTP read instead of a segmentation pass.

    When the feed has nothing, the SAME bounded bridge the file-based
    censor uses applies (`_MASK_GAP_BRIDGE_FRAMES`): reuse the last mask,
    widened, for a few frames, then cover the frame entirely. Carrying a
    stale mask indefinitely would be worse than useless — a person who
    moved would be uncovered while we masked where they used to be.

    Note this bites during the recorder's 45s linger, where the gate has
    already gone false and the room may genuinely be empty: those frames
    get covered rather than kept. Safe, and cheap in research value,
    since an empty room at the tail of a session carries little.

    `feed` is a seam for tests — a stand-in with `.freshest(shape)`.
    """
    import cv2
    import numpy as np

    stream = stdin if stdin is not None else sys.stdin.buffer
    frame_bytes = width * height * 3 // 2
    every = max(1, int(get_settings().censor_stream_segment_every))
    owns_feed = feed is None
    if owns_feed:
        feed = pod_masks.MaskFeed(
            camera_id,
            base_url=pod_url,
            token=pod_token,
            timeout=float(get_settings().perception_mask_timeout_s),
            interval_seconds=every / float(fps) if fps else 0.33,
        )
        feed.start()

    encoder = _FrameEncoder(width, height, float(fps), out_path)
    last_mask = None
    gap = 0
    index = 0
    try:
        while True:
            buf = stream.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break  # clean EOF, or a torn final frame we won't trust
            yuv = np.frombuffer(buf, dtype=np.uint8).reshape(
                (height * 3 // 2, width)
            )
            frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
            if index % every == 0:
                fresh = feed.freshest((height, width))
                if fresh is not None:
                    last_mask, gap = fresh, 0
                else:
                    gap += 1
            # Always extra-dilated, unlike the file path's gap==0 case.
            # Even the freshest pod mask describes where the person was
            # ~lag_s ago, so there is no such thing here as a mask that
            # matches the frame exactly.
            mask = None
            if last_mask is not None and gap <= _MASK_GAP_BRIDGE_FRAMES:
                mask = _dilate_mask(last_mask, extra=True)
            encoder.write(_research_frame(frame, mask))
            index += 1
    except BaseException:
        encoder.abort()
        raise
    finally:
        if owns_feed:
            feed.close()
    encoder.close()


def _person_mask_subprocess_cmd(
    in_path: str, out_path: str, *, mode: str = CENSOR_MODE_PIXELATE,
    camera_id: str, clip_start: float, pod_url: str,
) -> list[str]:
    """argv for the tier-1 child process. Split out as a seam so unit
    tests can substitute a scripted child (exit-code / sleep stand-ins)
    and exercise the real spawn/kill/timeout mechanics.

    NOTE WHAT IS NOT HERE: the pod token. argv is world-readable in the
    process table (`ps` on the host shows every backend child's full
    command line), so a credential passed this way is published to every
    local user for the life of the process. It travels in the child's
    ENVIRONMENT instead — see `_censor_child_env`. Everything on this
    line is non-secret: a path, a camera UUID, a timestamp, an internal
    URL.
    """
    argv = [
        sys.executable,
        "-m", "src.services.detection.censor",
        in_path,
        out_path,
        "--camera-id", str(camera_id),
        "--clip-start", repr(float(clip_start)),
        "--pod-url", pod_url,
    ]
    if mode != CENSOR_MODE_PIXELATE:
        argv += ["--mode", mode]
    return argv


# Env name the censor child reads its pod credential from. Same name on
# the pod (`NEMARA_PERCEPTION_TOKEN` in perception/server.py); the
# backend's own setting is NEMARA_PERCEPTION_MASK_TOKEN, and this is the
# hand-off between them.
_POD_TOKEN_ENV = "NEMARA_PERCEPTION_TOKEN"


def _censor_child_env(token: str) -> dict[str, str]:
    """Environment for the tier-1 child: ours, plus the pod token.

    Built explicitly rather than relying on plain inheritance, because
    the backend's setting can come from a `.env` file that was never
    exported into the process environment — inheritance would then hand
    the child an empty token and every censor would 401 with nothing
    obviously wrong. An empty token removes the variable rather than
    setting it blank, so "unset" means the same thing on both sides.
    """
    env = dict(os.environ)
    if token:
        env[_POD_TOKEN_ENV] = token
    else:
        env.pop(_POD_TOKEN_ENV, None)
    return env


async def _censor_person_mask_subprocess(
    in_path: str, out_path: str, *, mode: str = CENSOR_MODE_PIXELATE,
    camera_id: str, clip_start: float,
) -> None:
    """Tier 1: run the person-mask censor in an ISOLATED child process
    with a hard wall-clock deadline (`censor_timeout_seconds`).

    Why a subprocess (2026-07-28): a multi-threaded torch inference hit
    a native OpenMP deadlock and froze the whole backend — the hang
    lives below Python (no bytecode runs), so `asyncio.to_thread` could
    not contain it and the event loop starved with it. A wedged child
    process, by contrast, is always `kill()`-able. Torch is no longer in
    this path at all now that masks come off the pod, but the isolation
    stays: it also bounds a wedged HTTP read and a runaway ffmpeg, and
    costs a fork we were already paying.

    The child needs network and config it never needed before. Config
    rides argv; the credential rides the environment (see
    `_person_mask_subprocess_cmd`).

    Raises `TimeoutError` on deadline (child killed + reaped first) and
    `CensorError` on nonzero exit; `censor_clip` maps both onto the
    tier-2 full-frame fallback.
    """
    settings = get_settings()
    timeout = settings.censor_timeout_seconds
    proc = await asyncio.create_subprocess_exec(
        *_person_mask_subprocess_cmd(
            in_path, out_path, mode=mode, camera_id=camera_id,
            clip_start=clip_start, pod_url=settings.perception_mask_url,
        ),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
        env=_censor_child_env(settings.perception_mask_token),
    )
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass  # lost the race: child exited right at the deadline
        await proc.wait()  # reap — never leave a zombie behind
        raise TimeoutError(
            f"person-mask censor subprocess exceeded {timeout:g}s "
            "(killed)"
        ) from None
    if proc.returncode != 0:
        err = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise CensorError(
            f"person-mask censor subprocess exited {proc.returncode}: "
            f"{err or '(no stderr)'}"
        )


async def _censor_full_frame(
    in_path: str, out_path: str, *, keep_audio: bool = True,
) -> None:
    """Tier 2: pure-ffmpeg full-frame blur, source audio kept
    (2026-07-28: censored clips carry audio — product call; the visual
    is censored, the cry for help must stay audible). `keep_audio=False`
    drops the track instead — the intimate-situation carve-out.
    Deliberately no OpenCV/ultralytics in this path so a broken CV stack
    can never take the last-resort censor down with it."""
    audio_args = (
        ["-map", "0:a?", "-c:a", "copy"] if keep_audio else ["-an"]
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", in_path,
            "-vf", _FFMPEG_BLUR_FILTER,
            "-map", "0:v", *audio_args,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise CensorError("ffmpeg binary not found on PATH") from exc
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise CensorError(
            f"ffmpeg full-frame blur exited {proc.returncode}: "
            f"{err or '(no stderr)'}"
        )


async def _mux_source_audio(source_path: str, video_path: str) -> None:
    """Copy the SOURCE clip's audio track onto a censored video-only
    file, in place (2026-07-28: censored clips carry audio — the visual
    is censored, the cry for help must stay audible). Best-effort: a
    source with no audio track muxes cleanly to a silent clip via the
    optional `0:a?` map, and any mux failure leaves the silent censored
    video as-is (a censored clip without audio beats no clip)."""
    import tempfile as _tempfile

    tmp = _tempfile.mktemp(
        suffix=".mp4", dir=os.path.dirname(video_path) or None,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-i", source_path,
            "-map", "0:v", "-map", "1:a?",
            "-c", "copy", "-movflags", "+faststart",
            tmp,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
        if rc == 0 and os.path.getsize(tmp) > 0:
            os.replace(tmp, video_path)
        else:
            log.warning(
                "censor.audio_mux_failed rc=%s — shipping censored video "
                "without audio", rc,
            )
    except Exception:
        log.warning(
            "censor.audio_mux_failed — shipping censored video without "
            "audio", exc_info=True,
        )
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


async def censor_clip(
    in_path: str, out_path: str, *, full_frame: bool = False,
    keep_audio: bool = True, mode: str = CENSOR_MODE_PIXELATE,
    camera_id: str, clip_start: float,
) -> None:
    """Censor the clip at `in_path` into `out_path` (video pixelated/
    blurred; the SOURCE audio track is carried over — 2026-07-28
    product call: visuals are censored, sound stays).

    `camera_id` (backend UUID) and `clip_start` (EPOCH capture time of
    the clip's first frame) locate this clip's masks on the pod. They are
    REQUIRED rather than defaulted: a wrong-but-plausible default here
    would silently find no masks and cover every frame, which is exactly
    the failure this path exists to remove, and it would look like a
    rendering problem rather than a missing argument.

    ``keep_audio=False`` ships the clip SILENT. One caller uses it:
    an intimate-situation verdict, where the audio is itself the
    exposure and carrying it would defeat the blur
    (`sensitivity.SILENT_CATEGORIES`). Everywhere else audio stays,
    because a cry for help is exactly what must remain audible.

    ``full_frame=True`` skips tier 1 entirely — used when the caller has
    no per-clip verdict to trust (sensitivity screen failed), so the
    only safe treatment is blurring everything. Otherwise tier 1
    (person-mask, isolated subprocess) is attempted and ANY failure —
    timeout, nonzero exit, spawn error — falls back to tier 2, which
    stays in-process: it is a plain ffmpeg subprocess with no torch in
    the path, immune to the native hang that motivated tier-1
    isolation. Raises `CensorError` only when tier 2 also fails — at
    which point the caller must drop the clip.
    """
    if mode == CENSOR_MODE_RESEARCH:
        # No tier-2 for research. Full-frame blur is useless to a corpus,
        # so the only outcomes are "masked correctly" or "dropped" —
        # CensorError propagates and the caller discards the clip.
        if full_frame:
            raise CensorError(
                "censor: research mode has no full-frame fallback"
            )
        await _censor_person_mask_subprocess(
            in_path, out_path, mode=mode,
            camera_id=camera_id, clip_start=clip_start,
        )
        if keep_audio:
            await _mux_source_audio(in_path, out_path)
        return

    if not full_frame:
        try:
            await _censor_person_mask_subprocess(
                in_path, out_path,
                camera_id=camera_id, clip_start=clip_start,
            )
            # Tier 1 writes video only, so silence is simply not muxing.
            if keep_audio:
                await _mux_source_audio(in_path, out_path)
            return
        except TimeoutError as exc:
            log.warning(
                "censor.person_mask_timeout in=%s — %s; falling back to "
                "full-frame blur", in_path, exc,
            )
        except Exception:
            log.exception(
                "censor.person_mask_failed in=%s — falling back to "
                "full-frame blur", in_path,
            )
    await _censor_full_frame(in_path, out_path, keep_audio=keep_audio)


def _cli_main(argv: list[str] | None = None) -> int:
    """`python -m src.services.detection.censor IN OUT [--full-frame]`
    — child-process entry for the isolated tier-1 censor.

    Runs the synchronous censor pipeline and exits 0 on success,
    nonzero on any failure; the parent (`censor_clip`) maps nonzero /
    timeout onto its tier-2 fallback. Heavy imports (cv2, httpx) are
    lazy inside the pipeline functions, so importing this module
    headless stays cheap and side-effect-free — no FastAPI app, no DB
    engine.

    The pod TOKEN is not an argument and must not become one: it is read
    from `NEMARA_PERCEPTION_TOKEN` in the environment the parent built
    (`_censor_child_env`), because argv is visible to every local user
    via `ps`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.services.detection.censor",
        description="Censor a sensitive clip (blur persons, strip audio).",
    )
    parser.add_argument("in_path", help="source clip (mp4), or - to stream")
    parser.add_argument("out_path", help="censored output clip (mp4)")
    parser.add_argument(
        "--stream-i420",
        metavar="WxH@FPS",
        help="read raw I420 frames from stdin instead of decoding in_path "
             "(inline recorder masking); in_path is ignored",
    )
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="skip person masking; blur the whole frame via ffmpeg",
    )
    parser.add_argument(
        "--mode",
        choices=[CENSOR_MODE_PIXELATE, CENSOR_MODE_RESEARCH],
        default=CENSOR_MODE_PIXELATE,
        help="pixelate (caregiver-facing clips) or research (B2 corpus)",
    )
    parser.add_argument(
        "--camera-id",
        default="",
        help="backend camera UUID — locates this camera's masks on the pod",
    )
    parser.add_argument(
        "--clip-start",
        type=float,
        default=0.0,
        help="epoch capture time of the clip's first frame",
    )
    parser.add_argument(
        "--pod-url",
        default="",
        help="perception pod base URL (GET /masks)",
    )
    args = parser.parse_args(argv)
    pod_token = os.environ.get(_POD_TOKEN_ENV, "")
    try:
        if args.stream_i420:
            dims, _, rate = args.stream_i420.partition("@")
            w, _, h = dims.partition("x")
            mask_stream_sync(
                int(w), int(h), float(rate), args.out_path,
                camera_id=args.camera_id, pod_url=args.pod_url,
                pod_token=pod_token,
            )
        elif args.full_frame:
            asyncio.run(_censor_full_frame(args.in_path, args.out_path))
        else:
            _censor_person_mask_sync(
                args.in_path, args.out_path, mode=args.mode,
                camera_id=args.camera_id, clip_start=args.clip_start,
                pod_url=args.pod_url, pod_token=pod_token,
            )
    except Exception:  # noqa: BLE001 — any failure = nonzero exit
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
