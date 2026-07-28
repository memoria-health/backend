"""Clip censoring: person-mask blur + audio strip (2026-07-27).

When the sensitivity screen (`sensitivity.py`) flags a clip, this
module produces the ONLY version of it that may leave the box: every
person pixelated beyond recognition (all persons, not patient-only —
plan decision ⚑1: a caregiver mid-bathing-assist is a dignity subject
too) and the audio track fully stripped. Macro positions and "what
happened" stay readable; identity and exposure do not.

Two tiers, fail-closed:

  1. Person-mask censor — YOLO11n-seg (ultralytics; already a repo
     dependency via the identity stack) segments the `person` class per
     frame, masks are generously dilated, and the masked region is
     pixelated via OpenCV. A frame of a SENSITIVE clip in which
     segmentation finds NO person is pixelated FULL-FRAME — the seg
     model must never be the weak link (a missed body is exactly the
     failure mode we cannot afford). Re-encoded via ffmpeg with `-an`.
     The pipeline STREAMS: decode one frame → mask → censor → write to
     the piped encoder → drop it, so peak memory is O(1 frame + model)
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
     fallback when tier 1 raises for any reason (ultralytics missing,
     model load failure, decode error, ...).

If tier 2 also fails, `CensorError` propagates and the caller DROPS
the clip entirely (`events.clip_suppressed`) — raw footage never
reaches storage on any path.

CPU cost: clips are ~26s @ 4fps ≈ 100 frames; yolo11n-seg runs a few
ms-to-tens-of-ms per frame on CPU, so a censor pass is seconds — fine
for a fire-and-forget archival task. Deploy note: the seg weights
(`yolo11n-seg.pt`, ~6 MB) auto-download from ultralytics' CDN on first
use; prefetch in the image build (tools/download_models.py pattern) if
the deploy target has no egress.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING

from src.settings import get_settings

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class CensorError(RuntimeError):
    """Both censor tiers failed. The caller must DROP the clip —
    uploading the raw bytes is never an acceptable recovery."""


# YOLO11 nano segmentation weights. Auto-downloaded by ultralytics on
# first construction (~6 MB); cached process-wide below.
_SEG_MODEL_NAME = "yolo11n-seg.pt"
# COCO class id 0 = person.
_PERSON_CLASS_ID = 0
# Deliberately LOW confidence floor: a false-positive mask just blurs a
# bit more of the frame (harmless); a false negative leaves skin
# visible (the failure we exist to prevent).
_SEG_CONFIDENCE = 0.20
# Pixelation factor: the masked region is downsampled by this per axis
# and re-upsampled with nearest-neighbour. 24 → a 720p body becomes a
# handful of flat blocks — posture readable, identity/exposure not.
_PIXELATE_FACTOR = 24
# Heavy full-frame blur for the ffmpeg tier. luma_power=3 stacks the
# box kernel into a near-gaussian; radius 20 at clip resolutions leaves
# only macro shapes.
# radius 28 (was 20): field feedback 2026-07-28 — 20 left too much
# readable at clip resolutions.
_FFMPEG_BLUR_FILTER = (
    "boxblur=luma_radius=28:luma_power=3:chroma_radius=14:chroma_power=3"
)

# Max consecutive detection-dropout frames bridged with the previous
# mask (extra-dilated) before failing safe to full-frame pixelation.
# At 8fps, 3 frames = 375ms of plausible movement — covered by the
# widened dilation. Longer gaps mean the person may genuinely be
# elsewhere in frame → full-frame.
_MASK_GAP_BRIDGE_FRAMES = 3

_seg_model = None


def _get_seg_model():
    """Load + cache the YOLO seg model. Raises if ultralytics (or the
    weights download) is unavailable — the caller falls back to the
    full-frame ffmpeg tier."""
    global _seg_model
    if _seg_model is None:
        import torch
        from ultralytics import YOLO

        # Deadlock guard (2026-07-28): a multi-threaded torch inference
        # froze the entire process (event loop included) in the local
        # Docker VM — classic libomp hang. Single-threaded inference is
        # ~2-3x slower and immune. Belt to the compose-level
        # OMP_NUM_THREADS=1 braces.
        torch.set_num_threads(1)
        _seg_model = YOLO(_SEG_MODEL_NAME)
    return _seg_model


def _person_mask(model, frame: np.ndarray) -> np.ndarray | None:
    """Union of all person masks for one BGR frame, as a uint8 {0,1}
    array at frame resolution. None when no person was segmented."""
    import cv2
    import numpy as np

    results = model.predict(
        frame,
        classes=[_PERSON_CLASS_ID],
        conf=_SEG_CONFIDENCE,
        verbose=False,
    )
    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None
    stacked = result.masks.data.cpu().numpy()  # (n, mh, mw) in [0,1]
    union = (stacked.max(axis=0) > 0.5).astype(np.uint8)
    h, w = frame.shape[:2]
    if union.shape != (h, w):
        union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
    return union if union.any() else None


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


def _censor_frame(
    frame: np.ndarray, mask: np.ndarray | None,
) -> np.ndarray:
    """Censor one frame, pure for unit tests. ``mask`` is the {0,1}
    person union (or None). No person → FULL-FRAME pixelation (the
    sensitive verdict stands even where segmentation sees nothing).
    With a mask: dilate generously — limbs, held items, and seg
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


def _censor_person_mask_sync(in_path: str, out_path: str) -> None:
    """Tier 1, synchronous (runs inside the CLI child process — see
    `_cli_main`): STREAM one frame at a time — decode → person-mask
    pixelation → straight into the piped ffmpeg encoder — so peak
    memory is O(1 frame + model), never O(clip). Audio never enters
    the path (raw-video pipe)."""
    import cv2

    model = _get_seg_model()
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise CensorError(f"censor: cannot decode {in_path}")
    encoder: _FrameEncoder | None = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 4.0  # livekit_worker_target_fps default; sane fallback
        ok, frame = cap.read()
        if not ok:
            raise CensorError("censor: clip decoded to 0 frames")
        h, w = frame.shape[:2]
        encoder = _FrameEncoder(w, h, float(fps), out_path)
        # Gap bridging (2026-07-28 field feedback): a 1-2 frame detection
        # dropout mid-motion flashed the whole frame to pixelation, which
        # reads as flicker. For gaps up to _MASK_GAP_BRIDGE_FRAMES, reuse
        # the last mask dilated wider (at 8fps a 3-frame gap is 375ms —
        # the extra dilation covers plausible movement). Longer losses
        # still fail safe to full-frame.
        last_mask = None
        gap = 0
        while ok:
            mask = _person_mask(model, frame)
            if mask is not None:
                last_mask, gap = mask, 0
            elif last_mask is not None and gap < _MASK_GAP_BRIDGE_FRAMES:
                gap += 1
                mask = _dilate_mask(last_mask, extra=True)
            encoder.write(_censor_frame(frame, mask))
            ok, frame = cap.read()
    except BaseException:
        if encoder is not None:
            encoder.abort()
        raise
    finally:
        cap.release()
    encoder.close()


def _person_mask_subprocess_cmd(in_path: str, out_path: str) -> list[str]:
    """argv for the tier-1 child process. Split out as a seam so unit
    tests can substitute a scripted child (exit-code / sleep stand-ins)
    and exercise the real spawn/kill/timeout mechanics without YOLO."""
    return [
        sys.executable,
        "-m", "src.services.detection.censor",
        in_path,
        out_path,
    ]


async def _censor_person_mask_subprocess(in_path: str, out_path: str) -> None:
    """Tier 1: run the person-mask censor in an ISOLATED child process
    with a hard wall-clock deadline (`censor_timeout_seconds`).

    Why a subprocess (2026-07-28): a multi-threaded torch inference hit
    a native OpenMP deadlock and froze the whole backend — the hang
    lives below Python (no bytecode runs), so `asyncio.to_thread` could
    not contain it and the event loop starved with it. A wedged child
    process, by contrast, is always `kill()`-able. The child inherits
    our cwd, so it finds the seg weights the in-process warmup
    (`main._warm_censor_model`) already downloaded there — but it pays
    model load per invocation (~2-4s with cached weights), which is
    acceptable for a fire-and-forget archival task.

    Raises `TimeoutError` on deadline (child killed + reaped first) and
    `CensorError` on nonzero exit; `censor_clip` maps both onto the
    tier-2 full-frame fallback.
    """
    timeout = get_settings().censor_timeout_seconds
    proc = await asyncio.create_subprocess_exec(
        *_person_mask_subprocess_cmd(in_path, out_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
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


async def _censor_full_frame(in_path: str, out_path: str) -> None:
    """Tier 2: pure-ffmpeg full-frame blur + `-an`. Deliberately no
    OpenCV/ultralytics in this path so a broken CV stack can never take
    the last-resort censor down with it."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-i", in_path,
            "-vf", _FFMPEG_BLUR_FILTER,
            "-an",
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


async def censor_clip(
    in_path: str, out_path: str, *, full_frame: bool = False,
) -> None:
    """Censor the clip at `in_path` into `out_path` (video pixelated/
    blurred, audio stripped).

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
    if not full_frame:
        try:
            await _censor_person_mask_subprocess(in_path, out_path)
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
    await _censor_full_frame(in_path, out_path)


def _cli_main(argv: list[str] | None = None) -> int:
    """`python -m src.services.detection.censor IN OUT [--full-frame]`
    — child-process entry for the isolated tier-1 censor.

    Runs the synchronous censor pipeline and exits 0 on success,
    nonzero on any failure; the parent (`censor_clip`) maps nonzero /
    timeout onto its tier-2 fallback. Heavy imports (torch, cv2,
    ultralytics) are already lazy inside the pipeline functions, so
    importing this module headless stays cheap and side-effect-free —
    no FastAPI app, no DB engine.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.services.detection.censor",
        description="Censor a sensitive clip (blur persons, strip audio).",
    )
    parser.add_argument("in_path", help="source clip (mp4)")
    parser.add_argument("out_path", help="censored output clip (mp4)")
    parser.add_argument(
        "--full-frame",
        action="store_true",
        help="skip person segmentation; blur the whole frame via ffmpeg",
    )
    args = parser.parse_args(argv)
    try:
        if args.full_frame:
            asyncio.run(_censor_full_frame(args.in_path, args.out_path))
        else:
            _censor_person_mask_sync(args.in_path, args.out_path)
    except Exception:  # noqa: BLE001 — any failure = nonzero exit
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_cli_main())
