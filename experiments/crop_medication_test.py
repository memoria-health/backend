"""MVE crop experiment v2 — FP-RATE test under prod conditions.

Measures the medication false-positive RATE, native vs crop, holding everything
else at prod:
  - gemini-3.1-flash-lite, media_resolution=high, fps=4
  - motion overlay ON (real `overlay_motion`, the red frame-diff tint)
  - the full prod detector set in ONE prompt (not medication alone)
  - N independent runs each (VLM is stochastic — a single verdict is noise)

Two variants, crop is the ONLY difference:
  V0  overlay(native whole scene)
  V1  overlay(TRACKING crop) — fixed-size window whose center follows the person
      frame-by-frame (EMA-smoothed), so a moving subject stays centered + zoomed
      instead of ballooning the box like a static union crop would.

Writes the intermediate mp4s for inspection:
  scratch_v0_native_overlay.mp4
  scratch_v1_trackcrop.mp4          (crop only, no overlay — to eyeball the track)
  scratch_v1_trackcrop_overlay.mp4  (what the crop variant actually feeds Gemini)

Usage:  PYTHONPATH=. uv run python experiments/crop_medication_test.py [clip.mp4] [N]
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.services.detection.catalog import by_key
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.prompt import DetectorSpec
from src.services.detection.providers.gemini import gemini_provider
from src.settings import get_settings

_BACKEND = Path(__file__).resolve().parents[1]
# Intermediates go NEXT TO the input clip (keep it in the session scratchpad,
# NOT the iCloud/OneDrive-synced repo — scratch files there get swept away).
CLIP = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else "medclip_fp.mp4"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
WEIGHTS = str(_BACKEND / "yolo11s.pt")
_OUT = Path(CLIP).parent
# The 7 detectors that were live on this patient (from the prod tick logs).
DETECTOR_KEYS = ["calling_for_help", "emergency", "fall", "agitation",
                 "medication", "choking", "seizure"]
CROP_PAD = 1.5            # crop size = 90th-pct person box * this
EMA = 0.30               # center smoothing (lower = smoother/laggier)


def _even(v: int) -> int:
    return v - (v % 2)


def track_person(clip: str, sample_every: int = 2):
    """Per-frame largest-person centroid (None when YOLO misses) + box sizes."""
    model = YOLO(WEIGHTS)
    cap = cv2.VideoCapture(clip)
    W, H = int(cap.get(3)), int(cap.get(4))
    centers: list[tuple[float, float] | None] = []
    sizes: list[tuple[float, float]] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        c = None
        if i % sample_every == 0:
            best, area = None, 0.0
            for r in model.predict(frame, classes=[0], verbose=False):
                if r.boxes is not None:
                    for b in r.boxes.xyxy.cpu().numpy():
                        a = (b[2] - b[0]) * (b[3] - b[1])
                        if a > area:
                            area, best = a, b
            if best is not None:
                c = ((best[0] + best[2]) / 2, (best[1] + best[3]) / 2)
                sizes.append((best[2] - best[0], best[3] - best[1]))
        centers.append(c)
        i += 1
    cap.release()
    return centers, sizes, W, H


def make_tracking_crop(clip: str, out: str) -> tuple[int, int]:
    centers, sizes, W, H = track_person(clip)
    if not sizes:
        raise RuntimeError("YOLO found no person — cannot crop")
    sw = _even(min(W, int(np.percentile([s[0] for s in sizes], 90) * CROP_PAD)))
    sh = _even(min(H, int(np.percentile([s[1] for s in sizes], 90) * CROP_PAD)))

    # forward-fill missing centers, backfill the leading gap, then EMA-smooth.
    filled, last = [], None
    for c in centers:
        if c is not None:
            last = c
        filled.append(last)
    seed = next((c for c in filled if c is not None), (W / 2, H / 2))
    filled = [c if c is not None else seed for c in filled]
    smooth, sx, sy = [], filled[0][0], filled[0][1]
    for cx, cy in filled:
        sx, sy = (1 - EMA) * sx + EMA * cx, (1 - EMA) * sy + EMA * cy
        smooth.append((sx, sy))

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 4
    tmp = out + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sw, sh))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cx, cy = smooth[min(i, len(smooth) - 1)]
        x = max(0, min(W - sw, int(cx - sw / 2)))
        y = max(0, min(H - sh, int(cy - sh / 2)))
        vw.write(frame[y:y + sh, x:x + sw])
        i += 1
    cap.release()
    vw.release()
    _mux_audio(tmp, clip, out)
    return sw, sh


def apply_overlay(clip: str, out: str) -> None:
    """Reproduce the prod red-tint overlay via the real `overlay_motion`."""
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 4
    snaps: list[FrameSnapshot] = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = _even(bgr.shape[0]), _even(bgr.shape[1])
        bgr = bgr[:h, :w]
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        snaps.append(FrameSnapshot(w, h, i420.tobytes(), float(len(snaps))))
    cap.release()
    over = overlay_motion(snaps)
    W0, H0 = over[0].width, over[0].height
    tmp = out + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W0, H0))
    for f in over:
        arr = np.frombuffer(f.data, np.uint8).reshape(f.height * 3 // 2, f.width)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420))
    vw.release()
    _mux_audio(tmp, clip, out)


def _mux_audio(video_only: str, audio_src: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_only, "-i", audio_src,
         "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", out],
        check=True, capture_output=True,
    )
    Path(video_only).unlink(missing_ok=True)


def _probe(path: str) -> str:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _specs() -> list[DetectorSpec]:
    out = []
    for k in DETECTOR_KEYS:
        e = by_key(k)
        out.append(DetectorSpec(
            identifier=e.key, name=e.name, prompt_fragment=e.prompt_fragment,
            schema_fields=e.schema_json, always_on=False))
    return out


async def run_n(mp4: bytes, settings, specs, label: str) -> int:
    trues = 0
    for i in range(N):
        try:
            res = await gemini_provider(
                mp4=mp4, detectors=specs, log_tag=label, settings=settings,
                patient_in_clip=False)
            df = res.detector_fields
            taken = bool(df.get("medication_taken"))
            trues += taken
            print(f"  {label} run {i + 1:2d}: medication_taken={taken}  "
                  f"conf={df.get('medication_confidence')}  "
                  f"primary={df.get('primary_detection')!r}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {label} run {i + 1:2d}: ERROR {ex}")
    print(f">>> {label}: medication FP in {trues}/{N} runs\n")
    return trues


async def main() -> None:
    s = get_settings().model_copy(update={"motion_overlay_enabled": True})
    print(f"clip={CLIP}  model={s.gemini_model}  media_res={s.gemini_media_resolution!r}"
          f"  fps={s.gemini_video_fps}  overlay=ON  detectors={len(DETECTOR_KEYS)}  N={N}\n")

    v0 = str(_OUT / "v0_native_overlay.mp4")
    crop = str(_OUT / "v1_trackcrop.mp4")
    v1 = str(_OUT / "v1_trackcrop_overlay.mp4")

    print("building V0 = overlay(native)…")
    apply_overlay(CLIP, v0)
    print("building V1 = overlay(tracking-crop)…")
    sw, sh = make_tracking_crop(CLIP, crop)
    apply_overlay(crop, v1)
    print(f"  track-crop size = {sw}x{sh}")
    print(f"  V0 {v0}: {_probe(v0)}")
    print(f"  V1 {v1}: {_probe(v1)}\n")

    specs = _specs()
    await run_n(Path(v0).read_bytes(), s, specs, "V0 native+overlay")
    await run_n(Path(v1).read_bytes(), s, specs, "V1 trackcrop+overlay")


if __name__ == "__main__":
    asyncio.run(main())
