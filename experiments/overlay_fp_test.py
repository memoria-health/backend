"""B: does the motion overlay DRIVE false positives?  overlay ON vs OFF.

Benign clip (man on a couch with a laptop/phone — no real event), so ANY
detector firing is a false positive. Same prod config otherwise
(gemini-3.1-flash-lite, media_res=high, fps=4, all 7 detectors), N runs each.

  overlay ON  = overlay(native) + prompt explains the red tint
  overlay OFF = native clean     + no tint note

Usage:  PYTHONPATH=. uv run python experiments/overlay_fp_test.py <clip> [N]
"""

import asyncio
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.services.detection.catalog import by_key
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.prompt import DetectorSpec
from src.services.detection.providers.gemini import gemini_provider
from src.settings import get_settings

CLIP = str(Path(sys.argv[1]).resolve())
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
_OUT = Path(CLIP).parent
DETECTOR_KEYS = ["calling_for_help", "emergency", "fall", "agitation",
                 "medication", "choking", "seizure"]


def _even(v: int) -> int:
    return v - (v % 2)


def apply_overlay(clip: str, out: str) -> None:
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
    tmp = out + ".v.mp4"
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (over[0].width, over[0].height))
    for f in over:
        arr = np.frombuffer(f.data, np.uint8).reshape(f.height * 3 // 2, f.width)
        vw.write(cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_I420))
    vw.release()
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp, "-i", clip, "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out],
        check=True, capture_output=True)
    Path(tmp).unlink(missing_ok=True)


def _specs() -> list[DetectorSpec]:
    return [DetectorSpec(identifier=e.key, name=e.name,
                         prompt_fragment=e.prompt_fragment,
                         schema_fields=e.schema_json, always_on=False)
            for e in (by_key(k) for k in DETECTOR_KEYS)]


def _fired(df: dict) -> list[str]:
    return [k for k in DETECTOR_KEYS if df.get(by_key(k).detected_key) is True]


async def run_n(mp4: bytes, settings, specs, label: str) -> None:
    any_fp = 0
    per: Counter = Counter()
    for i in range(N):
        try:
            res = await gemini_provider(mp4=mp4, detectors=specs, log_tag=label,
                                        settings=settings, patient_in_clip=False)
            f = _fired(res.detector_fields)
            any_fp += bool(f)
            per.update(f)
            print(f"  {label} run {i + 1:2d}: fired={f or '—'}")
        except Exception as ex:  # noqa: BLE001
            print(f"  {label} run {i + 1:2d}: ERROR {ex}")
    print(f">>> {label}: {any_fp}/{N} runs had >=1 false positive; "
          f"per-detector={dict(per)}\n")


async def main() -> None:
    on = get_settings().model_copy(update={"motion_overlay_enabled": True})
    off = get_settings().model_copy(update={"motion_overlay_enabled": False})
    specs = _specs()
    print(f"clip={CLIP}  model={on.gemini_model}  detectors={len(DETECTOR_KEYS)}  N={N}\n")
    v_on = str(_OUT / "ovl_on.mp4")
    apply_overlay(CLIP, v_on)
    await run_n(Path(v_on).read_bytes(), on, specs, "overlay ON ")
    await run_n(Path(CLIP).read_bytes(), off, specs, "overlay OFF")


if __name__ == "__main__":
    asyncio.run(main())
