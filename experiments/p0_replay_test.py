"""P0 exit check — replay the labeled rapid-motion eval clips through the
REAL production RAPID path (rapid_call.run_rapid_seizure + gemini_provider),
not the experiment harness. Confirms the port preserved the validated
numbers (experiment: convulsion 8/10 R, 1/30 FP at N=5).

Usage:  PYTHONPATH=. uv run python experiments/p0_replay_test.py
"""

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.rapid_call import run_rapid_seizure
from src.services.reid.body_detector import PersonDetector
from src.settings import get_settings

_D = Path.home() / "Desktop" / "eval-set" / "rapid_motion"


def _clip_frames(path: Path) -> list[FrameSnapshot]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    out, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        if w % 2 or h % 2:
            bgr = bgr[: h - h % 2, : w - w % 2]
            h, w = bgr.shape[:2]
        i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        out.append(FrameSnapshot(
            width=w, height=h, data=i420.tobytes(), captured_at=i / fps,
        ))
        i += 1
    cap.release()
    return out


def _person_box(frames: list[FrameSnapshot]) -> list[tuple]:
    det = PersonDetector()
    boxes = []
    for f in frames[:: max(1, len(frames) // 6)]:
        y = np.frombuffer(f.data, dtype=np.uint8)
        bgr = cv2.cvtColor(
            y.reshape(f.height * 3 // 2, f.width), cv2.COLOR_YUV2BGR_I420)
        for b in det.detect(bgr):
            boxes.append(tuple(float(v) for v in b[:4]))
    return boxes


async def main() -> None:
    s = get_settings().model_copy(update={"rapid_seizure_enabled": True})
    clips = sorted(_D.glob("*.mp4"))
    if not clips:
        sys.exit(f"no clips under {_D}")
    hits = {"conv": [0, 0], "shiv": [0, 0], "neg": [0, 0]}
    for clip in clips:
        truth = clip.stem.split("_")[0][:4]
        truth = truth if truth in hits else "neg"
        frames = _clip_frames(clip)
        boxes = _person_box(frames)
        out = await run_rapid_seizure(
            frames=frames,
            per_person_boxes=[boxes] if boxes else [
                [(0.0, 0.0, float(frames[0].width), float(frames[0].height))]],
            settings=s, provider=gemini_provider, log_tag="replay",
        )
        fired = bool(out and out.get("seizure_detected"))
        gate_open = out is not None
        hits[truth][0] += int(fired)
        hits[truth][1] += 1
        why = (out or {}).get("seizure_reasoning", "gate closed / no call")
        print(f"  {clip.stem:32s} [{truth}] gate={'open' if gate_open else 'CLOSED'} "
              f"fired={fired}  {str(why)[:90]}")
    print(f"\n═══ PROD-PATH REPLAY: conv {hits['conv'][0]}/{hits['conv'][1]} fired | "
          f"shiv FP {hits['shiv'][0]}/{hits['shiv'][1]} | "
          f"neg FP {hits['neg'][0]}/{hits['neg'][1]} ═══")


if __name__ == "__main__":
    asyncio.run(main())
