"""RAPID decision run — O8 (overlay-only crop, 8fps) on gemini-3-flash-preview,
BOTH detectors in every call, all 10 events (1 clip per event).

Follows the packaging findings (2026-07-12): full-flash + overlay-only won
recall 10/10 on the convulsion TPs (lite is blind at any packaging; adding a
raw segment re-injects the resting prior). This run adds the precision side:
shivering TPs + 6 confounder takes, with convulsion and body_shaking as
mutual confounders in ONE prompt (matching bucket design, ≤3 detectors/call).

Usage:  PYTHONPATH=. uv run python experiments/rapid_o8_flash_test.py [N]
"""

import asyncio
import sys
from pathlib import Path

from experiments.collar_recall_test import _retry
from experiments.custom_detectors import BODY_SHAKING, CONVULSION
from experiments.rapid_concat_test import MD_OVERLAY, OVERLAY_NOTE
from experiments.rapid_motion_test import _WORK, _build, _call, _truth
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Desktop/eval-set/rapid_motion")
MODEL = "gemini-3-flash-preview"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
KEYS = ("convulsion_detected", "body_shaking_detected")
SPECS = [MD_OVERLAY, CONVULSION, BODY_SHAKING]


def _input(stem: str) -> bytes:
    co = _WORK / f"{stem}_R8_co0.mp4"
    if not co.exists():
        _build(_D / f"{stem}.mp4")
    return co.read_bytes()


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    s = get_settings().model_copy(update={"gemini_model": MODEL,
                                          "motion_overlay_enabled": False})
    clips = sorted(_D.glob("*.mp4"))
    print(f"RAPID O8 decision run  model={MODEL}  N={N}  both detectors/call  "
          f"clips={[c.stem for c in clips]}\n")

    fires = {t: {k: 0 for k in KEYS} for t in ("conv", "shiv", "neg")}
    denom = {t: 0 for t in ("conv", "shiv", "neg")}
    wrong = []

    for clip in clips:
        stem, truth = clip.stem, _truth(clip.stem)
        denom[truth] += N
        vid = _input(stem)
        res = await asyncio.gather(*(
            _retry(lambda: _call(vid, SPECS, OVERLAY_NOTE, s))
            for _ in range(N)))
        fields = [r.detector_fields for r in res if r is not None]
        rate = {k: sum(1 for f in fields if bool(f.get(k))) for k in KEYS}
        for k in KEYS:
            fires[truth][k] += rate[k]
        expected = {"conv": "convulsion_detected",
                    "shiv": "body_shaking_detected", "neg": None}[truth]
        for f in fields:
            for k in KEYS:
                if bool(f.get(k)) != (k == expected):
                    wrong.append((stem, k, bool(f.get(k)),
                                  f.get("motion_amplitude"),
                                  f.get("motion_voluntary"),
                                  str(f.get(k.replace("_detected", "_reasoning"), ""))[:140]))
                    break
        print(f"  {stem} [{truth}]:  conv={rate['convulsion_detected']}/{len(fields)}  "
              f"shake={rate['body_shaking_detected']}/{len(fields)}")

    print("\n═══ SUMMARY (fires / samples per clip-truth) ═══")
    for t in ("conv", "shiv", "neg"):
        print(f"  {t} clips:  conv={fires[t]['convulsion_detected']}/{denom[t]}  "
              f"shake={fires[t]['body_shaking_detected']}/{denom[t]}")

    print("\n═══ DISCRIMINATION ═══")
    for key, own in (("convulsion_detected", "conv"),
                     ("body_shaking_detected", "shiv")):
        others = [t for t in ("conv", "shiv", "neg") if t != own]
        rec = fires[own][key] / denom[own] if denom[own] else 0.0
        fp_n = sum(fires[t][key] for t in others)
        fp_d = sum(denom[t] for t in others)
        fp = fp_n / fp_d if fp_d else 0.0
        print(f"  {key.replace('_detected',''):<13} recall={rec:.0%}  "
              f"fp={fp:.0%}  discrimination={rec - fp:+.0%}")

    if wrong:
        print("\n── wrong verdicts (miss or false fire; amplitude/voluntary + reasoning) ──")
        for stem, k, fired, amp, vol, why in wrong[:20]:
            tag = "FALSE-FIRE" if fired else "MISS"
            print(f"  {stem} {k.replace('_detected','')} {tag}: [{amp}/{vol}] {why}")


if __name__ == "__main__":
    asyncio.run(main())
