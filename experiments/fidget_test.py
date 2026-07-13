"""MICRO-BODY test — general_fidgeting, gate #2 before implementation.

One detector (Aryaman 2026-07-13): hand-wringing / clothes-picking belong to the
already-validated micro-SLOW class, so this session tests the FASTER restless
end under a single general_fidgeting detector — repetitive purposeless motion
(finger tapping, leg/heel bouncing) — and places it on the slow/rapid boundary:

Arms:
  P4  person-crop clean 4fps  lite    (micro-slow recipe)
  P8  person-crop clean 8fps  flash   (rapid recipe, no overlay)

Clips in ~/Desktop/eval-set/fidget/, 8fps captures (4fps derived). Truth:
  fid_*   fidgeting TP (tapping, leg bouncing)
  neg_*   confounders (typing, scrolling, eating, gesturing, one adjustment)

Usage:  PYTHONPATH=. uv run python experiments/fidget_test.py [N]
"""

import asyncio
import sys
from pathlib import Path

from experiments.collar_recall_test import _apply_crop, _retry
from experiments.rapid_motion_test import _WORK, _call, _decimate, _track_persons_fps
from experiments.custom_detectors import MOTION_DESCRIPTION_CLEAN
from src.services.detection.prompt import DetectorSpec
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Desktop/eval-set/fidget")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARMS = {"P4": ("4", "gemini-3.1-flash-lite"),
        "P8": ("8", "gemini-3-flash-preview")}
KEY = "general_fidgeting_detected"

GENERAL_FIDGETING = DetectorSpec(
    identifier="general_fidgeting", name="General fidgeting", always_on=False,
    prompt_fragment=(
        "general_fidgeting_detected: boolean. TRUE when the person shows "
        "sustained, purposeless, repetitive RESTLESS movement of the hands or "
        "legs: rapid finger/hand tapping on a surface or their own body, "
        "leg/heel bouncing or knee jigging while seated, restless repetitive "
        "foot movement — continuing across the clip and accomplishing "
        "nothing.\n"
        "The REQUIRED distinction is PURPOSE — name it in your reasoning:\n"
        "  - TYPING or PHONE USE (FALSE): fingers act ON a device, gaze "
        "directed at it;\n"
        "  - EATING or DRINKING (FALSE): hand-to-mouth transport;\n"
        "  - GESTURING while talking (FALSE): broad expressive hand movement "
        "synchronized with speech and engagement;\n"
        "  - a single adjustment of clothing/position, then stillness "
        "(FALSE): brief and done, not sustained;\n"
        "  - walking or repositioning (FALSE).\n"
        "If purpose is uncertain, set FALSE with confidence <= 0.4.\n"
        "general_fidgeting_confidence: number 0-1. 0.7+ = clearly sustained "
        "purposeless repetitive hand/leg movement.\n"
        "general_fidgeting_reasoning: 1 short sentence NAMING what the "
        "movement was and the purpose you saw or ruled out."),
    schema_fields={"general_fidgeting_detected": {"type": "boolean"},
                   "general_fidgeting_confidence": {"type": "number"},
                   "general_fidgeting_reasoning": {"type": "string"}})

SPECS = [MOTION_DESCRIPTION_CLEAN, GENERAL_FIDGETING]


def _build(clip: Path):
    stem = clip.stem
    clip4 = str(_WORK / f"fid_{stem}_4fps.mp4")
    _decimate(str(clip), clip4)
    out = {}
    for tag, src, fps in (("8", str(clip), 8), ("4", clip4, 4)):
        geoms = _track_persons_fps(src, fps)
        vids = []
        for pi, g in enumerate(geoms):
            cc = str(_WORK / f"fid_{stem}_R{tag}_cc{pi}.mp4")
            _apply_crop(src, cc, g)
            vids.append(Path(cc).read_bytes())
        out[tag] = vids
    return out


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    base = get_settings().model_copy(update={"motion_overlay_enabled": False})
    clips = sorted(_D.glob("*.mp4"))
    if not clips:
        sys.exit(f"no clips in {_D} (expected fid_/neg_ .mp4 at 8fps)")
    print(f"GENERAL FIDGETING  N={N}  clips={[c.stem for c in clips]}\n")

    fires = {a: {"fid": 0, "neg": 0} for a in ARMS}
    denom = {"fid": 0, "neg": 0}
    for clip in clips:
        truth = "fid" if clip.stem.startswith("fid") else "neg"
        denom[truth] += N
        built = _build(clip)
        for arm, (tag, model) in ARMS.items():
            s = base.model_copy(update={"gemini_model": model})

            async def any_person(vids, settings):
                res = await asyncio.gather(*(
                    _retry(lambda v=v: _call(v, SPECS, "", settings))
                    for v in vids))
                fs = [r.detector_fields for r in res if r is not None]
                if not fs:
                    return None, {}
                return any(bool(f.get(KEY)) for f in fs), fs[0]

            samples = await asyncio.gather(*(any_person(built[tag], s)
                                             for _ in range(N)))
            n_fire = sum(1 for v, _ in samples if v)
            fires[arm][truth] += n_fire
            why = str(samples[0][1].get("general_fidgeting_reasoning", ""))[:100] \
                if samples and samples[0][1] else ""
            print(f"  {clip.stem} [{truth}] {arm}: {n_fire}/{N}  e.g. {why}")
        print()

    print("═══ DISCRIMINATION ═══")
    for arm in ARMS:
        rec = fires[arm]["fid"] / denom["fid"] if denom["fid"] else 0.0
        fp = fires[arm]["neg"] / denom["neg"] if denom["neg"] else 0.0
        print(f"  {arm}: recall={rec:.0%}  fp={fp:.0%}  "
              f"discrimination={rec - fp:+.0%}")


if __name__ == "__main__":
    asyncio.run(main())
