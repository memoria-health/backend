"""RAPID packaging test — temporal concat (overlay first, then raw) + solo
prompts, convulsion TPs only.

Changes vs rapid_motion_test.py (user-directed 2026-07-12):
  - Dual = ONE video, TWO segments back to back: motion-OVERLAY copy first,
    then the RAW clip. The prompt names the loop-back timestamp. Full
    resolution per channel (side-by-side halved it).
  - Overlay-only gets a real arm (O8).
  - ONE detector per prompt (solo), matching how prod configures calls; both
    detectors still run per input as separate calls so cross-routing
    (convulsion vs body_shaking) stays visible.
  - Both model tiers per cell: prod flash-lite vs full flash-preview.

Arms:  T4 concat 4fps · T8 concat 8fps · O8 overlay-only 8fps
Clips: conv_1, conv_2 (confirmed convulsion TPs; single person)

Usage:  PYTHONPATH=. uv run python experiments/rapid_concat_test.py [N]
"""

import asyncio
import subprocess
import sys
from pathlib import Path

import cv2

from experiments.custom_detectors import BODY_SHAKING, CONVULSION
from experiments.rapid_motion_test import _WORK, _build, _call
from experiments.collar_recall_test import _retry
from src.services.detection.prompt import DetectorSpec
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Desktop/eval-set/rapid_motion")
CLIPS = ["conv_1", "conv_2"]
MODELS = ["gemini-3.1-flash-lite", "gemini-3-flash-preview"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ARMS = ("T4", "T8", "O8")
DETS = (("conv", CONVULSION, "convulsion"),
        ("shake", BODY_SHAKING, "body_shaking"))


def _motion_desc(fragment: str) -> DetectorSpec:
    return DetectorSpec(
        identifier="motion_description", name="Motion description",
        always_on=True, prompt_fragment=fragment,
        schema_fields={"motion_region":    {"type": "string"},
                       "motion_spread":    {"type": "string"},
                       "motion_rhythm":    {"type": "string"},
                       "motion_amplitude": {"type": "string"},
                       "motion_voluntary": {"type": "string"}})


_FIELD_DEFS = (
    "motion_region: string. The body part(s) where motion concentrates "
    "('none' if nothing moves).\n"
    "motion_spread: string, one of: none | localized (one small area, e.g. "
    "one hand) | regional (one limb or the torso) | whole_body.\n"
    "motion_rhythm: string, one of: none | rhythmic (regular repeating beat) "
    "| continuous (steady, no distinct beats) | sporadic (isolated "
    "movements).\n"
    "motion_amplitude: string, one of: none | flicker (body outline stays "
    "put, surface trembles in place — fine trembling) | displacement (body "
    "parts visibly TRAVEL between frames — gross movement).\n"
)

MD_CONCAT = _motion_desc(
    "BEFORE the detector fields below, DESCRIBE THE MOTION using both "
    "segments of the video.\n"
    "From the FIRST segment (the motion-tinted copy) fill:\n" + _FIELD_DEFS +
    "From the SECOND segment (the raw copy of the same moment) fill:\n"
    "motion_voluntary: string, one of: voluntary (a recognizable purposeful "
    "action — rubbing, wiping, adjusting, exercising) | involuntary "
    "(stereotyped, uncontrolled, serves no visible purpose) | unclear. Judge "
    "voluntariness ONLY from what the person is actually doing in the raw "
    "segment, never from the tint.\n"
    "Fill these five fields FIRST; every detector answer below MUST be "
    "consistent with them.")

MD_OVERLAY = _motion_desc(
    "BEFORE the detector fields below, DESCRIBE THE MOTION you observe.\n" +
    _FIELD_DEFS +
    "motion_voluntary: string, one of: voluntary (a recognizable purposeful "
    "action) | involuntary (stereotyped, uncontrolled, serves no visible "
    "purpose) | unclear. Judge voluntariness from the person's behaviour in "
    "the scene visible beneath the tint.\n"
    "Fill these five fields FIRST; every detector answer below MUST be "
    "consistent with them.")


def _concat_note(seg_s: float) -> str:
    total = 2 * seg_s
    return (
        f"The video is {total:.0f} seconds long and contains the SAME "
        f"{seg_s:.0f}-second moment TWICE, back to back. From 0:00 to "
        f"0:{seg_s:02.0f}: a copy with a semi-transparent RED MOTION TINT "
        "overlaid on the areas that moved since the previous frame. At "
        f"0:{seg_s:02.0f} the video LOOPS BACK to the start of the moment "
        "and plays the RAW clip with no tint. The tint marks pixel CHANGE "
        "only (more red = more motion there); it is NOT blood, a light, or "
        "any real object — never report the red itself as an object or "
        "injury. Use the tinted FIRST segment to LOCATE and CHARACTERIZE "
        "motion (where, spread, rhythm, amplitude); use the raw SECOND "
        "segment to judge WHAT the person is actually doing (posture, "
        "purpose, context). It is ONE moment and ONE person shown twice — "
        "never report the repetition as two events or two people. Red alone "
        "is never sufficient evidence for any detector."
    )


OVERLAY_NOTE = (
    "The video has a semi-transparent RED MOTION TINT overlaid on the areas "
    "that moved since the previous frame. The tint marks pixel CHANGE only "
    "(more red = more motion there); it is NOT blood, a light, or any real "
    "object — never report the red itself as an object or injury. There is "
    "no separate raw copy: judge the person's behaviour from the scene "
    "visible beneath the tint. A body part that stays red across many frames "
    "while staying in place means fast repetitive localized motion; red that "
    "travels across the frame is ordinary movement. Red alone is never "
    "sufficient evidence for any detector."
)


def _dur_s(path: str) -> float:
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 8
    cap.release()
    return n / fps


def _concat(first: str, second: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", first, "-i", second, "-filter_complex",
         "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264", "-pix_fmt",
         "yuv420p", "-an", out],
        check=True, capture_output=True)


def _inputs(stem: str):
    """Reuse the cc/co artifacts from the 2x2 run; rebuild if missing."""
    need = [f"{stem}_R{t}_{k}0.mp4" for t in ("4", "8") for k in ("cc", "co")]
    if not all((_WORK / f).exists() for f in need):
        _build(_D / f"{stem}.mp4")
    out = {}
    for tag in ("4", "8"):
        cc, co = str(_WORK / f"{stem}_R{tag}_cc0.mp4"), str(_WORK / f"{stem}_R{tag}_co0.mp4")
        cat = str(_WORK / f"{stem}_R{tag}_cat0.mp4")
        _concat(co, cc, cat)                      # overlay FIRST, then raw
        out[f"T{tag}"] = (Path(cat).read_bytes(), _concat_note(_dur_s(cc)))
        if tag == "8":
            out["O8"] = (Path(co).read_bytes(), OVERLAY_NOTE)
    return out


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    base = get_settings().model_copy(update={"motion_overlay_enabled": False})
    print(f"RAPID concat/solo  N={N}  clips={CLIPS}  models={MODELS}\n")

    reasons = []
    for stem in CLIPS:
        inputs = _inputs(stem)
        for model in MODELS:
            s = base.model_copy(update={"gemini_model": model})
            for arm in ARMS:
                vid, note = inputs[arm]
                md = MD_OVERLAY if arm == "O8" else MD_CONCAT
                cells = {}
                for tag, det, ident in DETS:
                    res = await asyncio.gather(*(
                        _retry(lambda: _call(vid, [md, det], note, s))
                        for _ in range(N)))
                    fields = [r.detector_fields for r in res if r is not None]
                    fires = sum(1 for f in fields
                                if bool(f.get(f"{ident}_detected")))
                    cells[tag] = f"{fires}/{len(fields)}"
                    for f in fields[:1]:
                        reasons.append(
                            (stem, model.split('-', 1)[1], arm, tag,
                             f.get('motion_amplitude'), f.get('motion_voluntary'),
                             str(f.get(f'{ident}_reasoning', ''))[:130]))
                print(f"  {stem} {model:<24} {arm}:  "
                      f"conv={cells['conv']}  shake={cells['shake']}")
        print()

    print("── sample outputs (1st sample per cell: amplitude/voluntary + reasoning) ──")
    for stem, model, arm, tag, amp, vol, why in reasons:
        print(f"  {stem} {model:<16} {arm} {tag}: [{amp}/{vol}] {why}")


if __name__ == "__main__":
    asyncio.run(main())
