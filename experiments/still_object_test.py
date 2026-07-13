"""STILL object test — phone-on-couch + meal-plate-on-table, one still, N runs.

Two takes staged by Aryaman (both objects in the same frame):
  take 1 (TP): phone on couch + plate with food on table
  take 2 (FP): black controller/remote on couch instead of the phone

Sharpest single frame per clip -> one still call carrying both detectors,
N runs on BOTH model tiers (STILL is meant to run on lite; flash ×N doubles
as the escalation diagnostic if lite fails).

Usage:  PYTHONPATH=. uv run python experiments/still_object_test.py CLIP [CLIP2 ...] [--n N]
"""

import asyncio
import sys
from pathlib import Path

# collar_recall_test parses sys.argv[1] as an int at import time — capture our
# args and hand it a clean argv before importing.
_ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]

from experiments.collar_recall_test import _gemini_stills, _retry, _sharpest_stills
from src.services.detection.prompt import DetectorSpec
from src.settings import get_settings

MODELS = ("gemini-3.1-flash-lite",)   # STILL runs on lite in prod — test lite

PHONE = DetectorSpec(
    identifier="phone_on_couch", name="Phone left on couch", always_on=False,
    prompt_fragment=(
        "phone_on_couch_detected: boolean. TRUE when a SMARTPHONE is visible "
        "resting on the couch/sofa (not held by anyone): a thin flat slab with "
        "a full glass screen face — the screen may be dark, black, or "
        "reflective, with no physical buttons across its face.\n"
        "Distinguish from a TV REMOTE or GAME CONTROLLER — the most easily "
        "confused objects: a remote is elongated with a GRID OF BUTTONS on its "
        "face; a controller has grips/joysticks. NAME which object you see in "
        "your reasoning. If you cannot tell phone from remote/controller, set "
        "FALSE with confidence <= 0.4.\n"
        "Set FALSE for: remote/controller, glasses or their case, wallet, "
        "book, tablet (much larger), no object on the couch, or too unclear.\n"
        "phone_on_couch_confidence: number 0-1.\n"
        "phone_on_couch_reasoning: 1 short sentence NAMING the object seen."),
    schema_fields={"phone_on_couch_detected": {"type": "boolean"},
                   "phone_on_couch_confidence": {"type": "number"},
                   "phone_on_couch_reasoning": {"type": "string"}})

MEAL = DetectorSpec(
    identifier="meal_plate", name="Meal plate on table", always_on=False,
    prompt_fragment=(
        "meal_plate_detected: boolean. TRUE when a PLATE WITH FOOD ON IT is "
        "visible on a table (dining table, coffee table, side table).\n"
        "Set FALSE for: an EMPTY plate (plate alone is not a meal); a cup/mug "
        "or bottle only; non-food items on the table (books, laptop, remote); "
        "no plate visible; too unclear to judge.\n"
        "meal_plate_confidence: number 0-1.\n"
        "meal_plate_reasoning: 1 short sentence naming what is on the table."),
    schema_fields={"meal_plate_detected": {"type": "boolean"},
                   "meal_plate_confidence": {"type": "number"},
                   "meal_plate_reasoning": {"type": "string"}})

SPECS = [PHONE, MEAL]
KEYS = ("phone_on_couch_detected", "meal_plate_detected")


async def _token_probe(still: bytes, settings) -> None:
    """One raw call to read usage_metadata: how many prompt tokens does the
    single still consume (image + text), at MEDIA_RESOLUTION_HIGH?"""
    from google import genai
    from google.genai import types as gt
    client = genai.Client(api_key=settings.gemini_api_key)
    text = "Describe this image in one sentence."
    resp = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[gt.Content(role="user", parts=[
            gt.Part(inline_data=gt.Blob(data=still, mime_type="image/jpeg")),
            gt.Part(text=text)])],
        config=gt.GenerateContentConfig(
            media_resolution=gt.MediaResolution.MEDIA_RESOLUTION_HIGH))
    u = resp.usage_metadata
    txt = await client.aio.models.count_tokens(
        model=settings.gemini_model, contents=text)
    img = (u.prompt_token_count or 0) - (txt.total_tokens or 0)
    print(f"  TOKENS: prompt={u.prompt_token_count} (text≈{txt.total_tokens} "
          f"→ image≈{img}) @media_res=HIGH")


async def main() -> None:
    args = [a for a in _ARGS if not a.startswith("--") and a.endswith(".mp4")]
    n = 10
    if "--n" in _ARGS:
        n = int(_ARGS[_ARGS.index("--n") + 1])
    base = get_settings().model_copy(update={"motion_overlay_enabled": False})
    for clip in args:
        still = _sharpest_stills(clip, k=1)
        if not still:
            print(f"{clip}: no frames"); continue
        Path(clip).with_suffix(".still.jpg").write_bytes(still[0])
        print(f"\n════ {Path(clip).stem} (sharpest still saved alongside clip) ════")
        await _token_probe(still[0], base)
        for model in MODELS:
            s = base.model_copy(update={"gemini_model": model})
            res = await asyncio.gather(*(
                _retry(lambda: _gemini_stills(still, SPECS, s)) for _ in range(n)))
            fields = [r.detector_fields for r in res if r is not None]
            for k in KEYS:
                fires = sum(1 for f in fields if bool(f.get(k)))
                print(f"  {model:<24} {k.replace('_detected',''):<16} {fires}/{len(fields)}")
            f0 = fields[0] if fields else {}
            print(f"    e.g. phone: {str(f0.get('phone_on_couch_reasoning'))[:100]}")
            print(f"    e.g. meal:  {str(f0.get('meal_plate_reasoning'))[:100]}")


if __name__ == "__main__":
    asyncio.run(main())
