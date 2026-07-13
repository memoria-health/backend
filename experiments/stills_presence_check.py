"""Can a STILL identify a medication CONTAINER (presence), even though it can't
identify medication-TAKING (action)?

Tests the presence-vs-action axis directly: a custom "med_container_present"
detector (pill bottle / blister pack / organizer visible ANYWHERE in frame),
run on STILLS vs on NATIVE VIDEO for the 6 medication clips.

  - stills ≈ video on PRESENCE  -> confirms "stills fine for a stationary object"
  - video >> stills on PRESENCE  -> stills genuinely weaker even for objects
  - both ~0                       -> the container isn't visible (absence / too small),
                                    NOT a stills failure — read the reasoning strings

Usage:  PYTHONPATH=. uv run python experiments/stills_presence_check.py [N]
"""

import asyncio
import sys
from pathlib import Path

import cv2
from google import genai
from google.genai import types as genai_types

from src.services.detection.prompt import DetectorSpec, build_prompt_and_schema
from src.services.detection.providers._parse import parse_and_validate
from src.services.detection.providers.gemini import gemini_provider
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Downloads/memoria-crop-experiment/med_tp_raw")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CLIPS = ["233526_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4",
         "233536_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4",
         "233546_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4",
         "233606_9271358c-647f-44d8-a3ab-1b88a2044566.mp4",
         "233616_9271358c-647f-44d8-a3ab-1b88a2044566.mp4",
         "233636_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4"]

CONTAINER = DetectorSpec(
    identifier="med_container", name="Medication container present", always_on=False,
    prompt_fragment=(
        "med_container_present: boolean. TRUE if a MEDICATION CONTAINER is visible "
        "anywhere in any frame — a pill bottle, prescription vial, blister/foil pack, "
        "or weekly pill organizer. This is about the OBJECT being present, not whether "
        "anyone takes anything. FALSE if no such container is visible.\n"
        "med_container_confidence: number 0-1.\n"
        "med_container_reasoning: 1 short sentence naming what you see (e.g. 'an amber "
        "pill bottle on the kitchen counter', 'no medication container visible')."),
    schema_fields={"med_container_present": {"type": "boolean"},
                   "med_container_confidence": {"type": "number"},
                   "med_container_reasoning": {"type": "string"}})


def _sharpest_stills(clip: str, k: int = 3) -> list[bytes]:
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
    cap.release()
    picks, seg = [], max(1, len(frames) // k)
    for s in range(0, len(frames), seg):
        window = frames[s:s + seg]
        best = max(window, key=lambda f: cv2.Laplacian(
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        ok, buf = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            picks.append(buf.tobytes())
        if len(picks) >= k:
            break
    return picks


def _mt(t):
    return {"boolean": genai_types.Type.BOOLEAN, "number": genai_types.Type.NUMBER,
            "string": genai_types.Type.STRING}[t]


async def _stills_call(images, settings):
    built = build_prompt_and_schema([CONTAINER], motion_overlay=False)
    schema = genai_types.Schema(
        type=genai_types.Type.OBJECT, required=built.schema_required,
        properties={n: genai_types.Schema(type=_mt(f["type"]))
                    for n, f in built.schema_properties.items()})
    parts = [genai_types.Part(inline_data=genai_types.Blob(data=i, mime_type="image/jpeg"))
             for i in images] + [genai_types.Part(text=built.prompt)]
    client = genai.Client(api_key=settings.gemini_api_key)
    resp = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=schema,
            media_resolution=genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH))
    return parse_and_validate((resp.text or "").strip(), provider="gemini",
                              required=built.schema_required,
                              properties=built.schema_properties)


async def main():
    s = get_settings().model_copy(update={"motion_overlay_enabled": False})
    print(f"model={s.gemini_model}  N={N}  detector=med_container_present (PRESENCE)\n")
    tot_s = tot_v = 0
    for fn in CLIPS:
        clip = str(_D / fn)
        stills = _sharpest_stills(clip)
        mp4 = Path(clip).read_bytes()
        s_hits = v_hits = 0
        s_reason = v_reason = ""
        for _ in range(N):
            try:
                r = await _stills_call(stills, s)
                if r.detector_fields.get("med_container_present"):
                    s_hits += 1
                s_reason = r.detector_fields.get("med_container_reasoning", "")
            except Exception as ex:  # noqa: BLE001
                s_reason = f"ERR {type(ex).__name__}"
            try:
                r = await gemini_provider(mp4=mp4, detectors=[CONTAINER], log_tag="v",
                                          settings=s, patient_in_clip=False)
                if r.detector_fields.get("med_container_present"):
                    v_hits += 1
                v_reason = r.detector_fields.get("med_container_reasoning", "")
            except Exception as ex:  # noqa: BLE001
                v_reason = f"ERR {type(ex).__name__}"
        tot_s += s_hits
        tot_v += v_hits
        print(f"{fn[:6]}: stills={s_hits}/{N} video={v_hits}/{N}")
        print(f"        stills: {s_reason}")
        print(f"        video : {v_reason}")
    print(f"\nTOTAL  stills={tot_s}/{len(CLIPS) * N}  video={tot_v}/{len(CLIPS) * N}")


if __name__ == "__main__":
    asyncio.run(main())
