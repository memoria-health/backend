"""First-principles token measurement for the cost model. Measures the per-call
token components (video frame, still image, audio, prompt) and how they scale
with fps and media_resolution — so every bucket's cost is grounded, not guessed.
Prints the per-modality breakdown (usage_metadata.prompt_tokens_details)."""

import asyncio
from pathlib import Path

import cv2
from google import genai
from google.genai import types as t

from src.services.detection.catalog import by_key
from src.services.detection.prompt import DetectorSpec, build_prompt_and_schema
from src.settings import get_settings

CLIP = ("/Users/aryamandarda/Downloads/memoria-crop-experiment/med_tp_raw/"
        "233536_0c4eb1f8-2c90-4117-a700-97ba889dbecb.mp4")
KEYS7 = ["calling_for_help", "emergency", "fall", "agitation", "medication",
         "choking", "seizure"]
_RES = {"low": t.MediaResolution.MEDIA_RESOLUTION_LOW,
        "medium": t.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
        "high": t.MediaResolution.MEDIA_RESOLUTION_HIGH}


def _stills(clip, k):
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, b = cap.read()
        if not ok:
            break
        frames.append(b)
    cap.release()
    out, seg = [], max(1, len(frames) // k)
    for s in range(0, len(frames), seg):
        best = max(frames[s:s + seg], key=lambda f: cv2.Laplacian(
            cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        ok, buf = cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            out.append(buf.tobytes())
        if len(out) >= k:
            break
    return out


async def measure(client, model, label, prompt, *, mode, res, fps=4, nst=3):
    if mode == "video":
        media = [t.Part(inline_data=t.Blob(data=Path(CLIP).read_bytes(),
                        mime_type="video/mp4"), video_metadata=t.VideoMetadata(fps=fps))]
    else:
        media = [t.Part(inline_data=t.Blob(data=i, mime_type="image/jpeg"))
                 for i in _stills(CLIP, nst)]
    resp = await client.aio.models.generate_content(
        model=model, contents=[t.Content(role="user", parts=media + [t.Part(text=prompt)])],
        config=t.GenerateContentConfig(media_resolution=_RES[res],
                                       response_mime_type="application/json"))
    u = resp.usage_metadata
    mods = {str(getattr(d.modality, "name", d.modality)): d.token_count
            for d in (u.prompt_tokens_details or [])}
    print(f"{label:<34} total_in={u.prompt_token_count:>6}  {mods}")


async def main():
    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    m = s.gemini_model
    specs7 = [DetectorSpec(identifier=e.key, name=e.name, prompt_fragment=e.prompt_fragment,
                           schema_fields=e.schema_json, always_on=False)
              for e in (by_key(k) for k in KEYS7)]
    p7 = build_prompt_and_schema(specs7, motion_overlay=True).prompt
    med = by_key("medication")
    p1 = build_prompt_and_schema(
        [DetectorSpec(identifier=med.key, name=med.name, prompt_fragment=med.prompt_fragment,
                      schema_fields=med.schema_json)], motion_overlay=False).prompt
    print(f"model={m}\n")
    await measure(client, m, "video 4fps high  7det+ovl", p7, mode="video", res="high", fps=4)
    await measure(client, m, "video 2fps high  7det+ovl", p7, mode="video", res="high", fps=2)
    await measure(client, m, "video 1fps high  7det+ovl", p7, mode="video", res="high", fps=1)
    await measure(client, m, "video 4fps medium 7det+ovl", p7, mode="video", res="medium", fps=4)
    await measure(client, m, "video 4fps low   7det+ovl", p7, mode="video", res="low", fps=4)
    await measure(client, m, "stills x3 high   1det(med)", p1, mode="stills", res="high", nst=3)
    await measure(client, m, "video 4fps high  1det(med)", p1, mode="video", res="high", fps=4)


if __name__ == "__main__":
    asyncio.run(main())
