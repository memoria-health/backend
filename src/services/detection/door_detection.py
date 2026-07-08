"""One-shot door / doorway detection for the Wandering setup flow.

When a caregiver enables Wandering on a camera, the app sends a single frame
here; a VLM finds the doorways and returns their normalized boxes. The app
highlights them so the caregiver can pick (or tap-override) the EXTERIOR exit
door — the one that actually counts as wandering. A separate, cheap, one-shot
call (NOT the per-tick detector path), so it's decoupled from the clip pipeline.
"""
import asyncio
import json

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from src.settings import Settings

# Pin to a stable model with strong spatial grounding. The account's detector
# model (settings.gemini_model) may be a preview tier that 503s under load;
# door detection is a one-shot setup call, so it uses its own known-good model.
_DOOR_MODEL = "gemini-2.5-flash"

_DOOR_PROMPT = (
    "You are looking at ONE still frame from a home monitoring camera mounted "
    "high, looking down at a room. Identify every DOORWAY or DOOR a person could "
    "walk THROUGH — a hinged door, an open doorway/archway, or a sliding door. "
    "Do NOT include windows, closet cabinetry, mirrors, TVs, picture frames, or "
    "furniture.\n\n"
    "Return JSON {\"doors\": [ ... ]}. For each door give:\n"
    "  - \"box\": [x, y, w, h] as fractions of the image width/height (0-1), "
    "where x,y is the TOP-LEFT corner and w,h the size, tight around the "
    "doorway opening;\n"
    "  - \"label\": a short phrase for its location so a person can tell doors "
    "apart (e.g. 'left wall', 'center-back', 'far right').\n"
    "If there are no walk-through doors, return {\"doors\": []}."
)

_SCHEMA = genai_types.Schema(
    type=genai_types.Type.OBJECT,
    required=["doors"],
    properties={
        "doors": genai_types.Schema(
            type=genai_types.Type.ARRAY,
            items=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                required=["box", "label"],
                properties={
                    "box": genai_types.Schema(
                        type=genai_types.Type.ARRAY,
                        items=genai_types.Schema(type=genai_types.Type.NUMBER),
                    ),
                    "label": genai_types.Schema(type=genai_types.Type.STRING),
                },
            ),
        ),
    },
)


async def detect_doors(image_jpeg: bytes, *, settings: Settings) -> list[dict]:
    """Return `[{'box': [x, y, w, h] (0-1), 'label': str}, ...]` for the doorways
    in `image_jpeg`. Uses Gemini (strong spatial grounding). Raises on failure."""
    client = genai.Client(api_key=settings.gemini_api_key)
    parts = [
        genai_types.Part(
            inline_data=genai_types.Blob(data=image_jpeg, mime_type="image/jpeg")),
        genai_types.Part(text=_DOOR_PROMPT),
    ]
    config = genai_types.GenerateContentConfig(
        response_mime_type="application/json", response_schema=_SCHEMA)
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            response = await client.aio.models.generate_content(
                model=_DOOR_MODEL,
                contents=[genai_types.Content(role="user", parts=parts)],
                config=config,
            )
            break
        except genai_errors.ServerError as e:  # 503/overloaded — retry
            last_err = e
            await asyncio.sleep(1.5 * (attempt + 1))
    else:
        raise last_err or RuntimeError("door detection failed")
    data = json.loads((response.text or "{}").strip() or "{}")
    out: list[dict] = []
    for d in data.get("doors", []):
        box = d.get("box") or []
        if len(box) == 4:
            box = [max(0.0, min(1.0, float(v))) for v in box]
            out.append({"box": box, "label": str(d.get("label", "")).strip()})
    return out
