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

from src.services.genai_client import get_client
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


# ONE call — fast. Door highlights are just an ASSIST (the caregiver taps to
# place/confirm anyway), so speed beats the marginal recall of a multi-call
# union. The app doesn't block the camera view on this; it detects in the
# background and overlays dots when they arrive. The clustering below still
# dedups any near-duplicate boxes within the single response.
_N_CALLS = 1
_SAME_IOU = 0.3       # boxes overlapping this much are the same door
_SAME_CENTER = 0.08   # ...or whose centers are this close (handles box jitter)


def _iou(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes (normalized)."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _same_door(a: list[float], b: list[float]) -> bool:
    """True if two boxes plausibly mark the SAME door across calls."""
    if _iou(a, b) > _SAME_IOU:
        return True
    acx, acy = a[0] + a[2] / 2, a[1] + a[3] / 2
    bcx, bcy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 < _SAME_CENTER


def _media_resolution(value: str) -> "genai_types.MediaResolution | None":
    """Map the 'low'|'medium'|'high' setting to the Gemini enum (None = default)."""
    v = (value or "").strip().lower()
    if v == "low":
        return genai_types.MediaResolution.MEDIA_RESOLUTION_LOW
    if v == "medium":
        return genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    if v == "high":
        return genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH
    return None


async def _detect_once(
    client: "genai.Client",
    image_jpeg: bytes,
    media_resolution: "genai_types.MediaResolution | None" = None,
) -> list[dict]:
    """One VLM call → parsed door list. Retries only on transient ServerError."""
    parts = [
        genai_types.Part(
            inline_data=genai_types.Blob(data=image_jpeg, mime_type="image/jpeg")),
        genai_types.Part(text=_DOOR_PROMPT),
    ]
    config_kwargs: dict = {
        "response_mime_type": "application/json", "response_schema": _SCHEMA,
    }
    if media_resolution is not None:
        # More token budget per frame → finer detail on small / distant doors.
        config_kwargs["media_resolution"] = media_resolution
    config = genai_types.GenerateContentConfig(**config_kwargs)
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


async def detect_doors(image_jpeg: bytes, *, settings: Settings) -> list[dict]:
    """Return `[{'box': [x, y, w, h] (0-1), 'label': str}, ...]` for the doorways
    in `image_jpeg`. Runs `_N_CALLS` concurrent Gemini calls and unions the
    results (IoU-dedup) so a single unlucky empty response doesn't blank the
    picker. Raises only if EVERY call fails."""
    client = get_client(settings.gemini_api_key)
    media_res = _media_resolution(settings.gemini_media_resolution)
    results = await asyncio.gather(
        *(_detect_once(client, image_jpeg, media_res) for _ in range(_N_CALLS)),
        return_exceptions=True,
    )
    per_call = [r for r in results if isinstance(r, list)]
    if not per_call:
        raise next(r for r in results if isinstance(r, Exception))

    # Cluster every detection across calls; a real door recurs, a hallucination
    # shows up once. Greedy single-link clustering by _same_door.
    clusters: list[dict] = []  # {'items': [door,...], 'calls': set[int]}
    for ci, r in enumerate(per_call):
        for d in r:
            for c in clusters:
                if any(_same_door(d["box"], it["box"]) for it in c["items"]):
                    c["items"].append(d)
                    c["calls"].add(ci)
                    break
            else:
                clusters.append({"items": [d], "calls": {ci}})

    # Prefer doors corroborated by a MAJORITY of calls (>=2 of N). A real door
    # recurs across calls; a one-off hallucination shows up once and is dropped —
    # this is what stops over-counting (e.g. 3 boxes for 2 real doors). If the
    # frame is hard and NOTHING reaches majority, fall back to the union so the
    # picker never blanks. Cap so a noisy frame can't flood the UI.
    need = 2 if len(per_call) >= 2 else 1
    voted = [c for c in clusters if len(c["calls"]) >= need]
    chosen = voted if voted else clusters
    chosen.sort(
        key=lambda c: (-len(c["calls"]), round(c["items"][0]["box"][0], 2)))
    merged: list[dict] = []
    for c in chosen[:6]:
        boxes = [it["box"] for it in c["items"]]
        rep = [sum(b[i] for b in boxes) / len(boxes) for i in range(4)]  # mean box
        merged.append({"box": rep, "label": c["items"][0]["label"]})
    merged.sort(key=lambda d: (round(d["box"][0], 2), round(d["box"][1], 2)))
    return merged
