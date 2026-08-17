"""Per-clip sensitivity screening (2026-07-27).

EVERY event clip from EVERY camera is screened here at clip-creation
time, before upload — company baseline, invisible, no per-account
setting (docs/plans/2026-07-27-room-privacy-and-clip-censoring.md).
The verdict decides whether `clip_archival` uploads the clip as-is or
routes it through `censor.censor_clip` (person-mask blur + audio
stripped) first.

One structured-output Gemini call on the assembled MP4, PINNED to
``settings.sensitivity_model`` — deliberately NOT the per-account
``vlm_provider`` routing (same reasoning as `room_classifier`): a
safety-critical classifier must behave identically for every account,
and the eval corpus is scored against exactly one model. The clip is
sent as native video (inline blob), the same mechanism the detection
providers use for the buffered-window MP4s.

Fail-closed contract (enforced by the CALLER, `clip_archival`):
any error/timeout raised out of `screen_clip` — and any ambiguous
answer resolved here — means the clip is censored, never uploaded raw.
Within this module, ambiguity resolves TOWARD sensitive: a "not
sensitive" answer that still names categories is treated as sensitive.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google import genai

from src.services.genai_client import get_client
from google.genai import types as genai_types

if TYPE_CHECKING:
    from src.settings import Settings

log = logging.getLogger(__name__)

# The dignity-category enum. Stored verbatim in
# `events.sensitivity_categories` when a clip is censored.
SENSITIVITY_CATEGORIES = (
    "nudity_or_undress",
    "toileting",
    "bathing_showering",
    "personal_hygiene",
    "medical_care",
    "incontinence",
    "intimate_situation",
    "emotional_distress",
    "other_dignity",
)

# Categories that actually TRIGGER censoring (2026-07-28 field
# decision): blur exists for BODILY PRIVACY — exposure-class moments.
# `emotional_distress` is record-only: a fall with a cry for help is
# exactly what a caregiver must see clearly, and the first field test
# showed the screen blurring a staged fall because Gemini HEARD the
# shout. Distress is still reported in `categories` (logged, eval
# data) but never censors by itself. `other_dignity` stays a trigger
# but is prompt-narrowed to exposure-class catch-all only.
CENSOR_CATEGORIES = frozenset(
    c for c in SENSITIVITY_CATEGORIES if c != "emotional_distress"
)

# The ONE carve-out to "censored clips carry audio" (Aryaman
# 2026-07-29). Everywhere else the sound stays: the visual is censored
# but a cry for help must remain audible, and that is the whole reason
# audio was kept. Intimacy is the case where the audio IS the exposure —
# blurring the picture and shipping the sound would defeat the censor
# entirely. These clips ship silent.
SILENT_CATEGORIES = frozenset({"intimate_situation"})

# Hard ceiling on the VLM call. A hung call must not hold the archival
# task (and the raw clip bytes) open indefinitely — on expiry the
# caller censors full-frame, same as any other screen failure.
SCREEN_TIMEOUT_SECONDS = 60.0

_PROMPT = (
    "You are a privacy and dignity screener for an in-home care camera "
    "system for elderly and dementia patients. You are watching a short "
    "clip recorded inside a private home — possibly a bathroom or "
    "bedroom. Your ONLY job is to decide whether this clip contains a "
    "SENSITIVE moment that must be censored before any human caregiver "
    "can see it.\n\n"
    "A moment is SENSITIVE if it shows, partially shows, or plausibly "
    "shows any of:\n"
    "  - nudity_or_undress: any nudity or any state of undress — "
    "including PARTIAL undress (open gown, lifted shirt, lowered "
    "trousers, exposed underwear, bare torso, dressing/undressing in "
    "progress).\n"
    "  - toileting: using or preparing to use a toilet/commode/urinal/"
    "bedpan, or adjusting clothing near one.\n"
    "  - bathing_showering: bathing, showering, being washed, entering/"
    "exiting a bath or shower, wet and towelled states.\n"
    "  - personal_hygiene: intimate grooming or hygiene care "
    "(peri-care, changing pads or briefs, skin care on private areas).\n"
    "  - medical_care: exposed wounds, catheters, ostomy bags, injections, "
    "medical devices on the body, or hands-on clinical care.\n"
    "  - incontinence: an incontinence episode, soiled clothing or "
    "bedding, or cleanup of one.\n"
    "  - intimate_situation: any intimate or romantic physical situation.\n"
    "  - other_dignity: ANY other state of BODILY exposure or physical "
    "indignity not listed above. Exposure-class only — a fall, an "
    "accident, or distress WITHOUT bodily exposure does NOT qualify.\n\n"
    "RECORD-ONLY category (report it in categories, but it does NOT by "
    "itself make the clip sensitive):\n"
    "  - emotional_distress: severe visible distress — uncontrolled "
    "crying, panic, humiliation.\n\n"
    "AUDIO IS NEVER SENSITIVITY: ignore sounds entirely. Shouting, "
    "crying out, or calling for help is NOT a reason to censor — judge "
    "only what is VISIBLE.\n\n"
    "BE EXTRA CAUTIOUS — err on the side of SENSITIVE:\n"
    "  - Partial glimpses count. A few frames count. Background counts.\n"
    "  - Ambiguous poses count: if a posture COULD be undress, "
    "toileting, or exposure, treat it as sensitive.\n"
    "  - Occlusion counts: if furniture, a door frame, blur, or the "
    "camera angle hides what is happening and what is hidden could be "
    "sensitive, treat it as sensitive.\n"
    "  - Low light / poor image quality counts: if you cannot clearly "
    "rule OUT a sensitive state, treat it as sensitive.\n"
    "  - Reflections count: check mirrors and glass — a reflection of a "
    "sensitive moment is a sensitive moment.\n"
    "  - When in doubt, ALWAYS answer sensitive=true. A false positive "
    "only blurs a clip; a false negative exposes a person's most "
    "private moment.\n\n"
    "Answer sensitive=false when you are confident the entire clip "
    "shows fully-clothed, non-intimate activity — INCLUDING falls, "
    "accidents, or distress with no bodily exposure (still list "
    "emotional_distress in categories when present).\n\n"
    "Return:\n"
    "  - sensitive: boolean.\n"
    "  - categories: every category that applies (empty when "
    "sensitive=false).\n"
    "  - confidence: your self-assessed probability (0.0-1.0) that your "
    "sensitive verdict is correct."
)


@dataclass(frozen=True)
class ScreenResult:
    """One screening verdict. ``categories`` only carries members of
    `SENSITIVITY_CATEGORIES`; ``confidence`` is the model's self-report
    clamped to [0,1]."""

    sensitive: bool
    categories: list[str]
    confidence: float


def parse_verdict(raw: str) -> ScreenResult:
    """Parse the model's JSON answer, pure for unit tests.

    Ambiguity resolves SENSITIVE: a sensitive=false answer that still
    names categories is contradictory — believe the categories. Unknown
    category strings are dropped (never stored). Malformed JSON /
    missing fields raise ValueError so the caller's fail-closed path
    (full-frame censor) takes over.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"sensitivity: unparseable model answer: {exc}") from exc
    if not isinstance(parsed, dict) or "sensitive" not in parsed:
        raise ValueError("sensitivity: model answer missing 'sensitive'")
    sensitive = bool(parsed["sensitive"])
    raw_categories = parsed.get("categories") or []
    if not isinstance(raw_categories, list):
        raise ValueError("sensitivity: 'categories' is not a list")
    categories = [
        c for c in (str(item).strip().lower() for item in raw_categories)
        if c in SENSITIVITY_CATEGORIES
    ]
    triggering = [c for c in categories if c in CENSOR_CATEGORIES]
    if triggering and not sensitive:
        # Contradictory answer — the model saw exposure worth naming.
        sensitive = True
    elif sensitive and categories and not triggering:
        # Record-only findings (emotional_distress) with no exposure
        # category: blur exists for bodily privacy, not misfortune
        # (2026-07-28). Verdict downgrades to clean; categories still
        # reported for logging/eval. A sensitive=true with NO categories
        # at all stays sensitive (unknown exposure — fail closed).
        sensitive = False
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))
    return ScreenResult(
        sensitive=sensitive, categories=categories, confidence=confidence,
    )


async def screen_clip(mp4: bytes, *, settings: Settings) -> ScreenResult:
    """Screen one assembled clip. One pinned-model Gemini call, native
    video input, structured output. Raises on ANY failure (missing key,
    timeout, API error, malformed answer) — the caller treats every
    raise as "censor full-frame", so this function never needs to guess.
    """
    if not settings.gemini_api_key:
        raise RuntimeError("NEMARA_GEMINI_API_KEY not configured")
    response_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=["sensitive", "categories", "confidence"],
        properties={
            "sensitive": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "categories": genai_types.Schema(
                type=genai_types.Type.ARRAY,
                items=genai_types.Schema(
                    type=genai_types.Type.STRING,
                    enum=list(SENSITIVITY_CATEGORIES),
                ),
            ),
            "confidence": genai_types.Schema(type=genai_types.Type.NUMBER),
        },
    )
    # Same fps pin as the detection providers' native-video path so the
    # screener samples the clip at the rate it was captured at (empty
    # env value = Gemini's own ~1fps default).
    video_metadata: genai_types.VideoMetadata | None = None
    fps_raw = settings.gemini_video_fps.strip()
    if fps_raw:
        try:
            video_metadata = genai_types.VideoMetadata(fps=int(float(fps_raw)))
        except ValueError:
            video_metadata = None
    client = get_client(settings.gemini_api_key)
    response = await asyncio.wait_for(
        client.aio.models.generate_content(
            model=settings.sensitivity_model,
            contents=[
                genai_types.Content(
                    role="user",
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=mp4, mime_type="video/mp4",
                            ),
                            video_metadata=video_metadata,
                        ),
                        genai_types.Part(text=_PROMPT),
                    ],
                )
            ],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        ),
        timeout=SCREEN_TIMEOUT_SECONDS,
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("sensitivity: gemini returned empty response")
    return parse_verdict(raw)
