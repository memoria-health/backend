"""Compile a user's natural-language detector description into the
runtime contract the LLM pipeline needs.

When a caregiver writes a custom detector via the web dashboard, all
they provide is a short English description (e.g. "let me know if the
stove burner is on while no one's in the kitchen"). The detector
pipeline (`services/detection/registry.py`) needs three populated
columns on the row: `identifier`, `prompt_fragment`, `schema_json`.

This module fills that gap with one LLM call:

  - `identifier`  is generated server-side (`custom_<8-hex>`). We don't
                  let the model pick it — guarantees uniqueness, keeps
                  identifiers predictable (the dashboard's analyses
                  feed keys off them).

  - `schema_json` is templated server-side as the same standard trio
                  the preset detectors use:
                    {id}_detected:   boolean
                    {id}_confidence: number
                    {id}_reasoning:  string
                  This lets the existing AnalysisRow renderer pick up
                  custom detectors with no further frontend work.

  - `prompt_fragment` is what the LLM actually generates here. We give
                  it the user's name + description, plus a worked
                  example of the rigor we expect (lifted from the fall
                  preset), and ask for a single instruction block in
                  the same shape.

Provider order is Gemini → OpenAI fallback. Both are text-only here —
no video involved at compile time.
"""

from __future__ import annotations

import logging
import json
import re
from dataclasses import dataclass, field
from uuid import uuid4

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from src.settings import Settings

log = logging.getLogger(__name__)

# Curated icon set the inference may choose from. SF Symbol names —
# the iOS app + web both already render these. Keeping it a closed set
# means the model can't hallucinate an icon the clients can't draw.
_ALLOWED_ICONS = [
    "figure.fall", "figure.walk.motion", "door.left.hand.open", "pills",
    "fork.knife", "bed.double", "shower", "tv", "drop", "flame",
    "exclamationmark.triangle", "eye", "heart", "person", "cup.and.saucer",
    "moon", "sun.max", "clock", "bell", "car",
]


# ── Meta-prompt sent to Gemini / OpenAI to compile the user's description ──
#
# Worked example of the shape we want comes from the existing `fall`
# preset (see `services/onboarding.py:_FALL_PROMPT_FRAGMENT`). Showing
# the model what "good" looks like is far more reliable than describing
# the style in prose.

# Reactive (point_in_time) detectors bias toward catching everything —
# a missed safety event is far worse than a false alarm. Falls example.
_FEW_SHOT_EXAMPLE_INCLUSIVE = (
    "Example for a detector named 'Falls' with description "
    "'Sudden fall events':\n\n"
    "fall_detected: boolean. Be INCLUSIVE. Set TRUE for any of:\n"
    "  - a person fully collapsing, slipping, tripping, stumbling hard, sliding down to the floor;\n"
    "  - suddenly dropping to the ground from standing / a chair / a bed;\n"
    "  - a person who was visible upright then suddenly disappears downward out of the bottom of the frame;\n"
    "Set FALSE only if the person clearly remains upright the entire clip; OR walks out of frame sideways at "
    "normal walking pace; OR is never visible.\n"
    "fall_confidence: number 0–1. 0 = clearly nothing fall-like. 0.3–0.6 = ambiguous. 0.7+ = clear fall.\n"
    "fall_reasoning: 1 short sentence explaining your judgment — what specifically you saw or didn't see."
)

# Scheduled ("absence") detectors are the OPPOSITE: they fire an alert
# only if the watched signal was NEVER seen in a window. An eager
# false-positive ("yes it happened") would silently swallow that alert,
# so per-tick detection must be CONSERVATIVE — only count it as seen
# when clearly happening. Medication example, conservatively worded.
_FEW_SHOT_EXAMPLE_CONSERVATIVE = (
    "Example for a detector named 'Medication taken' with description "
    "'Confirms a pill was taken':\n\n"
    "medication_taken: boolean. Be CONSERVATIVE — set TRUE only when you "
    "CLEARLY see medication-taking: a pill/tablet going to the mouth, or "
    "unambiguous interaction with a pill bottle / weekly organizer / blister "
    "pack. Set FALSE if the view is obscured, the object is unclear, the "
    "person merely passes through, or no person is visible. When in doubt, "
    "FALSE.\n"
    "medication_confidence: number 0–1. 0 = clearly not happening. 0.3–0.6 = "
    "ambiguous (treat as FALSE). 0.7+ = clearly, unmistakably happening.\n"
    "medication_reasoning: 1 short sentence explaining your judgment — what "
    "specifically you saw or didn't see."
)


def _meta_prompt(
    *, identifier: str, name: str, description: str, user_text: str,
    scheduled: bool,
) -> str:
    """The instructions the compiler sends to the LLM. `identifier` is
    interpolated everywhere the runtime block needs the stable key.

    `scheduled` flips the inclusivity guidance: reactive detectors are
    inclusive (catch everything), scheduled/absence detectors are
    conservative (only count a clear positive, so a real miss surfaces
    the absence alert instead of being swallowed by a false positive).
    """
    if scheduled:
        detected_guidance = (
            "Be CONSERVATIVE — set TRUE ONLY when the thing is CLEARLY and "
            "unmistakably happening. On any ambiguity, partial view, or "
            "uncertainty, set FALSE. This detector fires an alert when the "
            "behavior is ABSENT over a time window, so a false positive "
            "would wrongly suppress that alert."
        )
        example = _FEW_SHOT_EXAMPLE_CONSERVATIVE
    else:
        detected_guidance = (
            "Use INCLUSIVE wording when borderline situations should still "
            "fire — a missed event is worse than a false alarm."
        )
        example = _FEW_SHOT_EXAMPLE_INCLUSIVE

    return (
        "You are configuring a computer-vision detector that runs on "
        "short (~15s) clips from a wall-mounted home security camera.\n\n"
        "The caregiver wants a new detector with this description:\n"
        f"  Name:        {name!r}\n"
        f"  Description: {description!r}\n"
        f"  What to detect (the caregiver's words):\n"
        f"  '''\n"
        f"  {user_text}\n"
        f"  '''\n\n"
        f"Generate a single instruction block in EXACTLY this format, "
        f"using `{identifier}` as the stable identifier:\n\n"
        f"  {identifier}_detected: boolean. <Detection criteria — when to set "
        f"TRUE vs FALSE, drawn from the caregiver's words but expanded with "
        f"common edge cases and false-positive guards. Be specific. Mention "
        f"how to handle 'person walks out of frame', 'no person visible', "
        f"and similar ambiguous cases relevant to this detector. "
        f"{detected_guidance}>\n\n"
        f"  {identifier}_confidence: number 0–1. <Calibration: 0 = clearly "
        f"not happening; 0.3–0.6 = ambiguous; 0.7+ = clearly happening.>\n\n"
        f"  {identifier}_reasoning: 1 short sentence explaining your "
        f"judgment, citing what you specifically saw.\n\n"
        f"Match the rigor of this example:\n\n"
        f"{example}\n\n"
        "Total length under ~1500 characters. Return ONLY the three-field "
        "instruction block — no preamble, no markdown, no quotes around it."
    )


# ── Public API ────────────────────────────────────────────────────


@dataclass(frozen=True)
class CompiledDetector:
    """The runtime contract for a freshly-compiled custom detector. Caller
    writes these three columns onto the new `detectors` row.
    """

    identifier: str
    prompt_fragment: str
    schema_json: dict[str, dict[str, str]]


def _gen_identifier() -> str:
    """Stable, unique identifier for a custom detector. `custom_` prefix
    is the convention noted in `schema.sql` so consumers can detect the
    source from the identifier alone. 8 hex = 2^32 namespace, plenty.
    """
    return f"custom_{uuid4().hex[:8]}"


def _schema_for(identifier: str) -> dict[str, dict[str, str]]:
    """Standard {id}_detected/_confidence/_reasoning trio. Matches the
    schema the existing preset detectors use, so the dashboard's
    AnalysisRow renderer picks up custom detectors with no changes.
    """
    return {
        f"{identifier}_detected":   {"type": "boolean"},
        f"{identifier}_confidence": {"type": "number"},
        f"{identifier}_reasoning":  {"type": "string"},
    }


_BACKTICK_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def _clean_fragment(raw: str, identifier: str) -> str:
    """Strip code fences, trim, and validate the output references the
    expected identifier. If the model wrapped the response in ```…```
    blocks (it sometimes does, despite the "no markdown" instruction),
    peel them off. If the result doesn't mention `{identifier}_detected`,
    we treat the compile as failed and let the caller fall over to the
    next provider.
    """
    cleaned = _BACKTICK_FENCE_RE.sub("", raw).strip()
    if f"{identifier}_detected" not in cleaned:
        raise ValueError(
            f"compiled prompt does not reference '{identifier}_detected'; "
            f"got: {cleaned[:200]}…"
        )
    return cleaned


async def _compile_with_gemini(
    *, identifier: str, name: str, description: str, user_text: str,
    scheduled: bool, settings: Settings,
) -> str:
    """One Gemini call. Returns the cleaned prompt_fragment string."""
    if not settings.gemini_api_key:
        raise RuntimeError("NEMARA_GEMINI_API_KEY not configured")
    client = genai.Client(api_key=settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=_meta_prompt(
                    identifier=identifier,
                    name=name,
                    description=description,
                    user_text=user_text,
                    scheduled=scheduled,
                ))],
            ),
        ],
        # No response_schema — we want freeform text, not JSON.
        config=genai_types.GenerateContentConfig(
            response_mime_type="text/plain",
        ),
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("gemini returned empty compile response")
    return _clean_fragment(raw, identifier)


async def _compile_with_openai(
    *, identifier: str, name: str, description: str, user_text: str,
    scheduled: bool, settings: Settings,
) -> str:
    """One OpenAI call. Returns the cleaned prompt_fragment string."""
    if not settings.openai_api_key:
        raise RuntimeError("NEMARA_OPENAI_API_KEY not configured")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "user", "content": _meta_prompt(
                identifier=identifier,
                name=name,
                description=description,
                user_text=user_text,
                scheduled=scheduled,
            )},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("openai returned empty compile response")
    return _clean_fragment(raw, identifier)


async def compile_custom_detector(
    *,
    name: str,
    description: str,
    user_text: str,
    scheduled: bool = False,
    settings: Settings,
) -> CompiledDetector:
    """Compile a caregiver's natural-language description into a
    runtime-ready (identifier, prompt_fragment, schema_json) triple.

    Tries Gemini first (cheaper + faster for short text); falls back to
    OpenAI if Gemini errors out or produces a malformed response. If
    both fail, raises RuntimeError — the caller should map this to a
    503 and surface a clear error to the user (don't create a broken
    detector row that silently never fires).

    `user_text` is the caregiver's raw "what to detect" string. We
    intentionally pass `name` + `description` separately so the model
    has a chance to disambiguate (e.g., the user might use a vague
    word in `user_text` but a clearer phrase in `description`).

    `scheduled` tunes the per-tick detection's bias: reactive detectors
    are inclusive (catch everything), scheduled/absence detectors are
    conservative (only a clear positive counts, so a real miss surfaces
    the absence alert).
    """
    identifier = _gen_identifier()

    try:
        fragment = await _compile_with_gemini(
            identifier=identifier, name=name, description=description,
            user_text=user_text, scheduled=scheduled, settings=settings,
        )
        log.info(
            "detector_compiler.gemini_ok identifier=%s name=%r fragment_chars=%d",
            identifier, name, len(fragment),
        )
        return CompiledDetector(
            identifier=identifier,
            prompt_fragment=fragment,
            schema_json=_schema_for(identifier),
        )
    except Exception as gemini_err:
        log.warning(
            "detector_compiler.gemini_failed identifier=%s name=%r err=%s — trying openai",
            identifier, name, gemini_err,
        )

    try:
        fragment = await _compile_with_openai(
            identifier=identifier, name=name, description=description,
            user_text=user_text, scheduled=scheduled, settings=settings,
        )
        log.info(
            "detector_compiler.openai_ok identifier=%s name=%r fragment_chars=%d",
            identifier, name, len(fragment),
        )
        return CompiledDetector(
            identifier=identifier,
            prompt_fragment=fragment,
            schema_json=_schema_for(identifier),
        )
    except Exception as openai_err:
        log.error(
            "detector_compiler.both_providers_failed identifier=%s name=%r err=%s",
            identifier, name, openai_err,
        )
        raise RuntimeError(
            "could not compile detector — both gemini and openai failed; "
            "check NEMARA_GEMINI_API_KEY and NEMARA_OPENAI_API_KEY"
        ) from openai_err


# ── Inference: classify + structure a plain-English ask ─────────────
#
# Separate from compile_custom_detector (which generates the runtime
# prompt_fragment). Inference runs synchronously when the user clicks
# "Generate" on the create form and returns the high-level fields for
# the review screen — shape, category, icon, sensitivity, and (if it's
# an absence detector) the schedule window. The heavy prompt_fragment
# compile still happens in the background on Create, as before.


@dataclass(frozen=True)
class InferredConfig:
    shape: str                       # "point_in_time" | "scheduled"
    category: str                    # "safety" | "routine"
    icon_name: str
    sensitivity: str                 # "low" | "standard" | "high"
    schedule: dict | None = field(default=None)  # None unless scheduled


def _infer_meta_prompt(*, name: str | None, prompt: str) -> str:
    return (
        "You configure home-care camera detectors from a caregiver's "
        "plain-English request. Classify the request and extract "
        "structured settings. Return ONLY JSON (no markdown).\n\n"
        f"Caregiver's request:\n  Name: {name or '(none given)'}\n"
        f"  What to detect: '''{prompt}'''\n\n"
        "Decide `shape`:\n"
        "  - 'scheduled' if the request is about whether something happens "
        "(or FAILS to happen) within a recurring time window — e.g. "
        "'check she takes her meds between 11am and 5pm daily', 'alert me "
        "if she doesn't get up by 10am'. These fire on ABSENCE at a "
        "deadline.\n"
        "  - 'point_in_time' for everything reactive — fires the moment "
        "it's seen (falls, wandering, agitation, a visitor, etc.).\n\n"
        "JSON fields:\n"
        '  "shape": "point_in_time" | "scheduled"\n'
        '  "category": "safety" | "routine"  (medication/meals/hydration/'
        'routines = routine; falls/wandering/agitation/hazards = safety)\n'
        f'  "icon_name": one of {_ALLOWED_ICONS}\n'
        '  "sensitivity": "low" | "standard" | "high"  (default "standard")\n'
        '  "schedule": null if point_in_time. If scheduled, an object:\n'
        '     {"cadence":"daily"|"weekly",'
        ' "days_of_week":[0..6 or null] (0=Mon..6=Sun, weekly only),'
        ' "window_start":"HH:MM" 24h,'
        ' "window_end":"HH:MM",'
        ' "deadline":"HH:MM" (usually = window_end)}\n\n'
        "IMPORTANT: leave any field you cannot confidently infer as null "
        "(for schedule sub-fields) — do NOT guess times. The user will "
        "fill blanks. Always pick a shape, category, icon, sensitivity."
    )


def _coerce_inferred(raw: dict) -> InferredConfig:
    """Validate/normalize the model's JSON into an InferredConfig.
    Unknown enum values fall back to safe defaults; out-of-set icons
    fall back to 'eye'."""
    shape = raw.get("shape")
    if shape not in ("point_in_time", "scheduled"):
        shape = "point_in_time"
    category = raw.get("category")
    if category not in ("safety", "routine"):
        category = "safety"
    sensitivity = raw.get("sensitivity")
    if sensitivity not in ("low", "standard", "high"):
        sensitivity = "standard"
    icon = raw.get("icon_name")
    if icon not in _ALLOWED_ICONS:
        icon = "eye"

    schedule: dict | None = None
    if shape == "scheduled":
        s = raw.get("schedule") or {}
        cadence = s.get("cadence") if s.get("cadence") in ("daily", "weekly") else None
        days = s.get("days_of_week")
        if not isinstance(days, list):
            days = None
        else:
            days = [d for d in days if isinstance(d, int) and 0 <= d <= 6] or None

        def _hhmm(v: object) -> str | None:
            return v if isinstance(v, str) and re.fullmatch(r"\d{2}:\d{2}", v) else None

        schedule = {
            "cadence": cadence,
            "days_of_week": days,
            "window_start": _hhmm(s.get("window_start")),
            "window_end": _hhmm(s.get("window_end")),
            "deadline": _hhmm(s.get("deadline")),
        }

    return InferredConfig(
        shape=shape, category=category, icon_name=icon,
        sensitivity=sensitivity, schedule=schedule,
    )


def _parse_json_lenient(raw: str) -> dict:
    """Parse JSON that may be wrapped in ```json fences."""
    cleaned = _BACKTICK_FENCE_RE.sub("", raw).strip()
    return json.loads(cleaned)


async def _infer_with_gemini(*, name: str | None, prompt: str, settings: Settings) -> dict:
    if not settings.gemini_api_key:
        raise RuntimeError("NEMARA_GEMINI_API_KEY not configured")
    client = genai.Client(api_key=settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_model,
        contents=[genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=_infer_meta_prompt(name=name, prompt=prompt))],
        )],
        config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("gemini returned empty inference response")
    return _parse_json_lenient(raw)


async def _infer_with_openai(*, name: str | None, prompt: str, settings: Settings) -> dict:
    if not settings.openai_api_key:
        raise RuntimeError("NEMARA_OPENAI_API_KEY not configured")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": _infer_meta_prompt(name=name, prompt=prompt)}],
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("openai returned empty inference response")
    return _parse_json_lenient(raw)


async def infer_detector_config(
    *, name: str | None, prompt: str, settings: Settings,
) -> InferredConfig:
    """Classify + structure a plain-English detector request for the
    create review screen. Gemini first, OpenAI fallback. Raises
    RuntimeError if both fail (caller maps to 503)."""
    try:
        raw = await _infer_with_gemini(name=name, prompt=prompt, settings=settings)
        log.info("detector_infer.gemini_ok name=%r shape=%s", name, raw.get("shape"))
        return _coerce_inferred(raw)
    except Exception as gemini_err:
        log.warning("detector_infer.gemini_failed name=%r err=%s — trying openai", name, gemini_err)
    try:
        raw = await _infer_with_openai(name=name, prompt=prompt, settings=settings)
        log.info("detector_infer.openai_ok name=%r shape=%s", name, raw.get("shape"))
        return _coerce_inferred(raw)
    except Exception as openai_err:
        log.error("detector_infer.both_failed name=%r err=%s", name, openai_err)
        raise RuntimeError(
            "could not infer detector config — both gemini and openai failed"
        ) from openai_err
