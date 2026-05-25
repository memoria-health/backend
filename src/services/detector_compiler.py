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
import re
from dataclasses import dataclass
from uuid import uuid4

from google import genai
from google.genai import types as genai_types
from openai import AsyncOpenAI

from src.settings import Settings

log = logging.getLogger(__name__)


# ── Meta-prompt sent to Gemini / OpenAI to compile the user's description ──
#
# Worked example of the shape we want comes from the existing `fall`
# preset (see `services/onboarding.py:_FALL_PROMPT_FRAGMENT`). Showing
# the model what "good" looks like is far more reliable than describing
# the style in prose.

_FEW_SHOT_EXAMPLE = (
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


def _meta_prompt(*, identifier: str, name: str, description: str, user_text: str) -> str:
    """The instructions the compiler sends to the LLM. `identifier` is
    interpolated everywhere the runtime block needs the stable key.
    """
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
        f"and similar ambiguous cases relevant to this detector. Use "
        f"INCLUSIVE wording when borderline situations should still fire.>\n\n"
        f"  {identifier}_confidence: number 0–1. <Calibration: 0 = clearly "
        f"not happening; 0.3–0.6 = ambiguous; 0.7+ = clearly happening.>\n\n"
        f"  {identifier}_reasoning: 1 short sentence explaining your "
        f"judgment, citing what you specifically saw.\n\n"
        f"Match the rigor of this example:\n\n"
        f"{_FEW_SHOT_EXAMPLE}\n\n"
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
    settings: Settings,
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
    settings: Settings,
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
    """
    identifier = _gen_identifier()

    try:
        fragment = await _compile_with_gemini(
            identifier=identifier, name=name, description=description,
            user_text=user_text, settings=settings,
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
            user_text=user_text, settings=settings,
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
