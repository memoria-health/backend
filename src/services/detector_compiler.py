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

# Reactive but NON-emergency (routine) detectors. These fire the moment they're
# seen (like safety detectors) but a false alarm is merely annoying, not a
# missed emergency — so the bar must be HIGH. Watching-TV example, the classic
# false-positive case: a glance at the screen must NOT trip it.
_FEW_SHOT_EXAMPLE_ROUTINE_REACTIVE = (
    "Example for a detector named 'Watching TV' with description "
    "'Patient is watching television':\n\n"
    "watching_tv_detected: boolean. Be CONSERVATIVE — this is a routine, "
    "non-emergency behaviour, so only set TRUE when the person is CLEARLY and "
    "SUSTAINEDLY engaged with the TV: seated or lying down, oriented toward a "
    "visibly-on screen, for most of the clip. Set FALSE for a brief glance "
    "toward the TV, walking past it, a screen that's off or out of view, or a "
    "person in the room doing something else. When unsure, FALSE.\n"
    "watching_tv_confidence: number 0–1. 0 = clearly not; 0.3–0.6 = ambiguous "
    "(keep detected FALSE); 0.7+ = clearly, sustainedly watching.\n"
    "watching_tv_reasoning: 1 short sentence on what you saw — how long they "
    "were engaged, the screen state, and their body orientation."
)


def _meta_prompt(
    *, identifier: str, name: str, description: str, user_text: str,
    absence: bool, category: str,
) -> str:
    """The instructions the compiler sends to the LLM. `identifier` is
    interpolated everywhere the runtime block needs the stable key.

    Inclusivity is chosen by STAKES, not by the trigger alone:
      - absence → conservative: only a clear positive
        counts, so a real miss surfaces the absence alert instead of being
        swallowed.
      - reactive + safety  → inclusive: a missed emergency is far worse than a
        false alarm, so fire on borderline signs.
      - reactive + routine → conservative: a false alarm is just noise, so only
        a clear, sustained occurrence fires (this is the "watching TV fired on a
        glance" fix). The LLM is told to calibrate strictness to the stakes.
    """
    if absence:
        detected_guidance = (
            "Be CONSERVATIVE — set TRUE ONLY when the thing is CLEARLY and "
            "unmistakably happening. On any ambiguity, partial view, or "
            "uncertainty, set FALSE. This detector fires an alert when the "
            "behavior is ABSENT over a time window, so a false positive "
            "would wrongly suppress that alert."
        )
        example = _FEW_SHOT_EXAMPLE_CONSERVATIVE
    elif category == "safety":
        detected_guidance = (
            "FIRST weigh the stakes: this is a SAFETY/health detector, so a "
            "missed event could be an emergency. Be GENEROUS — use INCLUSIVE "
            "criteria so borderline or partial signs still fire; set FALSE only "
            "when you're confident nothing concerning happened."
        )
        example = _FEW_SHOT_EXAMPLE_INCLUSIVE
    else:
        detected_guidance = (
            "FIRST weigh the stakes: this is a ROUTINE, non-emergency behaviour, "
            "so a false alarm is just noise the caregiver must dismiss. Be "
            "CONSERVATIVE — set TRUE only on a CLEAR, SUSTAINED, deliberate "
            "occurrence; a fleeting glance, an incidental or momentary instance, "
            "or anything ambiguous → FALSE. Use your judgment to calibrate how "
            "strict to be: the more disruptive a false alarm would be, the "
            "higher the bar."
        )
        example = _FEW_SHOT_EXAMPLE_ROUTINE_REACTIVE

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
    absence: bool, category: str, settings: Settings,
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
                    absence=absence,
                    category=category,
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
    absence: bool, category: str, settings: Settings,
) -> str:
    """One OpenAI call. Returns the cleaned prompt_fragment string.

    Uses `openai_compile_model` (a stronger frontier/thinking model) — compile
    is a one-off on Create, so the better prompt is worth the extra latency.
    """
    if not settings.openai_api_key:
        raise RuntimeError("NEMARA_OPENAI_API_KEY not configured")
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    response = await client.chat.completions.create(
        model=settings.openai_compile_model,
        messages=[
            {"role": "user", "content": _meta_prompt(
                identifier=identifier,
                name=name,
                description=description,
                user_text=user_text,
                absence=absence,
                category=category,
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
    absence: bool = False,
    category: str = "safety",
    settings: Settings,
    identifier: str | None = None,
    refinements: list[str] | None = None,
) -> CompiledDetector:
    """Compile a caregiver's natural-language description into a
    runtime-ready (identifier, prompt_fragment, schema_json) triple.

    Tries the frontier OpenAI compile model FIRST (a thinking model writes a
    sharper, better-calibrated seeded prompt; compile is a rare one-off on
    Create, so the extra latency is fine), falling back to Gemini if OpenAI
    errors or returns a malformed response. If both fail, raises RuntimeError —
    the caller maps it to a "compile failed" badge (don't create a broken row
    that silently never fires).

    `user_text` is the caregiver's raw "what to detect" string. `name` +
    `description` are passed separately so the model can disambiguate.

    `absence` + `category` tune the firing bias: absence (schedule/negative)
    and routine reactive detectors are conservative (only a clear, sustained
    positive fires); safety reactive detectors stay inclusive (a missed
    emergency is worse than a false alarm).
    """
    # Recompiles (prompt edit / caregiver refinement) pass the existing
    # identifier so analyses history and schema keys stay stable; only
    # a brand-new detector mints one.
    identifier = identifier or _gen_identifier()
    if refinements:
        user_text = (
            f"{user_text}\n\nThe caregiver reviewed this detector's "
            "behavior and asked for these changes (apply ALL of them, "
            "newest last; they refine, never replace, the description "
            "above — and never weaken the safety bias):\n"
            + "\n".join(f"  {i + 1}. {r}" for i, r in enumerate(refinements))
        )

    try:
        fragment = await _compile_with_openai(
            identifier=identifier, name=name, description=description,
            user_text=user_text, absence=absence, category=category,
            settings=settings,
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
        log.warning(
            "detector_compiler.openai_failed identifier=%s name=%r err=%s — trying gemini",
            identifier, name, openai_err,
        )

    try:
        fragment = await _compile_with_gemini(
            identifier=identifier, name=name, description=description,
            user_text=user_text, absence=absence, category=category,
            settings=settings,
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
        log.error(
            "detector_compiler.both_providers_failed identifier=%s name=%r err=%s",
            identifier, name, gemini_err,
        )
        raise RuntimeError(
            "could not compile detector — both openai and gemini failed; "
            "check NEMARA_OPENAI_API_KEY and NEMARA_GEMINI_API_KEY"
        ) from gemini_err


# ── Inference: classify + structure a plain-English ask ─────────────
#
# Separate from compile_custom_detector (which generates the runtime
# prompt_fragment). Inference runs synchronously when the user clicks
# "Generate" on the create form and returns the high-level fields for
# the review screen — shape, category, icon, sensitivity, and (if it's
# an absence detector) the schedule window. The heavy prompt_fragment
# compile still happens in the background on Create, as before.


# Continuous ("state") detector hysteresis buckets → seconds. The model
# picks a bucket by the activity; we map to seconds and clamp to a sane band
# so a bad inference can't make a detector never re-fire. Retune centrally
# here. Default bucket when the model omits/garbles it: "moderate".
_GAP_BUCKET_SECONDS: dict[str, int] = {
    "brief": 60,        # ~1 min — pacing / restlessness / agitation: a short lull = new
    "moderate": 300,    # ~5 min — eating, on the phone, reading
    "extended": 900,    # ~15 min — watching TV, sleeping, sitting: bridge brief breaks
}
_DEFAULT_GAP_BUCKET = "moderate"


@dataclass(frozen=True)
class InferredConfig:
    # Alerting spine (2026-07-26).
    alert_trigger: str               # "leading"|"trailing"|"duration"|"absence"
    is_sustained: bool               # instant vs sustained
    category: str                    # "safety" | "routine"
    icon_name: str
    sensitivity: str                 # "low" | "standard" | "high"
    schedule: dict | None = field(default=None)  # None unless alert_trigger=schedule
    # Set only when is_sustained: the rising-edge episode gap (seconds).
    episode_gap_seconds: int | None = field(default=None)
    # Set only when alert_trigger == "duration": the "too long" threshold.
    duration_seconds: int | None = field(default=None)
    # Alert-meaning name nudge (2026-07-28): absence names must be
    # miss-phrased or the "Possible <name>" title inverts. None when the
    # model returned nothing usable.
    suggested_name: str | None = field(default=None)
    # General active window (ANY trigger): "HH:MM-HH:MM" 24h or "ALWAYS".
    # None when the model gave nothing usable (caller keeps the current value).
    when_scope: str | None = field(default=None)
    # A room/place the caregiver named ("kitchen"); the caller matches it to a
    # real camera room. None if no place was named.
    where_hint: str | None = field(default=None)


def _infer_meta_prompt(*, name: str | None, prompt: str) -> str:
    return (
        "You configure home-care camera detectors from a caregiver's "
        "plain-English request. Classify the request and extract "
        "structured settings. Return ONLY JSON (no markdown).\n\n"
        f"Caregiver's request:\n  Name: {name or '(none given)'}\n"
        f"  What to detect: '''{prompt}'''\n\n"
        "STEP 1 — `alert_trigger`: which MOMENT does the caregiver want to be "
        "told about?\n"
        "  - 'leading' (the default): the moment it STARTS or happens — a "
        "fall, choking, she starts pacing, a visitor arrives.\n"
        "  - 'trailing': when a sustained activity ENDS — 'when she's done "
        "eating', 'once she's finished her bath'.\n"
        "  - 'duration': once something has gone on TOO LONG — 'if the stove's "
        "been on more than 10 minutes', 'if she's on the floor over 5 "
        "minutes', 'prolonged stillness'. Put the stated length in "
        "duration_seconds.\n"
        "  - 'absence': the thing did NOT happen within its window — 'if she "
        "doesn't take her meds by 5pm', 'alert me if she hasn't gotten up by "
        "10am'. Non-occurrence is only knowable at the window's end, so any "
        "'doesn't / hasn't happened by <time>' request is 'absence'. (A "
        "positive 'tell me WHEN she does X' is just 'leading' — use the active "
        "window for the timing, not this.)\n\n"
        "STEP 2 — `is_sustained`: true if the thing unfolds over a stretch "
        "(watching TV, pacing, eating, bathing, agitation, lying still); false "
        "if it's instantaneous (a fall, choking, arriving, leaving). "
        "'trailing' and 'duration' are always sustained.\n\n"
        "JSON fields:\n"
        '  "alert_trigger": "leading" | "trailing" | "duration" | "absence"\n'
        '  "is_sustained": true | false\n'
        '  "duration_seconds": integer seconds — REQUIRED when '
        'alert_trigger="duration" (else null). Convert the stated time '
        '("10 minutes" -> 600, "an hour" -> 3600). Null if no length is given.\n'
        '  "episode_gap": "brief" | "moderate" | "extended" | null. REQUIRED '
        'when is_sustained (null otherwise): how long the activity must be '
        'ABSENT before a return counts as a NEW, separate event.\n'
        '     - "brief" (~1 min): pacing / restlessness / agitation.\n'
        '     - "moderate" (~5 min): eating, on the phone, reading.\n'
        '     - "extended" (~15 min): watching TV, sleeping, sitting — a brief '
        'break (e.g. a bathroom trip) should NOT split the session.\n'
        '  "category": "safety" | "routine".\n'
        '     - "safety" = could be an emergency or cause harm if missed: '
        'falls, wandering / leaving the house, fire / stove / hazards, choking, '
        'aggression or injury, prolonged no-movement, medication missed. '
        'When bodily harm or a health risk is plausible, choose "safety".\n'
        '     - "routine" = everyday, non-emergency behaviour where a false '
        'alarm is merely annoying noise: watching TV, using a phone or tablet, '
        'a visitor arriving, eating a snack, reading, sitting on the couch.\n'
        '  "suggested_name": a short display name stating what the ALERT '
        "MEANS. Event titles read 'Possible <name>', so for "
        'alert_trigger="absence" the name MUST be phrased as the MISS '
        '("Missed evening medication", "Skipped breakfast") — NEVER the '
        'positive activity ("Evening medication" would title as "Possible '
        'Evening medication", the inverse of the alert). For other triggers, '
        "name the activity itself. Return the caregiver's own name when it "
        'already reads correctly; improve it only when it would mislead.\n'
        f'  "icon_name": one of {_ALLOWED_ICONS}\n'
        '  "sensitivity": "low" | "standard" | "high" (default "standard"). '
        'Lower it for low-stakes routine behaviours (e.g. watching TV -> '
        '"low"); raise it for high-stakes safety events (e.g. a fall -> '
        '"high").\n'
        '  "schedule": null unless alert_trigger="absence". Else the window it '
        'must have happened within:\n'
        '     {"cadence":"daily"|"weekly",'
        ' "days_of_week":[0..6 or null] (0=Mon..6=Sun, weekly only),'
        ' "window_start":"HH:MM" 24h,'
        ' "window_end":"HH:MM",'
        ' "deadline":"HH:MM" (usually = window_end)}\n\n'
        '  "when_scope": the active window the detector should WATCH during, '
        'as "HH:MM-HH:MM" 24h, or "ALWAYS" for any time of day. This is WHEN '
        'it is allowed to watch (separate from the absence deadline above). '
        '"at night" -> "22:00-06:00", "in the mornings" -> "06:00-11:00", '
        '"during the day" -> "08:00-18:00". Use "ALWAYS" if no time of day is '
        'implied.\n'
        '  "where_hint": a single room or place the caregiver named, like '
        '"kitchen", "bedroom", "living room", "bathroom", or "garage", or null '
        'if they did not name a place. Just the place name, nothing else.\n\n'
        "IMPORTANT: leave any TIME or DURATION you cannot confidently infer "
        "as null — do NOT guess. The user fills blanks. Always pick "
        "alert_trigger, is_sustained, category, icon, sensitivity."
    )


def _coerce_inferred(raw: dict) -> InferredConfig:
    """Validate/normalize the model's JSON into an InferredConfig.
    Unknown enum values fall back to safe defaults; out-of-set icons
    fall back to 'eye'."""
    alert_trigger = raw.get("alert_trigger")
    if alert_trigger not in ("leading", "trailing", "duration", "absence"):
        alert_trigger = "leading"
    is_sustained = bool(raw.get("is_sustained"))
    # trailing / duration are sustained by definition; absence is neither.
    if alert_trigger in ("trailing", "duration"):
        is_sustained = True
    elif alert_trigger == "absence":
        is_sustained = False

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
    if alert_trigger == "absence":
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

    # Sustained detectors always get a gap (default bucket if the model
    # omitted/garbled it) so they can never be left without a rising-edge
    # window. None for instant / schedule.
    episode_gap_seconds: int | None = None
    if is_sustained:
        bucket = raw.get("episode_gap")
        episode_gap_seconds = _GAP_BUCKET_SECONDS.get(
            bucket if isinstance(bucket, str) else "",
            _GAP_BUCKET_SECONDS[_DEFAULT_GAP_BUCKET],
        )

    # Duration threshold: only for the duration trigger; null if not stated
    # (the review screen then prompts for it — never guessed).
    duration_seconds: int | None = None
    if alert_trigger == "duration":
        d = raw.get("duration_seconds")
        if isinstance(d, (int, float)) and d > 0:
            duration_seconds = int(d)

    # General active window (any trigger). Accept "HH:MM-HH:MM" or "ALWAYS";
    # anything else stays None so the caller keeps the current value.
    when_scope: str | None = None
    ws = raw.get("when_scope")
    if isinstance(ws, str):
        ws = ws.strip()
        if ws.upper() == "ALWAYS":
            when_scope = "ALWAYS"
        elif re.fullmatch(r"\d{2}:\d{2}-\d{2}:\d{2}", ws):
            when_scope = ws

    # A place the caregiver named; the caller matches it to a real camera room.
    where_hint: str | None = None
    wh = raw.get("where_hint")
    if isinstance(wh, str) and wh.strip():
        where_hint = wh.strip()

    return InferredConfig(
        suggested_name=(
            str(raw.get("suggested_name")).strip()[:60]
            if isinstance(raw.get("suggested_name"), str)
            and raw.get("suggested_name").strip() else None
        ),
        alert_trigger=alert_trigger,
        is_sustained=is_sustained,
        category=category, icon_name=icon,
        sensitivity=sensitivity, schedule=schedule,
        episode_gap_seconds=episode_gap_seconds,
        when_scope=when_scope, where_hint=where_hint,
        duration_seconds=duration_seconds,
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
        log.info("detector_infer.gemini_ok name=%r trigger=%s", name, raw.get("alert_trigger"))
        return _coerce_inferred(raw)
    except Exception as gemini_err:
        log.warning("detector_infer.gemini_failed name=%r err=%s — trying openai", name, gemini_err)
    try:
        raw = await _infer_with_openai(name=name, prompt=prompt, settings=settings)
        log.info("detector_infer.openai_ok name=%r trigger=%s", name, raw.get("alert_trigger"))
        return _coerce_inferred(raw)
    except Exception as openai_err:
        log.error("detector_infer.both_failed name=%r err=%s", name, openai_err)
        raise RuntimeError(
            "could not infer detector config — both gemini and openai failed"
        ) from openai_err


# ── Caregiver-facing summary (customs visibility, 2026-07-18) ─────


_SUMMARY_PROMPT = """The text below is the internal instruction block a \
home-camera AI follows for one detector. Rewrite it for the patient's \
FAMILY — people with zero technical background.

Return 3 to 5 short lines, one per line, no numbering or bullets:
- at least one line starting "Counts: " — what WILL trigger it
- at least one line starting "Doesn't count: " — what will NOT
- one line starting "Caution: " — how careful it is before alerting

Rules: everyday words only. Never mention booleans, confidence, fields, \
identifiers, frames, clips, cameras, models, or code of any kind. Never \
promise certainty. Under 320 characters total.

Internal instructions:
'''
{fragment}
'''"""


async def summarize_detector_prompt(
    *, fragment: str, settings: Settings,
) -> list[str] | None:
    """Plain-English criteria lines for a compiled prompt_fragment.

    One cheap Gemini call (same model as detection — lite-tier). Runs
    after every compile/recompile so the summary can never drift from
    the fragment it describes. Returns None on ANY failure — the UI
    hides the card; a summary must never block or break a compile."""
    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[genai_types.Content(
                role="user",
                parts=[genai_types.Part(
                    text=_SUMMARY_PROMPT.format(fragment=fragment),
                )],
            )],
            config=genai_types.GenerateContentConfig(
                response_mime_type="text/plain",
            ),
        )
        lines = [
            ln.strip().lstrip("-• ").strip()
            for ln in (response.text or "").splitlines()
            if ln.strip()
        ]
        return lines[:5] or None
    except Exception:  # noqa: BLE001 — summary is best-effort by contract
        log.warning("detector_compiler.summarize_failed", exc_info=True)
        return None
