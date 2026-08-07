"""Gemini provider.

Verbatim port of `nemara-temp-dashboard/lib/providers/gemini.ts`. Sends
the MP4 inline (base64) so Gemini's native video understanding kicks
in. `GEMINI_VIDEO_FPS` (empty default) lets Gemini pick its own
sampling; set to "1"–"24" to pin. Adds the Gemini-only speech bonus
fields (`speech_heard` + `speech_transcript`) to the prompt + schema
— these aren't part of the OpenAI fallback (no audio there).
"""

from collections.abc import Sequence

from google import genai
from google.genai import types as genai_types

from src.services.detection.prompt import (
    BASE_PROMPT,
    DetectorSpec,
    build_identity_addendum,
    build_prompt_and_schema,
)
from src.services.detection.providers._parse import parse_and_validate
from src.services.detection.providers.types import ProviderResult
from src.settings import Settings

_SPEECH_PROMPT_FRAGMENT = (
    "speech_heard: boolean. True ONLY if you can clearly hear actual, "
    "intelligible human speech in the audio. Set FALSE for silence, "
    "keyboard/typing, background or ambient noise, music, or any sound "
    "you cannot clearly make out as spoken words. When in doubt, set "
    "FALSE.\n"
    "speech_transcript: string. ONLY transcribe words you can actually "
    "hear, verbatim (up to ~200 chars; multiple speakers separated by "
    "' / '). If speech_heard is false, return an empty string. Do NOT "
    "guess, infer, or invent speech from the video, the scene, the "
    "on-screen activity, or non-speech sounds — never fabricate a "
    "transcript."
)
_SPEECH_FIELDS = {
    "speech_heard":      {"type": "boolean"},
    "speech_transcript": {"type": "string"},
}

# The fields a verdict is allowed to rest on, in the order the model
# must produce them. Everything not named here (people_present, every
# detector verdict) follows, keeping its own relative order.
#
# `description` sits last of these deliberately: it justifies what was
# detected, so it wants the evidence above it, but it must still land
# before any verdict — a narrative written AFTER the labels is a
# rationalisation of them, which is the failure mode the 2026-08-06
# rework set out to remove.
_EVIDENCE_ORDER = (
    "heard",
    "audio_source",
    "speech_heard",
    "speech_transcript",
    "seen",
    "description",
)


def _evidence_first_ordering(properties: dict) -> list[str]:
    """Property order for the response schema: evidence first, then —
    within each detector's own field group — reasoning BEFORE verdict.

    The second half is the field-level twin of the first (2026-08-08).
    Structured output emits fields in this order, and every fragment
    declares `<ident>_detected` before `<ident>_reasoning` — so the
    model was committing the boolean as one constrained token and
    writing its "reasoning" afterwards, as justification of a verdict
    already locked. That is the documented failure mode of constrained
    decoding (10-30% reasoning degradation, arXiv:2408.02442; verified
    on Gemini specifically by the alphabetical-ordering study the SDK
    docs cite), and it is why every FP's reasoning read as a confident
    narrative of something that never happened: the text existed to
    defend the token, not to derive it.

    Grouping is by shared prefix (key minus its last `_segment`), so
    `fall_reasoning` precedes `fall_detected`/`fall_confidence`,
    `custom_X_reasoning` precedes its verdict, and groups with no
    reasoning field (people_present/people_count) keep their declared
    order untouched. Group order itself is unchanged — only the order
    WITHIN a detector moves.
    """
    lead = [k for k in _EVIDENCE_ORDER if k in properties]
    rest = [k for k in properties if k not in set(lead)]
    groups: dict[str, list[str]] = {}
    group_order: list[str] = []
    for k in rest:
        prefix = k.rsplit("_", 1)[0]
        if prefix not in groups:
            groups[prefix] = []
            group_order.append(prefix)
        groups[prefix].append(k)
    out = list(lead)
    for prefix in group_order:
        ks = groups[prefix]
        out.extend(k for k in ks if k.endswith("_reasoning"))
        out.extend(k for k in ks if not k.endswith("_reasoning"))
    return out


def _media_resolution(value: str) -> genai_types.MediaResolution | None:
    """Map the env-string resolution to the SDK enum."""
    v = value.strip().lower()
    if v == "low":    return genai_types.MediaResolution.MEDIA_RESOLUTION_LOW
    if v == "medium": return genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    if v == "high":   return genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH
    return None


_THINKING_LEVELS = {"minimal", "low", "medium", "high"}


def _thinking_config(value: str) -> genai_types.ThinkingConfig | None:
    """Map the env-string thinking level to a ThinkingConfig, or None to
    send nothing and let the model use its own default.

    Note this is `thinking_level`, NOT the Gemini-2.5-era
    `thinking_budget` — a budget (even 0) is rejected outright by
    gemini-3.5-flash-lite with 400 INVALID_ARGUMENT. Unknown strings
    degrade to None rather than raising: a typo'd env value must fall
    back to the model default, never take detection down."""
    v = value.strip().lower()
    if v not in _THINKING_LEVELS:
        return None
    return genai_types.ThinkingConfig(thinking_level=v.upper())


def _map_type(t: str) -> genai_types.Type:
    if t == "boolean": return genai_types.Type.BOOLEAN
    if t == "number":  return genai_types.Type.NUMBER
    if t == "string":  return genai_types.Type.STRING
    raise ValueError(f"[gemini] unknown schema type: {t}")


async def gemini_provider(
    *,
    mp4: bytes,
    detectors: Sequence[DetectorSpec],
    log_tag: str,
    settings: Settings,
    patient_in_clip: bool = False,
    model: str | None = None,
    motion_overlay: bool = False,
) -> ProviderResult:
    if not settings.gemini_api_key:
        raise RuntimeError("NEMARA_GEMINI_API_KEY not configured")

    fps_raw = settings.gemini_video_fps.strip()
    sample_fps: float | None
    if fps_raw == "":
        sample_fps = None
    else:
        try:
            sample_fps = float(fps_raw)
        except ValueError as e:
            raise RuntimeError(
                f"NEMARA_GEMINI_VIDEO_FPS must be empty or 0–24, got: {fps_raw!r}"
            ) from e
        if not (0 <= sample_fps <= 24):
            raise RuntimeError(
                f"NEMARA_GEMINI_VIDEO_FPS must be in [0, 24], got: {sample_fps}"
            )

    # `motion_overlay` is a property of THIS CALL's artifact, not of the
    # deployment: only the RAPID/overlay channel tints its crop, so only
    # that channel's prompt should explain the red tint. The caller passes
    # it (executor: `plan.channel == "overlay"`); every other path leaves
    # it False.
    #
    # It used to be read from a global `settings.motion_overlay_enabled`,
    # which was wrong in both directions: the scene tint was removed
    # 2026-07-23 as a measured over-trigger, yet the flag stayed true in
    # .env, so every scene and person prompt carried ~730 chars
    # describing a red overlay that no scene clip has had since.
    built = build_prompt_and_schema(detectors, motion_overlay=motion_overlay)
    identity_addendum = build_identity_addendum(patient_in_clip)
    full_prompt = f"{built.prompt}\n\n{_SPEECH_PROMPT_FRAGMENT}{identity_addendum}"
    properties = {**built.schema_properties, **_SPEECH_FIELDS}
    required = built.schema_required + ["speech_heard", "speech_transcript"]

    # EVIDENCE BEFORE VERDICT IS AN ORDERING CLAIM, SO IT HAS TO BE
    # ORDERED (2026-08-07).
    #
    # BASE_PROMPT opens with "FILLING THEM IN THE ORDER GIVEN" and the
    # whole 2026-08-06 rework depends on it: `heard` and `seen` are the
    # ONLY places speech and objects may be established, and a detector
    # that needs a cup at a mouth must find it written above or answer
    # FALSE. That is a constraint on GENERATION ORDER — it only binds if
    # the evidence fields are actually emitted first.
    #
    # Nothing made them. A JSON schema's `properties` is a map, and
    # without `property_ordering` the model may emit fields in any
    # order, so the rule was a sentence in the prompt with no mechanism
    # under it. Measured live: a tick where `seen` read "holding a
    # smartphone in both hands while looking down at it" and the
    # description agreed, while the drinking-water verdict claimed the
    # subject "picked up a blue plastic bottle, raised it to his mouth"
    # — at confidence 1.0. The model saw correctly, described correctly,
    # and contradicted itself, because the verdict was not downstream of
    # anything.
    #
    # Stated explicitly rather than leaning on dict insertion order:
    # `_SPEECH_FIELDS` is merged in AFTER the detector fields, so plain
    # insertion order would have emitted the transcript last — the
    # audio half of the same bug, for the detectors that read speech.
    # Everything the model is allowed to treat as evidence goes first,
    # then the narrative, then every verdict.
    response_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=required,
        property_ordering=_evidence_first_ordering(properties),
        properties={
            name: genai_types.Schema(
                type=_map_type(field["type"]),
                description=field.get("description"),
            )
            for name, field in properties.items()
        },
    )

    client = genai.Client(api_key=settings.gemini_api_key)
    # `VideoMetadata.fps` is protobuf-typed (numeric, but the proto
    # may reject Python floats); cast to int before passing.
    video_metadata = (
        genai_types.VideoMetadata(fps=int(sample_fps)) if sample_fps else None
    )
    parts: list[genai_types.Part] = [
        genai_types.Part(
            inline_data=genai_types.Blob(data=mp4, mime_type="video/mp4"),
            video_metadata=video_metadata,
        ),
        genai_types.Part(text=full_prompt),
    ]
    config_kwargs: dict[str, object] = {
        "response_mime_type": "application/json",
        "response_schema": response_schema,
    }
    media_res = _media_resolution(settings.gemini_media_resolution)
    if media_res is not None:
        config_kwargs["media_resolution"] = media_res
    thinking = _thinking_config(settings.gemini_thinking_level)
    if thinking is not None:
        config_kwargs["thinking_config"] = thinking

    # NOTE: model fallback lives in the ROUTING chain (vlm_routing's
    # gemini -> gemini_fallback tier), not here — a provider-internal
    # ladder stacked on the chain double-retries (found 2026-07-27
    # when the merge surfaced both implementations).
    response = await client.aio.models.generate_content(
        model=model or settings.gemini_model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(**config_kwargs),
    )
    raw = (response.text or "").strip()
    if not raw:
        raise RuntimeError("gemini returned empty response")
    res = parse_and_validate(
        raw, provider="gemini", required=required, properties=properties,
    )
    meta = getattr(response, "usage_metadata", None)
    if meta is not None:
        import dataclasses

        # Billing truth (2026-07-19, the $14/day audit): thinking tokens
        # bill at the OUTPUT rate but live in `thoughts_token_count`, not
        # `candidates_token_count`. Recording candidates alone made most
        # of the real output bill invisible to our ledger.
        #
        # Corrected 2026-08-01: the original note said "Gemini 3.x models
        # THINK by default". That holds for 3.6-flash, NOT for
        # 3.5-flash-lite — measured, lite returned 0 thought tokens with
        # no config AND at thinking_level minimal/low, and reasoned only
        # at medium/high. Whether thoughts are non-zero now
        # depends on `gemini_thinking_level` (we send 'high'), so this
        # accounting matters more than it did, not less.
        # `output` is now the BILLED output (candidates + thoughts);
        # `thoughts` rides separately for observability.
        thoughts = int(getattr(meta, "thoughts_token_count", 0) or 0)
        res = dataclasses.replace(res, usage={
            "input": int(getattr(meta, "prompt_token_count", 0) or 0),
            "output": (
                int(getattr(meta, "candidates_token_count", 0) or 0)
                + thoughts
            ),
            "thoughts": thoughts,
        })
    return res
