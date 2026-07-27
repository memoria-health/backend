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


def _media_resolution(value: str) -> genai_types.MediaResolution | None:
    """Map the env-string resolution to the SDK enum."""
    v = value.strip().lower()
    if v == "low":    return genai_types.MediaResolution.MEDIA_RESOLUTION_LOW
    if v == "medium": return genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM
    if v == "high":   return genai_types.MediaResolution.MEDIA_RESOLUTION_HIGH
    return None


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

    # Gemini gets the native-video clip, which is the one the motion overlay is
    # applied to — so its prompt explains the red tint. The frame-sampled
    # providers use Re-ID's clean frames, so they leave this off.
    built = build_prompt_and_schema(
        detectors, motion_overlay=settings.motion_overlay_enabled)
    identity_addendum = build_identity_addendum(patient_in_clip)
    full_prompt = f"{built.prompt}\n\n{_SPEECH_PROMPT_FRAGMENT}{identity_addendum}"
    properties = {**built.schema_properties, **_SPEECH_FIELDS}
    required = built.schema_required + ["speech_heard", "speech_transcript"]

    response_schema = genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=required,
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

        # Billing truth (2026-07-19, the $14/day audit): Gemini 3.x
        # models THINK by default and thinking tokens bill at the
        # OUTPUT rate — but they live in `thoughts_token_count`, not
        # `candidates_token_count`. Recording candidates alone made
        # most of the real output bill invisible to our ledger.
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
