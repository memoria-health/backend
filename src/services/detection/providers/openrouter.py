"""OpenRouter provider — run an OpenRouter-hosted VLM (default Qwen3-VL-32B)
on the same frame-based path as the OpenAI provider.

OpenRouter exposes an OpenAI-compatible Chat Completions API at
`https://openrouter.ai/api/v1`, so this reuses the exact OpenAI flow:
ffmpeg-extract low-detail JPEG frames (OpenRouter does NOT accept native
video), send each as a base64 image_url part. Model ids are namespaced
slugs, e.g. `qwen/qwen3-vl-32b-instruct`.

Structured output is best-effort: many open models don't honour
`response_format={json_schema, strict}`, so we try it and, on failure, fall
back to JSON-object mode + a "return ONLY JSON" instruction, then strip any
code fences before the shared `parse_and_validate`. The schema shape is
unchanged, so the result is the same `ProviderResult` every other provider
returns. (Mirrors nemara-vlm-benchmark/scripts/benchmark/provider_openrouter.)
"""

import asyncio
import logging
from collections.abc import Sequence

from openai import AsyncOpenAI

from src.services.detection.frame_annotate import annotate_base64_async
from src.services.detection.frame_extraction import ExtractedFrame, extract_frames
from src.services.detection.prompt import (
    DetectorSpec,
    build_identity_addendum,
    build_prompt_and_schema,
)
from src.services.detection.providers._parse import parse_and_validate
from src.services.detection.providers.types import ProviderResult
from src.settings import Settings

log = logging.getLogger(__name__)


def _strip_fences(raw: str) -> str:
    """Strip ```json … ``` / ``` … ``` fences some open models emit, so the
    shared parser's bare `json.loads` succeeds."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


async def _maybe_annotate(
    data_b64: str,
    bbox: tuple[float, float, float, float] | None,
) -> str:
    if bbox is None:
        return data_b64
    return await annotate_base64_async(data_b64, bbox)


async def openrouter_provider(
    *,
    mp4: bytes,
    detectors: Sequence[DetectorSpec],
    log_tag: str,
    settings: Settings,
    patient_in_clip: bool = False,
    patient_bbox: tuple[float, float, float, float] | None = None,
    pre_extracted_frames: list[ExtractedFrame] | None = None,
    per_frame_bboxes: list[tuple[float, float, float, float] | None] | None = None,
    sample_fps: int | None = None,
) -> ProviderResult:
    if not settings.openrouter_api_key:
        raise RuntimeError("NEMARA_OPENROUTER_API_KEY not configured")

    built = build_prompt_and_schema(detectors)
    identity_addendum = build_identity_addendum(patient_in_clip)

    # Reuse Re-ID's already-decoded frames + per-frame boxes when present
    # (same fps/width the frame path needs); else extract fresh. Identical
    # to the OpenAI provider so the model literally sees the same frames.
    if pre_extracted_frames is not None and sample_fps is not None:
        frames = pre_extracted_frames
        fps = sample_fps
    else:
        fps = settings.openai_frames_fps
        frames = await extract_frames(mp4, fps=fps, max_width=480)

    frame_payloads: list[tuple[str, str]] = [
        (f.data_base64, f.mime_type) for f in frames
    ]
    if per_frame_bboxes is not None and len(per_frame_bboxes) == len(frames):
        annotated = await asyncio.gather(
            *(
                _maybe_annotate(b64, bbox)
                for (b64, _mime), bbox in zip(frame_payloads, per_frame_bboxes)
            ),
        )
        frame_payloads = [(b64, mime) for b64, (_, mime) in zip(annotated, frame_payloads)]
    elif patient_in_clip and patient_bbox is not None:
        annotated = await asyncio.gather(
            *(annotate_base64_async(b64, patient_bbox) for b64, _mime in frame_payloads),
        )
        frame_payloads = [(b64, mime) for b64, (_, mime) in zip(annotated, frame_payloads)]

    image_parts = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "low"},
        }
        for b64, mime in frame_payloads
    ]
    text = (
        f"{built.prompt}\n\n"
        f"The following {len(frames)} images are sequential video frames sampled "
        f"at {fps} fps from a 15-second clip (frame 1 is earliest, frame "
        f"{len(frames)} is most recent). Reason over them as a temporal sequence."
        f"{identity_addendum}"
    )
    messages = [
        {"role": "user", "content": [{"type": "text", "text": text}, *image_parts]}
    ]

    client = AsyncOpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers={
            "HTTP-Referer": "https://github.com/nemara/nemara-backend",
            "X-Title": "nemara-backend",
        },
    )
    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "monitoring_analysis",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": built.schema_required,
                "properties": built.schema_properties,
            },
        },
    }

    # Try strict structured output; fall back to JSON-object mode + an explicit
    # "return only JSON" instruction for models that reject json_schema.
    try:
        completion = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=messages,
            response_format=json_schema,  # type: ignore[arg-type]
        )
        raw = _strip_fences(completion.choices[0].message.content or "")
        if not raw:
            raise RuntimeError("empty response under json_schema")
    except Exception as schema_err:  # noqa: BLE001
        log.info(
            "%s openrouter json_schema unsupported (%s) — retrying json_object",
            log_tag, schema_err,
        )
        messages_json = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text + "\n\nReturn ONLY a single JSON object "
                        "matching the requested fields. No prose, no code fences.",
                    },
                    *image_parts,
                ],
            }
        ]
        completion = await client.chat.completions.create(
            model=settings.openrouter_model,
            messages=messages_json,
            response_format={"type": "json_object"},  # type: ignore[arg-type]
        )
        raw = _strip_fences(completion.choices[0].message.content or "")
        if not raw:
            raise RuntimeError("openrouter returned empty response")

    return parse_and_validate(
        raw,
        provider="openrouter",
        required=built.schema_required,
        properties=built.schema_properties,
    )
