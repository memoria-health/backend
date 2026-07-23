"""OpenAI fallback provider.

Verbatim port of `nemara-temp-dashboard/lib/providers/openai.ts`. Uses
ffmpeg-extracted frames (4fps default, env `OPENAI_FRAMES_FPS`), sends each
frame as a base64 image_url part with `detail: 'low'`, requests
`response_format={type: 'json_schema', strict: true}`.

The OpenAI path does NOT include the Gemini-only speech fields —
no audio surface.
"""

from collections.abc import Sequence

from openai import AsyncOpenAI

from src.services.detection.frame_extraction import ExtractedFrame, extract_frames
from src.services.detection.prompt import (
    DetectorSpec,
    build_identity_addendum,
    build_prompt_and_schema,
)
from src.services.detection.providers._parse import parse_and_validate
from src.services.detection.providers.types import ProviderResult
from src.settings import Settings


async def openai_provider(
    *,
    mp4: bytes,
    detectors: Sequence[DetectorSpec],
    log_tag: str,
    settings: Settings,
    patient_in_clip: bool = False,
    pre_extracted_frames: list[ExtractedFrame] | None = None,
    sample_fps: int | None = None,
) -> ProviderResult:
    if not settings.openai_api_key:
        raise RuntimeError("NEMARA_OPENAI_API_KEY not configured")

    built = build_prompt_and_schema(detectors)
    identity_addendum = build_identity_addendum(patient_in_clip)

    # Re-ID already decoded the clip into frames at the same fps/width
    # the OpenAI path needs. Reusing them saves a second ffmpeg pass
    # AND keeps the bboxes index-aligned with what the LLM sees. Falls
    # back to a fresh extract when Re-ID didn't run (disabled, no
    # enrollment, timed out, or errored).
    if pre_extracted_frames is not None and sample_fps is not None:
        frames = pre_extracted_frames
        fps = sample_fps
    else:
        fps = settings.openai_frames_fps
        frames = await extract_frames(mp4, fps=fps, max_width=480)

    frame_payloads: list[tuple[str, str]] = [
        (f.data_base64, f.mime_type) for f in frames
    ]

    image_parts = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime};base64,{b64}",
                "detail": "low",
            },
        }
        for b64, mime in frame_payloads
    ]

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    completion = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{built.prompt}\n\n"
                            f"The following {len(frames)} images are sequential "
                            f"video frames sampled at {fps} fps from a 15-second "
                            f"clip (frame 1 is earliest, frame {len(frames)} is "
                            f"most recent). Reason over them as a temporal sequence."
                            f"{identity_addendum}"
                        ),
                    },
                    *image_parts,
                ],
            }
        ],
        response_format={
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
        },
    )

    raw = (completion.choices[0].message.content or "").strip()
    if not raw:
        raise RuntimeError("openai returned empty response")
    return parse_and_validate(
        raw,
        provider="openai",
        required=built.schema_required,
        properties=built.schema_properties,
    )
