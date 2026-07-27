"""DashScope (Alibaba Model Studio) provider — Qwen3-VL via Alibaba's
NATIVE video path.

Unlike OpenRouter (which only accepts a flat list of independent images),
Alibaba's OpenAI-compatible endpoint accepts a real `video` content part —
a list of frames PLUS a sibling `fps` field — so Qwen3-VL applies its native
temporal/video understanding instead of treating frames as unrelated images.
We extract FRESH frames at the configured resolution/fps (default 1080p @
4fps; deliberately NOT the low-res ReID frames) and send them as one `video`
part.

We POST raw JSON with httpx rather than the OpenAI SDK so the non-standard
`{"type":"video","video":[...],"fps":N}` content part is sent verbatim
(the SDK's typed content params can drop unknown fields). Structured output
uses json_object mode + an explicit "return ONLY JSON" instruction; the
shared `parse_and_validate` enforces the schema regardless. Any error (HTTP
or body-level) raises, so `run_vlm` falls back to OpenRouter.
"""

import logging
from collections.abc import Sequence

import httpx

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
    """Strip ```json … ``` fences some models emit so `json.loads` succeeds."""
    s = (raw or "").strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


async def dashscope_provider(
    *,
    mp4: bytes,
    detectors: Sequence[DetectorSpec],
    log_tag: str,
    settings: Settings,
    patient_in_clip: bool = False,
    pre_extracted_frames: list[ExtractedFrame] | None = None,
    sample_fps: int | None = None,
) -> ProviderResult:
    if not settings.dashscope_api_key:
        raise RuntimeError("NEMARA_DASHSCOPE_API_KEY not configured")

    built = build_prompt_and_schema(detectors)
    identity_addendum = build_identity_addendum(patient_in_clip)

    # Fresh frames at the DashScope resolution/fps (default 1080p @ 4fps) —
    # the whole point of the direct path is higher specificity + native video,
    # so we re-sample rather than reuse the low-res ReID frames.
    fps = settings.dashscope_frames_fps
    frames = await extract_frames(
        mp4,
        fps=fps,
        max_width=settings.dashscope_frame_max_width,
        jpeg_quality=settings.dashscope_jpeg_quality,
    )
    # Trim to the frame budget (default 10fps × 11s = 110), keeping the MOST
    # RECENT frames — the tail of the clip is what just happened.
    if len(frames) > settings.dashscope_max_frames:
        frames = list(frames[-settings.dashscope_max_frames:])

    frame_b64 = [f.data_base64 for f in frames]

    video_frames = [f"data:image/jpeg;base64,{b64}" for b64 in frame_b64]
    text = (
        f"{built.prompt}\n\n"
        f"The video above is {len(frames)} sequential frames sampled at {fps} "
        f"fps (earliest first). Reason over it as a temporal video."
        f"{identity_addendum}\n\n"
        f"Return ONLY a single JSON object matching the requested fields. "
        f"No prose, no code fences."
    )
    payload = {
        "model": settings.dashscope_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames, "fps": fps},
                    {"type": "text", "text": text},
                ],
            }
        ],
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0)) as client:
        resp = await client.post(
            f"{settings.dashscope_base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"dashscope HTTP {resp.status_code}: {resp.text[:300]}"
        )
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"dashscope error: {data['error']}")

    raw = _strip_fences(data["choices"][0]["message"]["content"] or "")
    if not raw:
        raise RuntimeError("dashscope returned empty response")

    return parse_and_validate(
        raw,
        provider="dashscope",
        required=built.schema_required,
        properties=built.schema_properties,
    )
