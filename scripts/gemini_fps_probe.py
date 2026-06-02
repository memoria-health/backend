"""Probe gemini-3.1-flash-lite over a 15s video at fps={1,3,5}.

Mirrors the production detector call shape (BASE_PROMPT + always-on
people_present + adult preset detectors + speech bonus), reports
per-run latency, input/output/thinking tokens, and standard-tier
cost in USD. Pricing constants pulled from
https://ai.google.dev/gemini-api/docs/pricing on 2026-06-01:
  input  $0.25 per 1M tokens (text / image / video, standard tier)
  output $1.50 per 1M tokens
Override via --input-price / --output-price if tier or rates change.

Run from anywhere; .env is loaded relative to this file's parent.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types as genai_types

from src.services.detection.prompt import (
    DetectorSpec,
    build_identity_addendum,
    build_prompt_and_schema,
)
from src.services.onboarding import _PRESET_DETECTORS_ADULT
from src.settings import Settings

_SPEECH_PROMPT_FRAGMENT = (
    "speech_heard: boolean. True if any human speech / voices are "
    "audible (any volume, any language, including muffled or partial "
    "words). False if silence, background noise only, or non-speech "
    "sounds.\n"
    "speech_transcript: string. If speech_heard is true, transcribe "
    "the speech verbatim (best effort, up to ~200 chars). Multiple "
    "speakers separated by ' / '. If speech_heard is false, empty "
    "string."
)
_SPEECH_FIELDS = {
    "speech_heard":      {"type": "boolean"},
    "speech_transcript": {"type": "string"},
}


def _map_type(t: str) -> genai_types.Type:
    if t == "boolean": return genai_types.Type.BOOLEAN
    if t == "number":  return genai_types.Type.NUMBER
    if t == "string":  return genai_types.Type.STRING
    raise ValueError(f"unknown schema type: {t}")


def _adult_specs() -> list[DetectorSpec]:
    return [
        DetectorSpec(
            identifier=p["identifier"],
            name=p["name"],
            prompt_fragment=p["prompt_fragment"],
            schema_fields=p["schema_json"],
        )
        for p in _PRESET_DETECTORS_ADULT
    ]


@dataclass
class RunResult:
    fps: int | None
    ok: bool
    latency_s: float
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    total_tokens: int
    cost_usd: float
    description: str
    error: str | None


def _build_response_schema(
    properties: dict[str, dict[str, str]],
    required: list[str],
) -> genai_types.Schema:
    return genai_types.Schema(
        type=genai_types.Type.OBJECT,
        required=required,
        properties={
            name: genai_types.Schema(type=_map_type(field["type"]))
            for name, field in properties.items()
        },
    )


async def _run_once(
    *,
    client: genai.Client,
    model: str,
    mp4: bytes,
    full_prompt: str,
    response_schema: genai_types.Schema,
    fps: int | None,
    input_price: float,
    output_price: float,
) -> RunResult:
    video_metadata = genai_types.VideoMetadata(fps=fps) if fps is not None else None
    parts: list[genai_types.Part] = [
        genai_types.Part(
            inline_data=genai_types.Blob(data=mp4, mime_type="video/mp4"),
            video_metadata=video_metadata,
        ),
        genai_types.Part(text=full_prompt),
    ]
    t0 = time.monotonic()
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=[genai_types.Content(role="user", parts=parts)],
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
            ),
        )
    except Exception as e:
        return RunResult(
            fps=fps, ok=False,
            latency_s=time.monotonic() - t0,
            input_tokens=0, output_tokens=0, thinking_tokens=0,
            total_tokens=0, cost_usd=0.0,
            description="", error=f"{type(e).__name__}: {e}",
        )
    latency = time.monotonic() - t0
    um = response.usage_metadata
    input_tokens = (um.prompt_token_count or 0) if um else 0
    output_tokens = (um.candidates_token_count or 0) if um else 0
    thinking_tokens = (
        (um.thoughts_token_count or 0)
        if um and hasattr(um, "thoughts_token_count") else 0
    )
    total_tokens = (um.total_token_count or 0) if um else 0
    cost = (
        input_tokens * input_price
        + (output_tokens + thinking_tokens) * output_price
    ) / 1_000_000.0
    raw = (response.text or "").strip()
    description = ""
    if raw:
        import json
        try:
            description = (json.loads(raw).get("description") or "")[:200]
        except json.JSONDecodeError:
            description = raw[:200]
    return RunResult(
        fps=fps, ok=True,
        latency_s=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        total_tokens=total_tokens,
        cost_usd=cost,
        description=description, error=None,
    )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip", default=str(Path(__file__).parent / "fixtures" / "clip_15s.mp4"),
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument(
        "--fps", default="1,3,5",
        help="comma-separated ints; include 'none' for default sampling",
    )
    parser.add_argument("--input-price", type=float, default=0.25)
    parser.add_argument("--output-price", type=float, default=1.50)
    args = parser.parse_args()

    fps_values: list[int | None] = []
    for token in args.fps.split(","):
        token = token.strip().lower()
        if token in ("", "none", "default"):
            fps_values.append(None)
        else:
            fps_values.append(int(token))

    env_file = Path(__file__).resolve().parent.parent / ".env"
    settings = Settings(_env_file=str(env_file))  # type: ignore[call-arg]
    if not settings.gemini_api_key:
        print("ERROR: NEMARA_GEMINI_API_KEY missing from .env")
        return 1

    clip_path = Path(args.clip).resolve()
    if not clip_path.exists():
        print(f"ERROR: clip not found at {clip_path}")
        return 1
    mp4 = clip_path.read_bytes()

    specs = _adult_specs()
    built = build_prompt_and_schema(specs)
    full_prompt = (
        f"{built.prompt}\n\n{_SPEECH_PROMPT_FRAGMENT}"
        f"{build_identity_addendum(patient_in_clip=False)}"
    )
    properties = {**built.schema_properties, **_SPEECH_FIELDS}
    required = built.schema_required + ["speech_heard", "speech_transcript"]
    response_schema = _build_response_schema(properties, required)

    client = genai.Client(api_key=settings.gemini_api_key)

    print(f"model       : {args.model}")
    print(f"clip        : {clip_path} ({len(mp4) / 1024:.1f} KiB)")
    print(f"detectors   : {[s.identifier for s in specs]}")
    print(f"prompt chars: {len(full_prompt)}")
    print(f"fps sweep   : {fps_values}")
    print(f"pricing $/M : input={args.input_price} output={args.output_price}")
    print()

    results: list[RunResult] = []
    for fps in fps_values:
        label = "default" if fps is None else f"fps={fps}"
        print(f"[run] {label} ... ", end="", flush=True)
        r = await _run_once(
            client=client,
            model=args.model,
            mp4=mp4,
            full_prompt=full_prompt,
            response_schema=response_schema,
            fps=fps,
            input_price=args.input_price,
            output_price=args.output_price,
        )
        status = "ok" if r.ok else f"FAIL ({r.error})"
        print(f"{r.latency_s:.2f}s  in={r.input_tokens}  out={r.output_tokens}  "
              f"think={r.thinking_tokens}  ${r.cost_usd:.6f}  {status}")
        results.append(r)

    print()
    hdr = "fps      | ok  | latency |    input |   output | thinking |    total |     cost USD"
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    for r in results:
        fps_lbl = "default" if r.fps is None else str(r.fps)
        ok = "yes" if r.ok else "no"
        print(
            f"{fps_lbl:<8} | {ok:<3} | {r.latency_s:>6.2f}s | "
            f"{r.input_tokens:>8} | {r.output_tokens:>8} | "
            f"{r.thinking_tokens:>8} | {r.total_tokens:>8} | "
            f"${r.cost_usd:>10.6f}"
        )
    print()
    for r in results:
        fps_lbl = "default" if r.fps is None else f"fps={r.fps}"
        if r.ok:
            print(f"[{fps_lbl}] description: {r.description!r}")
        else:
            print(f"[{fps_lbl}] error: {r.error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
