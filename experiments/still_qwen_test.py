"""Qwen (OpenRouter) still test — same two frames, same detector prompts.

Question: Qwen-VL uses NATIVE DYNAMIC RESOLUTION (tokens scale with input
pixels, no forced downscale). Does the full 1080p frame — no zone crop —
recover what Gemini's fixed ~1.1k-token encoder lost (plate at distance,
phone-vs-remote identity)? Usage tokens are printed to verify native-res
(~2.5-2.7k expected for 1080p vs Gemini's 1.1k).

Usage:
  NEMARA_OPENROUTER_API_KEY=sk-or-v1-... PYTHONPATH=. uv run python \
      experiments/still_qwen_test.py [--n 10] [--model qwen/qwen3-vl-32b-instruct]
"""

import asyncio
import base64
import json
import os
import sys
from pathlib import Path

import httpx

sys.argv = sys.argv[:1]
from experiments.still_object_test import SPECS, KEYS
from src.services.detection.prompt import build_prompt_and_schema

D = Path.home() / "Downloads/memoria-crop-experiment/still_raw"
CLIPS = [("030939 (TP: phone+plate)", "030939_0c4eb1f8-2c90-4117-a700-97ba889dbecb.still.jpg"),
         ("031039 (FP: remote)", "031039_0c4eb1f8-2c90-4117-a700-97ba889dbecb.still.jpg")]

MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen3-vl-32b-instruct")
N = int(os.environ.get("QWEN_N", "10"))
KEY = os.environ.get("NEMARA_OPENROUTER_API_KEY", "")


async def one(client: httpx.AsyncClient, prompt: str, b64: str):
    r = await client.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEY}"},
        json={
            "model": MODEL,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": prompt +
                 "\n\nRespond with ONLY a JSON object containing every field above."},
            ]}],
        },
        timeout=120)
    r.raise_for_status()
    body = r.json()
    txt = body["choices"][0]["message"]["content"].strip()
    if txt.startswith("```"):
        txt = txt.strip("`").removeprefix("json").strip()
    usage = body.get("usage", {})
    return json.loads(txt), usage.get("prompt_tokens")


async def main() -> None:
    if not KEY:
        sys.exit("set NEMARA_OPENROUTER_API_KEY")
    built = build_prompt_and_schema(SPECS, motion_overlay=False)
    async with httpx.AsyncClient() as client:
        for label, fn in CLIPS:
            b64 = base64.b64encode((D / fn).read_bytes()).decode()
            res = []
            for _ in range(N):
                try:
                    res.append(await one(client, built.prompt, b64))
                except Exception as ex:  # noqa: BLE001
                    print(f"  !! {type(ex).__name__}: {str(ex)[:80]}")
            fields = [f for f, _ in res]
            toks = [t for _, t in res if t]
            print(f"\n════ {label} — {MODEL} (full frame, native res) ════")
            print(f"  prompt tokens: {toks[0] if toks else '?'} "
                  f"(Gemini high was ~1,108)")
            for k in KEYS:
                fires = sum(1 for f in fields if bool(f.get(k)))
                print(f"  {k.replace('_detected',''):<16} {fires}/{len(fields)}")
            if fields:
                print(f"  e.g. phone: {str(fields[0].get('phone_on_couch_reasoning'))[:110]}")
                print(f"  e.g. meal:  {str(fields[0].get('meal_plate_reasoning'))[:110]}")


if __name__ == "__main__":
    asyncio.run(main())
