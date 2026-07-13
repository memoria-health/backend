"""Forensic probe for the all-zero rapid run: one R8 dual call per key clip,
printing EVERY schema field (description, motion_*, both detectors' verdict +
confidence + reasoning) so we can tell perception failure (wrong/blank
description) from decision failure (right description, FALSE verdict).

Usage:  PYTHONPATH=. uv run python experiments/rapid_debug_probe.py
"""

import asyncio
from pathlib import Path

from experiments.rapid_motion_test import _WORK, _dual
from src.settings import get_settings

CLIPS = ["conv_1", "conv_2", "shiv_1", "shiv_2", "neg_4"]


async def main() -> None:
    settings = get_settings().model_copy(update={"motion_overlay_enabled": False})
    for stem in CLIPS:
        sbs = Path(_WORK / f"{stem}_R8_sbs0.mp4").read_bytes()
        r = await _dual(sbs, settings)
        print(f"\n════ {stem} (R8 dual) ════")
        f = dict(r.detector_fields)
        desc = f.pop("description", None) or getattr(r, "description", "")
        print(f"  description: {desc}")
        for k in ("motion_region", "motion_spread", "motion_rhythm",
                  "motion_amplitude", "motion_voluntary"):
            print(f"  {k}: {f.get(k)}")
        for d in ("convulsion", "body_shaking"):
            print(f"  {d}: detected={f.get(f'{d}_detected')} "
                  f"conf={f.get(f'{d}_confidence')}")
            print(f"    reasoning: {f.get(f'{d}_reasoning')}")


if __name__ == "__main__":
    asyncio.run(main())
