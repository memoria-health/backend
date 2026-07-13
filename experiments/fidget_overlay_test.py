"""Fidgeting O8 arm — overlay-only person crop, 8fps, flash (the RAPID recipe),
missing from the clean-only fidget run. Prompt carries the overlay note.

Usage:  PYTHONPATH=. uv run python experiments/fidget_overlay_test.py [N]
"""

import asyncio
import sys
from pathlib import Path

from experiments.collar_recall_test import _apply_crop, _apply_overlay, _retry
from experiments.fidget_test import GENERAL_FIDGETING, KEY, _D
from experiments.rapid_concat_test import MD_OVERLAY, OVERLAY_NOTE
from experiments.rapid_motion_test import _WORK, _call, _track_persons_fps
from src.settings import get_settings

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SPECS = [MD_OVERLAY, GENERAL_FIDGETING]


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    s = get_settings().model_copy(update={"gemini_model": "gemini-3-flash-preview",
                                          "motion_overlay_enabled": False})
    clips = sorted(_D.glob("*.mp4"))
    fires = {"fid": 0, "neg": 0}
    denom = {"fid": 0, "neg": 0}
    for clip in clips:
        stem = clip.stem
        truth = "fid" if stem.startswith("fid") else "neg"
        denom[truth] += N
        natovl = str(_WORK / f"fidov_{stem}_natovl.mp4")
        _apply_overlay(str(clip), natovl)
        geoms = _track_persons_fps(str(clip), 8)
        vids = []
        for pi, g in enumerate(geoms):
            co = str(_WORK / f"fidov_{stem}_co{pi}.mp4")
            _apply_crop(natovl, co, g)
            vids.append(Path(co).read_bytes())

        async def one_sample():
            res = await asyncio.gather(*(
                _retry(lambda v=v: _call(v, SPECS, OVERLAY_NOTE, s))
                for v in vids))
            fs = [r.detector_fields for r in res if r is not None]
            if not fs:
                return None, {}
            return any(bool(f.get(KEY)) for f in fs), fs[0]

        samples = await asyncio.gather(*(one_sample() for _ in range(N)))
        n_fire = sum(1 for v, _ in samples if v)
        fires[truth] += n_fire
        why = str(samples[0][1].get("general_fidgeting_reasoning", ""))[:100] \
            if samples and samples[0][1] else ""
        print(f"  {stem} [{truth}] O8: {n_fire}/{N}  e.g. {why}")

    rec = fires["fid"] / denom["fid"] if denom["fid"] else 0.0
    fp = fires["neg"] / denom["neg"] if denom["neg"] else 0.0
    print(f"\n═══ O8 overlay-only: recall={rec:.0%}  fp={fp:.0%}  "
          f"discrimination={rec - fp:+.0%} ═══")


if __name__ == "__main__":
    asyncio.run(main())
