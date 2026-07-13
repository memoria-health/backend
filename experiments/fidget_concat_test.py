"""Fidgeting T8 arm — temporal concat (overlay segment first, then clean), 8fps
flash. Reuses the crop artifacts already built by the clean run (fid_*_R8_cc0)
and the overlay run (fidov_*_co0).

Hypothesis (Aryaman): tint locates the motion, then the clean segment supplies
purpose/context — for a detector whose discriminator IS purpose, concat may
beat both single channels. (For convulsion concat LOST to overlay-only 7/10 vs
10/10 — but convulsion never needed clean context; fidgeting does.)

Usage:  PYTHONPATH=. uv run python experiments/fidget_concat_test.py [N]
"""

import asyncio
import sys
from pathlib import Path

from experiments.collar_recall_test import _retry
from experiments.fidget_test import GENERAL_FIDGETING, KEY, _D
from experiments.rapid_concat_test import MD_CONCAT, _concat, _concat_note, _dur_s
from experiments.rapid_motion_test import _WORK, _call
from src.settings import get_settings

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
SPECS = [MD_CONCAT, GENERAL_FIDGETING]


async def main() -> None:
    s = get_settings().model_copy(update={"gemini_model": "gemini-3-flash-preview",
                                          "motion_overlay_enabled": False})
    clips = sorted(_D.glob("*.mp4"))
    fires = {"fid": 0, "neg": 0}
    denom = {"fid": 0, "neg": 0}
    for clip in clips:
        stem = clip.stem
        truth = "fid" if stem.startswith("fid") else "neg"
        # concat per person: overlay crop (fidov_ run) + clean crop (fid_ run)
        vids, notes = [], []
        pi = 0
        while True:
            co = _WORK / f"fidov_{stem}_co{pi}.mp4"
            cc = _WORK / f"fid_{stem}_R8_cc{pi}.mp4"
            if not (co.exists() and cc.exists()):
                break
            cat = str(_WORK / f"fidcat_{stem}_{pi}.mp4")
            _concat(str(co), str(cc), cat)          # overlay FIRST, then clean
            vids.append(Path(cat).read_bytes())
            notes.append(_concat_note(_dur_s(str(cc))))
            pi += 1
        if not vids:
            print(f"  {stem}: missing artifacts, skipped")
            continue
        denom[truth] += N

        async def one_sample():
            res = await asyncio.gather(*(
                _retry(lambda v=v, n=n: _call(v, SPECS, n, s))
                for v, n in zip(vids, notes)))
            fs = [r.detector_fields for r in res if r is not None]
            if not fs:
                return None, {}
            return any(bool(f.get(KEY)) for f in fs), fs[0]

        samples = await asyncio.gather(*(one_sample() for _ in range(N)))
        n_fire = sum(1 for v, _ in samples if v)
        fires[truth] += n_fire
        why = str(samples[0][1].get("general_fidgeting_reasoning", ""))[:100] \
            if samples and samples[0][1] else ""
        print(f"  {stem} [{truth}] T8: {n_fire}/{N}  e.g. {why}")

    rec = fires["fid"] / denom["fid"] if denom["fid"] else 0.0
    fp = fires["neg"] / denom["neg"] if denom["neg"] else 0.0
    print(f"\n═══ T8 concat (overlay→clean): recall={rec:.0%}  fp={fp:.0%}  "
          f"discrimination={rec - fp:+.0%} ═══")


if __name__ == "__main__":
    asyncio.run(main())
