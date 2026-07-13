"""DILUTION CURVE — how many detectors can share one prompt?

Crop-clean inputs held CONSTANT (prod tracker, per-person fan-out); the only
variable is how many detectors ride the prompt. Nested sets so each step adds
detectors to the previous:

  1det: collar_tug
  3det: collar_tug + medication + fall
  5det: collar_tug + medication + fall + agitation + choking

Target metric = collar_tug_detected rate (all clips are confirmed TPs).
Reuses collar_recall_test's tracking/crop/call machinery.

Usage:  PYTHONPATH=. uv run python experiments/dilution_curve_test.py [N]
"""

import asyncio
import sys

from experiments.collar_recall_test import (
    _D, _WORK, EVENTS, N, _build, _vid,
)
from experiments.custom_detectors import COLLAR_TUG
from src.services.detection.catalog import by_key
from src.services.detection.prompt import DetectorSpec
from src.settings import get_settings


def _cat(keys):
    return [DetectorSpec(identifier=e.key, name=e.name,
                         prompt_fragment=e.prompt_fragment,
                         schema_fields=e.schema_json, always_on=False)
            for e in (by_key(k) for k in keys)]


SETS = {
    "1det": [COLLAR_TUG],
    "3det": [COLLAR_TUG] + _cat(["medication", "fall"]),
    "5det": [COLLAR_TUG] + _cat(["medication", "fall", "agitation", "choking"]),
}


async def main() -> None:
    _WORK.mkdir(parents=True, exist_ok=True)
    s_off = get_settings().model_copy(update={"motion_overlay_enabled": False})
    print(f"DILUTION CURVE  crop-clean, prod tracking, N={N}, "
          f"clips={[f[0][:-4] for _, f in EVENTS]}\n")

    totals = {k: 0 for k in SETS}
    grand = 0
    for ev_name, files in EVENTS:
        built = await _build(_D / files[0])
        crops = built["crop_clean"]
        grand += N

        async def any_person(specs):
            res = await asyncio.gather(*(_vid(c, specs, s_off) for c in crops))
            vals = [v for v in res if v is not None]
            return any(vals) if vals else None

        async def run_set(specs):
            runs = await asyncio.gather(*(any_person(specs) for _ in range(N)))
            return sum(1 for r in runs if r)

        results = await asyncio.gather(*(run_set(s) for s in SETS.values()))
        row = "  ".join(f"{k}={r}/{N}" for k, r in zip(SETS, results))
        print(f"  {ev_name} ({files[0][:-4]}, persons={built['npersons']}):  {row}")
        for k, r in zip(SETS, results):
            totals[k] += r

    print("\n═══ DILUTION CURVE (crop-clean recall vs detectors-per-prompt) ═══")
    for k in SETS:
        print(f"  {k}: {totals[k]}/{grand}  ({round(100 * totals[k] / grand)}%)")


if __name__ == "__main__":
    asyncio.run(main())
