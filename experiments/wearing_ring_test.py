"""wearing_ring — RECALL + PRECISION in one experiment, 6 arms, prod tracking.

First gesture test with deliberate FP-test takes (confuser gestures acted by
the same person), so every arm gets both sides:
  - pos clips (3 events, 4 clips): fire = recall            (higher better)
  - neg clips (3 FP-test takes):   fire = false positive    (lower better)

Arms (same 6 as collar, reusing its preprocessing + prod tracker):
  A native+ovl solo · D crop-clean solo · B crop+ovl solo · C stills solo
  E native+ovl multi5 · F crop-clean multi5

Usage:  PYTHONPATH=. uv run python experiments/wearing_ring_test.py [N]
"""

import asyncio
import sys
from pathlib import Path

# Reuse the collar harness machinery (prod tracking, crops, overlay, stills,
# retry). Its module-level EVENTS/N are unused here except N (same argv).
from experiments.collar_recall_test import (
    ARMS, N, _build, _gemini_stills, _retry,
)
from experiments.custom_detectors import WEARING_RING
from src.services.detection.catalog import by_key
from src.services.detection.prompt import DetectorSpec
from src.services.detection.providers.gemini import gemini_provider
from src.settings import get_settings

_D = Path("/Users/aryamandarda/Desktop/eval-set/wearing_ring")
TARGET_KEY = "wearing_ring_detected"

# RULE: 1 representative clip per event (events with >1 clip pick one).
POS_EVENTS = [
    ("E1_living",  ["pos_1_a_living.mp4"]),
    ("E2_living",  ["pos_2_a_living.mp4"]),
    ("E5_bedroom", ["pos_3_a_bedroom.mp4"]),
]
NEG_CLIPS = ["neg_1_living.mp4", "neg_2_living.mp4", "neg_3_bedroom.mp4"]


def _specs_solo():
    return [WEARING_RING]


def _specs_multi():
    cat = [DetectorSpec(identifier=e.key, name=e.name,
                        prompt_fragment=e.prompt_fragment,
                        schema_fields=e.schema_json, always_on=False)
           for e in (by_key(k) for k in
                     ("medication", "fall", "agitation", "choking"))]
    return [WEARING_RING] + cat


async def _vid(mp4: bytes, specs, settings):
    r = await _retry(lambda: gemini_provider(
        mp4=mp4, detectors=specs, log_tag="ring", settings=settings,
        patient_in_clip=False))
    return None if r is None else bool(r.detector_fields.get(TARGET_KEY))


async def _still(images, specs, settings):
    r = await _retry(lambda: _gemini_stills(images, specs, settings))
    return None if r is None else bool(r.detector_fields.get(TARGET_KEY))


async def _run_clip(built, s_on, s_off):
    solo, multi = _specs_solo(), _specs_multi()

    async def any_person(clips, specs, settings):
        res = await asyncio.gather(*(_vid(c, specs, settings) for c in clips))
        vals = [v for v in res if v is not None]
        return any(vals) if vals else None

    async def one_run():
        a, d, b, c, e, f = await asyncio.gather(
            _vid(built["natovl"], solo, s_on),
            any_person(built["crop_clean"], solo, s_off),
            any_person(built["crop_ovl"], solo, s_on),
            _still(built["stills"], solo, s_on),
            _vid(built["natovl"], multi, s_on),
            any_person(built["crop_clean"], multi, s_off),
        )
        return {"A": a, "D": d, "B": b, "C": c, "E": e, "F": f}

    runs = await asyncio.gather(*(one_run() for _ in range(N)))
    return {arm: [r[arm] for r in runs] for arm in ARMS}


async def main() -> None:
    s_on = get_settings().model_copy(update={"motion_overlay_enabled": True})
    s_off = get_settings().model_copy(update={"motion_overlay_enabled": False})
    print(f"target=wearing_ring  model={s_on.gemini_model}  N={N}  "
          f"pos_events={len(POS_EVENTS)}  neg_clips={len(NEG_CLIPS)}\n")

    labels = {"A": "A native+ovl (solo)",   "D": "D crop-clean (solo)",
              "B": "B crop+ovl   (solo)",   "C": "C stills     (solo)",
              "E": "E native+ovl (multi5)", "F": "F crop-clean (multi5)"}

    # ── RECALL on positives ──
    ev_recall = {a: [] for a in ARMS}
    for ev_name, files in POS_EVENTS:
        print(f"── {ev_name} ({len(files)} clip(s)) — TP ──")
        per_clip = {}
        for fn in files:
            built = await _build(_D / fn)
            per_clip[fn] = await _run_clip(built, s_on, s_off)
            rate = {a: sum(1 for x in per_clip[fn][a] if x) for a in ARMS}
            print(f"  {fn[:-4]} (persons={built['npersons']}):  "
                  + "  ".join(f"{a}={rate[a]}/{N}" for a in ARMS))
        for a in ARMS:
            succ = sum(1 for i in range(N)
                       if any(bool(per_clip[fn][a][i]) for fn in files))
            ev_recall[a].append(succ)
        print()

    # ── PRECISION on FP-test negatives ──
    neg_fires = {a: 0 for a in ARMS}
    for fn in NEG_CLIPS:
        built = await _build(_D / fn)
        res = await _run_clip(built, s_on, s_off)
        rate = {a: sum(1 for x in res[a] if x) for a in ARMS}
        print(f"── {fn[:-4]} — FP TEST ──  "
              + "  ".join(f"{a}={rate[a]}/{N}" for a in ARMS))
        for a in ARMS:
            neg_fires[a] += rate[a]
    print()

    print("═══ SUMMARY ═══")
    print(f"{'arm':<26}{'event recall':<28}{'FP rate (neg clips)'}")
    for a in ARMS:
        per = " ".join(f"E{i + 1}:{x}/{N}" for i, x in enumerate(ev_recall[a]))
        tot, den = sum(ev_recall[a]), len(POS_EVENTS) * N
        fp_den = len(NEG_CLIPS) * N
        print(f"{labels[a]:<26}{per} = {tot}/{den:<6}{neg_fires[a]}/{fp_den}")


if __name__ == "__main__":
    asyncio.run(main())
