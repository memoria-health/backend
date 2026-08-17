"""Custom-detector compiler eval vs the built-in taxonomy.

Measures the LLM-backed custom-create path — `infer_detector_config`
(LLM: trigger / sustained / category / severity) followed by
`compiler_v2.classify_custom_dims` (pure rules: subject / bucket /
motion / audio) — by asking it to build detectors we already have as
hand-perfected built-ins, phrased the way a real caregiver would type
them in a few words.

Ground truth is `catalog_v2.DIMS` + `catalog.CATALOG`. No DB is
touched and no detector rows are created; only the compile-path
functions run (one infer LLM call per phrase — Gemini with OpenAI
fallback, keys from .env via src.settings).

Run from the repo root:

    uv run python scripts/compiler_eval.py [--out scripts/compiler_eval_baseline.txt]

Scored fields (custom output vs built-in ground truth):
    subject_scope, bucket, requires_motion, needs_audio, category,
    temporal shape (momentary / sustained / scheduled), severity band
    (emergency >= 80, high 50-79, low < 50).
Cooldown is reported informationally: the custom path emits no
per-detector cooldown (rows keep the 30s global default), while the
env built-ins carry 1800s (action) / 600s (ongoing-emergency) spacing.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

from src.services.detection import catalog, catalog_v2
from src.services.detection.scheduling import compiler_v2
from src.services.detector_compiler import infer_detector_config
from src.settings import Settings

# ── Eval set: built-in identifier → short, casual caregiver phrasings ──
# Three per detector: casual/declarative forms plus a strict imperative
# "alert me when they …" form (the actual custom-create UX phrasing —
# Aryaman 2026-08-17), all worded differently so rule fixes tuned on one
# phrase are proven on the others. Skipped: sleep / activity /
# setup_test (record-only or internal — no caregiver would ask for them
# as alerts).
PHRASES: dict[str, tuple[str, ...]] = {
    "fall": (
        "alert me if my dad falls",
        "grandma fell down",
        "notify me when my mother falls over",
    ),
    "agitation": (
        "mom seems really agitated and restless",
        "tell me when dad gets worked up and paces around",
        "alert me when she gets really restless and upset",
    ),
    "wandering": (
        "dad wandered out the front door",
        "alert me if mom leaves the house",
        "let me know if he walks out of the apartment",
    ),
    "medication": (
        "mom didn't take her evening pills",
        "did grandpa take his meds",
        "tell me when mom takes her medicine",
    ),
    "calling_for_help": (
        "grandma is calling out for help",
        "alert me if mom yells for help",
        "notify me if grandpa shouts for help",
    ),
    "choking": (
        "grandpa is choking",
        "alert me if dad chokes on his food",
        "alert me if grandma starts choking during meals",
    ),
    "meal_eaten": (
        "tell me when mom eats her lunch",
        "make sure dad eats his dinner",
        "let me know when dad has eaten his breakfast",
    ),
    "stove_left_on": (
        "tell me if the stove gets left on",
        "the burner is still on and no one's in the kitchen",
        "alert me when the oven stays on unattended",
    ),
    "emergency": (
        "something is seriously wrong with dad",
        "alert me if grandma is unconscious or bleeding",
        "alert me if something bad happens to mom like she passes out",
    ),
    "possible_intruder": (
        "someone wearing a mask enters the house",
        "a stranger is in the house",
        "alert me when an unfamiliar person shows up inside the house",
    ),
    "env_hazard_glass": (
        "a glass broke on the kitchen floor",
        "alert me if something shatters and leaves shards",
        "alert me when they drop and break a glass",
    ),
    "env_hazard_water": (
        "water spilled on the kitchen floor",
        "someone knocked over a drink and it spilled",
        "alert me when they spill water on the floor",
    ),
    "fire_smoke": (
        "I see smoke in the living room",
        "alert me if there's a fire",
        "alert me if anything catches fire in the house",
    ),
}

FIELDS = (
    "subject_scope", "bucket", "requires_motion", "needs_audio",
    "category", "temporal", "severity_band",
)


def severity_band(severity: int) -> str:
    if severity >= 80:
        return "emergency"
    if severity >= 50:
        return "high"
    return "low"


def temporal_shape(alert_trigger: str, is_sustained: bool) -> str:
    """Coarse temporal class: does the detector latch a state
    (sustained/duration/trailing), fire on the moment (leading /
    record_only instance), or evaluate a schedule window (absence)?"""
    if alert_trigger == "absence":
        return "scheduled"
    if is_sustained or alert_trigger in ("trailing", "duration"):
        return "sustained"
    return "momentary"


@dataclass(frozen=True)
class GroundTruth:
    subject_scope: str
    bucket: str
    requires_motion: bool
    needs_audio: bool
    category: str
    temporal: str
    severity_band: str
    cooldown_seconds: int | None


def ground_truth(identifier: str) -> GroundTruth:
    dims = catalog_v2.DIMS[identifier]
    entry = catalog.by_key(identifier)
    assert entry is not None, identifier
    return GroundTruth(
        subject_scope=dims.subject_scope,
        bucket=dims.bucket,
        requires_motion=dims.requires_motion,
        needs_audio=dims.needs_audio,
        category=entry.category,
        temporal=temporal_shape(entry.alert_trigger, entry.is_sustained),
        severity_band=severity_band(dims.severity),
        cooldown_seconds=dims.cooldown_seconds,
    )


async def compile_phrase(
    settings: Settings, sem: asyncio.Semaphore, phrase: str,
) -> dict:
    """The real custom-create compile path, DB-free: infer (LLM) then
    dims classification (pure rules), exactly the pair
    `create_detector` + `_compile_and_update_detector` run."""
    async with sem:
        inferred = await infer_detector_config(
            name=None, prompt=phrase, settings=settings,
        )
    dims = compiler_v2.classify_custom_dims(
        # In the real flow the app pre-fills the name from the infer
        # suggestion and create passes it through to classification.
        name=inferred.suggested_name or "",
        user_text=phrase,
        is_sustained=inferred.is_sustained,
        category=inferred.category,
    )
    return {
        "subject_scope": dims["subject_scope"],
        "bucket": dims["bucket"],
        "requires_motion": dims["requires_motion"],
        "needs_audio": dims["needs_audio"],
        "category": inferred.category,
        "temporal": temporal_shape(inferred.alert_trigger, inferred.is_sustained),
        "severity_band": severity_band(inferred.severity),
        "severity": inferred.severity,
        "alert_trigger": inferred.alert_trigger,
        "suggested_name": inferred.suggested_name,
    }


async def run_eval(out_path: str | None) -> int:
    settings = Settings()
    sem = asyncio.Semaphore(4)
    cases = [
        (ident, phrase)
        for ident, phrases in PHRASES.items()
        for phrase in phrases
    ]
    results = await asyncio.gather(
        *(compile_phrase(settings, sem, phrase) for _, phrase in cases),
        return_exceptions=True,
    )

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    field_hits: dict[str, int] = dict.fromkeys(FIELDS, 0)
    field_total: dict[str, int] = dict.fromkeys(FIELDS, 0)
    errors = 0

    for (ident, phrase), got in zip(cases, results):
        gt = ground_truth(ident)
        emit(f"── {ident}: {phrase!r}")
        if isinstance(got, BaseException):
            errors += 1
            emit(f"   COMPILE ERROR: {got}")
            emit()
            continue
        misses = []
        for f in FIELDS:
            want = getattr(gt, f)
            have = got[f]
            field_total[f] += 1
            ok = have == want
            field_hits[f] += ok
            mark = "ok  " if ok else "MISS"
            detail = f"{f}: {mark} got={have!r} want={want!r}"
            if f == "severity_band":
                detail += f" (raw={got['severity']})"
            if f == "temporal":
                detail += f" (trigger={got['alert_trigger']})"
            emit(f"   {detail}")
            if not ok:
                misses.append(f)
        cd = gt.cooldown_seconds
        emit(
            f"   cooldown (info): builtin={cd or 'default'}s, "
            "custom path emits none (30s global default)"
        )
        emit(f"   name suggestion: {got['suggested_name']!r}")
        emit()

    emit("═" * 64)
    emit("PER-FIELD ACCURACY")
    total_hits = total = 0
    for f in FIELDS:
        h, t = field_hits[f], field_total[f]
        total_hits += h
        total += t
        pct = 100.0 * h / t if t else 0.0
        emit(f"  {f:<15} {h:>2}/{t:<2}  {pct:5.1f}%")
    if total:
        emit(f"  {'OVERALL':<15} {total_hits:>2}/{total:<2}  {100.0 * total_hits / total:5.1f}%")
    if errors:
        emit(f"  compile errors: {errors}")

    if out_path:
        with open(out_path, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\nsaved -> {out_path}")
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="also write the scorecard here")
    args = ap.parse_args()
    return asyncio.run(run_eval(args.out))


if __name__ == "__main__":
    sys.exit(main())
