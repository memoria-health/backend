"""Eval harness for the sensitivity classifier (2026-07-27 plan, "Eval
+ QA"). Runs `sensitivity.screen_clip` — the exact pinned-model call
production uses — over a directory of labeled clips and prints a
confusion summary. The staged eval corpus (acted/clothed-proxy clips
across every category + normal-scene negatives) is a follow-up; this
harness is what it will be scored with.

Corpus layout (clips are .mp4):

    <corpus_dir>/
      sensitive/            → expected sensitive=true
        toileting/a.mp4     → optional per-category subdir: also checks
        b.mp4                 the predicted categories include it
      clean/                → expected sensitive=false
        kitchen.mp4

Target: false-negative rate ~0 on the staged set — every FN is listed
loudly; FP rate is the tunable. Exit code 1 if any FN (or any clip
errored), so this can gate a model/prompt change in CI.

Usage:
    uv run python scripts/eval_sensitivity.py demo/sensitivity_eval
    uv run python scripts/eval_sensitivity.py <dir> --model gemini-x
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

# Runnable from the repo root without install (same pattern as
# scripts/backfill_detector_dims.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.services.detection.sensitivity import (
    SENSITIVITY_CATEGORIES,
    screen_clip,
)
from src.settings import get_settings


@dataclass
class Case:
    path: Path
    expected_sensitive: bool
    expected_category: str | None  # from a sensitive/<category>/ subdir


@dataclass
class Outcome:
    case: Case
    predicted_sensitive: bool | None  # None = the call errored
    categories: list[str]
    confidence: float
    error: str | None = None


def _collect(corpus_dir: Path) -> list[Case]:
    cases: list[Case] = []
    for label_dir, expected in (("sensitive", True), ("clean", False)):
        root = corpus_dir / label_dir
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.mp4")):
            category: str | None = None
            if expected and path.parent != root:
                sub = path.parent.name
                category = sub if sub in SENSITIVITY_CATEGORIES else None
            cases.append(
                Case(path=path, expected_sensitive=expected,
                     expected_category=category)
            )
    return cases


async def _run(cases: list[Case], settings) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for i, case in enumerate(cases, 1):
        label = "sensitive" if case.expected_sensitive else "clean"
        print(f"[{i}/{len(cases)}] {case.path} (expect {label}) ... ",
              end="", flush=True)
        try:
            result = await screen_clip(case.path.read_bytes(), settings=settings)
        except Exception as exc:
            print(f"ERROR: {exc}")
            outcomes.append(Outcome(case, None, [], 0.0, error=str(exc)))
            continue
        print(
            f"sensitive={result.sensitive} conf={result.confidence:.2f} "
            f"categories={result.categories}"
        )
        outcomes.append(
            Outcome(case, result.sensitive, result.categories, result.confidence)
        )
    return outcomes


def _summarize(outcomes: list[Outcome]) -> int:
    # In production an errored screen fails CLOSED (censor), so for the
    # confusion matrix an error counts as predicted-sensitive — but it
    # is also reported separately (an eval full of errors proves nothing).
    tp = fp = fn = tn = 0
    fns: list[Outcome] = []
    fps: list[Outcome] = []
    errors: list[Outcome] = []
    category_hits = category_misses = 0
    for o in outcomes:
        predicted = True if o.predicted_sensitive is None else o.predicted_sensitive
        if o.error is not None:
            errors.append(o)
        if o.case.expected_sensitive and predicted:
            tp += 1
            if o.case.expected_category is not None and o.error is None:
                if o.case.expected_category in o.categories:
                    category_hits += 1
                else:
                    category_misses += 1
        elif o.case.expected_sensitive and not predicted:
            fn += 1
            fns.append(o)
        elif not o.case.expected_sensitive and predicted:
            fp += 1
            fps.append(o)
        else:
            tn += 1

    n_sensitive = tp + fn
    n_clean = fp + tn
    print("\n── confusion summary ────────────────────────────────")
    print(f"  clips: {len(outcomes)}  (sensitive={n_sensitive} clean={n_clean} "
          f"errors={len(errors)})")
    print(f"  TP={tp}  FN={fn}  FP={fp}  TN={tn}")
    if n_sensitive:
        print(f"  false-negative rate: {fn / n_sensitive:.3f}  "
              f"(recall {tp / n_sensitive:.3f}) — target ~0")
    if n_clean:
        print(f"  false-positive rate: {fp / n_clean:.3f} — the tunable")
    if category_hits + category_misses:
        print(f"  category attribution: {category_hits}/"
              f"{category_hits + category_misses} labeled clips matched")
    if fns:
        print("\n  FALSE NEGATIVES (each one is a dignity exposure):")
        for o in fns:
            print(f"    - {o.case.path} conf={o.confidence:.2f}")
    if fps:
        print("\n  false positives (over-censored, review for tuning):")
        for o in fps:
            print(f"    - {o.case.path} categories={o.categories}")
    if errors:
        print("\n  errored clips (fail closed in prod; fix for a clean eval):")
        for o in errors:
            print(f"    - {o.case.path}: {o.error}")
    return 1 if fns or errors else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the sensitivity classifier on a labeled clip corpus.",
    )
    parser.add_argument("corpus_dir", type=Path,
                        help="directory with sensitive/ and clean/ subdirs of .mp4s")
    parser.add_argument("--model", default=None,
                        help="override NEMARA_SENSITIVITY_MODEL for this run")
    args = parser.parse_args()

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"sensitivity_model": args.model})

    cases = _collect(args.corpus_dir)
    if not cases:
        print(f"no .mp4s under {args.corpus_dir}/sensitive or "
              f"{args.corpus_dir}/clean — nothing to score", file=sys.stderr)
        return 2
    print(f"model={settings.sensitivity_model} clips={len(cases)}\n")
    outcomes = asyncio.run(_run(cases, settings))
    return _summarize(outcomes)


if __name__ == "__main__":
    raise SystemExit(main())
