"""Detection v2 dimensions for the built-in catalog (P1).

The single Python source of truth mapping every catalog identifier to its
v2 dimension values — evidence branch, subject, instrument bucket, and the
decision/delivery attributes that now live on the `detectors` row (spec:
docs/architecture/detection-architecture-v2.md §1–§3). The prod backfill
block in scripts/apply_detector_v2_delta.sql is generated from this table;
keep the two in sync.

Validated at import time against `catalog.CATALOG` and `catalog._SEVERITY`
(same fail-loud convention as `catalog._validate_catalog`): every catalog
key mapped, no strays, enum values legal, severities in lockstep until the
in-code severity map is retired.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.services.detection import catalog

SUBJECT_SCOPES = ("patient", "anyone", "environment")
BUCKETS = ("scene", "person", "fine_motion", "rapid", "face")
EVIDENCE_UNITS = ("verdict", "instance", "sample")
DUTY_CYCLES = ("every_eligible_tick", "thinned", "scheduled_only")
DEGRADATION_WARNINGS = ("none", "distance_class")


@dataclass(frozen=True)
class DetectorDims:
    """One built-in's v2 dimension values, column-named 1:1 with the
    `detectors` row so `row_kwargs()` can splat straight into inserts."""

    # Whether the per-track motion floor applies (True) or a still person is
    # the whole point (False) — the collapse of the old evidence_branch enum
    # (2026-07-26): only person_motion vs not was ever read downstream.
    requires_motion: bool
    subject_scope: str
    bucket: str
    severity: int
    needs_audio: bool = False
    evidence_unit: str = "verdict"
    duty_cycle: str = "every_eligible_tick"
    degradation_warning: str = "none"
    # Point-shape re-fire spacing seeded onto the row at creation; None
    # keeps the global default (30s). Env taxonomy (Aryaman 2026-08-17):
    # glass/water are ACTION detectors (the prompt fires on the event
    # happening in-clip, never on the lingering state), so this is only
    # a spam backstop for a VLM that mistakes a standing puddle/shards
    # for a fresh incident; fire is deliberately state-phrased and
    # re-alerts each cooldown while still burning.
    cooldown_seconds: int | None = None

DIMS: dict[str, DetectorDims] = {
    # ── Seeded presets ──────────────────────────────────────────────
    "fall":        DetectorDims(True, "patient", "scene", 95),
    "agitation":   DetectorDims(True, "patient", "scene", 60,
                                needs_audio=True, evidence_unit="sample",
                                duty_cycle="thinned"),
    # "seizure" removed 2026-07-28 with its catalog entry (VLM samples
    # below tremor frequency; rapid lane stays for warned customs).
    "wandering":   DetectorDims(True, "patient", "scene", 55),
    # Sleep: requires_motion=True is the COST DESIGN, not an oversight —
    # burst ticks label the moments, tick silence between them is the
    # asleep signal (sleep_summary.py). A still sleeper costs $0.
    # NOT thinned (2026-07-28, Aryaman walkthrough): night bursts are
    # sparse and each one may BE the wake / out-of-bed transition —
    # dropping any label corrupts the whole night. The economy comes
    # from tick silence, never from skipping labels.
    "sleep":       DetectorDims(True, "patient", "scene", 15),
    # Activity: PERSON bucket (body-evidence + binds to the patient's
    # identified track — visitors never pollute the score); rides the
    # person call that fires on motion ticks. Same silence-persists
    # economy as sleep. record_only.
    "activity":    DetectorDims(True, "patient", "person", 14),
    "medication":  DetectorDims(True, "patient", "person", 20),
    # ── Adult library ───────────────────────────────────────────────
    "calling_for_help": DetectorDims(True, "patient", "scene", 88,
                                     needs_audio=True),
    "choking":          DetectorDims(True, "patient", "person", 100),
    "meal_eaten":       DetectorDims(True, "patient", "person", 22),
    "stove_left_on":    DetectorDims(False, "environment", "scene", 74,
                                     evidence_unit="sample",
                                     duty_cycle="thinned",
                                     degradation_warning="distance_class"),
    # ── Both cohorts ────────────────────────────────────────────────
    "emergency":         DetectorDims(True, "patient", "scene", 80),
    # Intruder = visible concealment (mask), never "unenrolled person" —
    # only the patient is enrolled, so identity can't be the trigger.
    "possible_intruder": DetectorDims(True, "anyone", "scene", 78),
    "env_hazard_glass":  DetectorDims(False, "environment", "scene", 62,
                                      duty_cycle="thinned",
                                      degradation_warning="distance_class",
                                      cooldown_seconds=1800),
    "env_hazard_water":  DetectorDims(False, "environment", "scene", 56,
                                      duty_cycle="thinned",
                                      degradation_warning="distance_class",
                                      cooldown_seconds=1800),
    "fire_smoke":        DetectorDims(False, "environment", "scene", 98,
                                      cooldown_seconds=600),
    # ── Internal ────────────────────────────────────────────────────
    "setup_test":        DetectorDims(True, "anyone", "scene", 200),
}


def dims_for(identifier: str) -> DetectorDims | None:
    """Dims for a built-in identifier; None for customs (the compiler
    classifies those at creation, P5)."""
    return DIMS.get(identifier)


def row_kwargs(identifier: str) -> dict:
    """Column kwargs for Detector inserts — empty dict for customs, so
    call sites can splat unconditionally."""
    dims = DIMS.get(identifier)
    return asdict(dims) if dims else {}


def _validate() -> None:
    catalog_keys = {e.key for e in catalog.CATALOG}
    if set(DIMS) != catalog_keys:
        missing = catalog_keys - set(DIMS)
        stray = set(DIMS) - catalog_keys
        raise RuntimeError(
            f"[catalog_v2] out of sync with catalog: missing={sorted(missing)} "
            f"stray={sorted(stray)}"
        )
    for key, d in DIMS.items():
        for value, legal in (
            (d.subject_scope, SUBJECT_SCOPES),
            (d.bucket, BUCKETS),
            (d.evidence_unit, EVIDENCE_UNITS),
            (d.duty_cycle, DUTY_CYCLES),
            (d.degradation_warning, DEGRADATION_WARNINGS),
        ):
            if value not in legal:
                raise RuntimeError(f"[catalog_v2] {key}: {value!r} not in {legal}")
        # Lockstep with the in-code severity map until it is retired —
        # promotion falls back to it for rows with NULL severity.
        if d.severity != catalog.severity_for(key):
            raise RuntimeError(
                f"[catalog_v2] {key}: severity {d.severity} != "
                f"catalog._SEVERITY {catalog.severity_for(key)}"
            )


_validate()
