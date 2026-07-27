"""Effective v2 dimensions for a detector row (P3).

The scheduler is ONE generic function evaluated per detector ROW —
behavior comes from the row's dimension columns, never from
per-detector code. This module resolves those dimensions with a
three-step fallback, so every row that exists today gets a sane,
run-like-today answer:

  1. The row's own v2 columns (P1 backfill / born-classified seeding /
     the future P5 compiler for customs).
  2. The in-code catalog map (`catalog_v2.DIMS`) by identifier — covers
     built-in rows on a database that predates the P1 backfill.
  3. The legacy default — pre-P5 customs whose columns are NULL and
     whose identifier isn't in the catalog. They behave exactly like
     today: wide SCENE call, due every eligible tick, no subject
     suppression (`anyone`). The P5 compiler replaces this guess with
     a real classification; `dims_source` says which step answered so
     soak data can tell them apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.services.detection import catalog_v2

if TYPE_CHECKING:
    from src.db.detector import Detector

# Buckets whose artifact is a crop targeted at a specific track. SCENE
# is scene-level (full frame), never per-track.
CROP_BUCKETS = frozenset({"person", "fine_motion", "rapid", "face"})


@dataclass(frozen=True)
class EffectiveDims:
    """The dimension values the scheduler actually acts on."""

    # Collapse of the old evidence_branch enum (2026-07-18, Aryaman):
    # 'environment' was always derivable from subject_scope; the only
    # real information was person_motion vs person_still — which is
    # this boolean (does the per-track motion floor apply, or is a
    # still person the whole point). Same semantics, one axis fewer.
    requires_motion: bool
    subject_scope: str      # patient | anyone | environment
    bucket: str             # scene | person | fine_motion | rapid | face
    duty_cycle: str         # every_eligible_tick | thinned | scheduled_only
    needs_audio: bool
    severity: int
    alert_trigger: str      # leading | trailing | duration | schedule
    dims_source: str        # row | catalog | legacy_default
    # k-of-N confirmation (P4, Aryaman: N extra calls on the SAME
    # clip, ≥k must agree). Parsed from the row's eventing JSONB;
    # 0/0 = no confirmation (every row today — parity).
    confirm_k: int = 0
    confirm_n: int = 0
    # How this detector FIRES (2026-07-18 unification) — the one
    # readable answer to "when does an event happen":
    #   instant     first qualifying occurrence (point/state downstream)
    #   accumulate  entry_m qualifying ticks within entry_t_seconds
    #   window      scheduled shape — hit/miss against its window
    fire_rule: str = "instant"
    entry_m: int = 1
    entry_t_seconds: int = 0


_LEGACY_DEFAULT = dict(
    requires_motion=True,
    subject_scope="anyone",
    bucket="scene",
    duty_cycle="every_eligible_tick",
    needs_audio=False,
    severity=50,
)


def _normalize_bucket(bucket: str) -> str:
    """snapshot bucket retired (2026-07-18): it always rode the scene
    call anyway (full-frame stills failed object checks — packer note);
    rows/catalogs that still say 'snapshot' resolve to 'scene' so
    pre-delta databases keep working."""
    return "scene" if bucket == "snapshot" else bucket


def _normalize_duty(duty: str) -> str:
    """'sampled' renamed to 'thinned' (2026-07-18) — the old name
    collided with accumulate-style firing. Pre-delta rows keep working."""
    return "thinned" if duty == "sampled" else duty


def _fire_rule(alert_trigger: str | None, entry_m: int) -> str:
    """The unified firing semantics (see EffectiveDims.fire_rule).

    schedule → window; everything else accumulates (M>1) or fires on-sight.
    Phase E adds dedicated 'trailing' / 'duration' rules; until then those
    triggers ride the leading path (no existing detector uses them)."""
    if alert_trigger == "absence":
        return "window"
    return "accumulate" if entry_m > 1 else "instant"


def resolve_dims(row: "Detector") -> EffectiveDims:
    """Resolve the effective dimensions for one detector row."""
    from src.services.detection.scheduling.decision import parse_eventing

    ev = parse_eventing(getattr(row, "eventing", None))
    confirm = dict(
        confirm_k=ev.confirm_k, confirm_n=ev.confirm_n,
        fire_rule=_fire_rule(row.alert_trigger, ev.entry_m),
        entry_m=ev.entry_m, entry_t_seconds=ev.entry_t_seconds,
    )
    catalog_dims = catalog_v2.dims_for(row.identifier or "")
    if row.evidence_branch and row.subject_scope and row.bucket and row.duty_cycle:
        if row.severity is not None:
            severity = row.severity
        elif catalog_dims is not None:
            severity = catalog_dims.severity
        else:
            severity = _LEGACY_DEFAULT["severity"]
        return EffectiveDims(
            # The DB column keeps the old enum; only person_motion
            # means "apply the motion floor".
            requires_motion=(row.evidence_branch == "person_motion"),
            subject_scope=row.subject_scope,
            bucket=_normalize_bucket(row.bucket),
            duty_cycle=_normalize_duty(row.duty_cycle),
            needs_audio=bool(row.needs_audio),
            severity=severity,
            alert_trigger=row.alert_trigger,
            dims_source="row",
            **confirm,
        )
    if catalog_dims is not None:
        return EffectiveDims(
            requires_motion=(catalog_dims.evidence_branch == "person_motion"),
            subject_scope=catalog_dims.subject_scope,
            bucket=_normalize_bucket(catalog_dims.bucket),
            duty_cycle=_normalize_duty(catalog_dims.duty_cycle),
            needs_audio=catalog_dims.needs_audio,
            # Row severity wins even on the catalog branch — same
            # precedence event promotion uses (row value preferred,
            # catalog fallback), so the packer's severity-ordered fill
            # can never disagree with primary selection.
            severity=(
                row.severity if row.severity is not None
                else catalog_dims.severity
            ),
            alert_trigger=row.alert_trigger,
            dims_source="catalog",
            **confirm,
        )
    return EffectiveDims(
        alert_trigger=row.alert_trigger, dims_source="legacy_default", **confirm, **_LEGACY_DEFAULT,
    )
