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
# and SNAPSHOT are scene-level (full frame / zone), never per-track.
CROP_BUCKETS = frozenset({"person", "fine_motion", "rapid", "face"})


@dataclass(frozen=True)
class EffectiveDims:
    """The dimension values the scheduler actually acts on."""

    evidence_branch: str    # environment | person_motion | person_still
    subject_scope: str      # patient | anyone | environment
    bucket: str             # scene | person | fine_motion | rapid | face | snapshot
    duty_cycle: str         # every_eligible_tick | sampled | scheduled_only
    needs_audio: bool
    severity: int
    shape: str              # detector row shape (state/point_in_time/scheduled)
    dims_source: str        # row | catalog | legacy_default


_LEGACY_DEFAULT = dict(
    evidence_branch="person_motion",
    subject_scope="anyone",
    bucket="scene",
    duty_cycle="every_eligible_tick",
    needs_audio=False,
    severity=50,
)


def resolve_dims(row: "Detector") -> EffectiveDims:
    """Resolve the effective dimensions for one detector row."""
    catalog_dims = catalog_v2.dims_for(row.identifier or "")
    if row.evidence_branch and row.subject_scope and row.bucket and row.duty_cycle:
        if row.severity is not None:
            severity = row.severity
        elif catalog_dims is not None:
            severity = catalog_dims.severity
        else:
            severity = _LEGACY_DEFAULT["severity"]
        return EffectiveDims(
            evidence_branch=row.evidence_branch,
            subject_scope=row.subject_scope,
            bucket=row.bucket,
            duty_cycle=row.duty_cycle,
            needs_audio=bool(row.needs_audio),
            severity=severity,
            shape=row.shape,
            dims_source="row",
        )
    if catalog_dims is not None:
        return EffectiveDims(
            evidence_branch=catalog_dims.evidence_branch,
            subject_scope=catalog_dims.subject_scope,
            bucket=catalog_dims.bucket,
            duty_cycle=catalog_dims.duty_cycle,
            needs_audio=catalog_dims.needs_audio,
            # Row severity wins even on the catalog branch — same
            # precedence event promotion uses (row value preferred,
            # catalog fallback), so the packer's severity-ordered fill
            # can never disagree with primary selection.
            severity=(
                row.severity if row.severity is not None
                else catalog_dims.severity
            ),
            shape=row.shape,
            dims_source="catalog",
        )
    return EffectiveDims(shape=row.shape, dims_source="legacy_default", **_LEGACY_DEFAULT)
