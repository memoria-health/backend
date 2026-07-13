"""Detection v2 runtime — L1 scheduler + L2 packer (P3).

This package IS the new architecture and deliberately lives in its own
files (D1, Aryaman 2026-07-13): the legacy single-call tick in
`camera_ingest.py` stays untouched, and the only coupling point is the
`detection_v2_scheduler_enabled` flag branch in the tick body.

Layer map (spec §2, docs/architecture/detection-architecture-v2.md):

    L1  scheduler.resolve_due()   which detectors run this tick, and
                                  against which track targets
    L2  packer / artifacts /      due set → provider calls per bucket
        executor                  recipe (P3 tasks T1-T3)

Everything here is pure computation over inputs the tick already has
(detector loadout, TickSignals, IdentityResult, settings) — no DB, no
providers, no side effects. That's what makes the truth tables
exhaustively unit-testable.
"""

from src.services.detection.scheduling.dims import EffectiveDims, resolve_dims
from src.services.detection.scheduling.scheduler import (
    DueDetector,
    DueSet,
    ScheduledDetector,
    resolve_due,
)

__all__ = [
    "DueDetector",
    "DueSet",
    "EffectiveDims",
    "ScheduledDetector",
    "resolve_dims",
    "resolve_due",
]
