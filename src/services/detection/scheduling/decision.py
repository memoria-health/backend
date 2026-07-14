"""L3 decision primitives (detection v2, P4).

The v2 decision layer, built PARALLEL to the legacy logic in
`event_promotion` / the absence runner (locked rule C: one owner per
flag state — `detection_v2_decision_enabled` off ⇒ legacy fires and
this module is inert; on ⇒ these primitives own the decisions).
Cleanup Ledger rows 2-4 name what dies once the soak proves this.

Three primitives:

  windowed-count entry   a state enters when ≥M qualifying ticks land
                         within T seconds (patient-level, pooled
                         across every camera — locked multi-camera
                         rule). M=1 (the NULL-eventing default) is
                         today's on-sight rising edge exactly.
  per-detector cooldown  point-shape re-fire spacing from the row's
                         `cooldown_seconds`, replacing the global 30s.
  k-of-N confirm         config parsing for the executor's re-check
                         pass (N extra calls on the SAME clip; ≥k must
                         agree — Aryaman 2026-07-13). No built-in
                         carries config, so parity holds.

Eventing JSONB shape (detectors.eventing, NULL today):
    {"entry": {"m": 3, "t_seconds": 600}, "confirm": {"k": 2, "n": 3}}
Absent/junk keys fall back to parity defaults, never raise — a
malformed row must degrade to today's behavior, not kill promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Float, and_, cast, func, select

from src.db.analysis import Analysis

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class EventingConfig:
    """Parsed per-detector eventing dials, parity-defaulted."""

    entry_m: int = 1          # qualifying ticks needed to enter a state
    entry_t_seconds: int = 0  # window those ticks must land in (0 = on-sight)
    confirm_k: int = 0        # re-checks that must agree (0 = no confirm)
    confirm_n: int = 0        # re-check calls to spend on a fire


def parse_eventing(raw: object) -> EventingConfig:
    """Tolerant parse of the eventing JSONB. Anything malformed —
    wrong types, negative values, k>n — degrades to the parity
    default for that dial."""
    if not isinstance(raw, dict):
        return EventingConfig()

    def _pos_int(container: object, key: str, default: int) -> int:
        if not isinstance(container, dict):
            return default
        v = container.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return default
        return v

    entry = raw.get("entry")
    confirm = raw.get("confirm")
    m = max(1, _pos_int(entry, "m", 1))
    t = _pos_int(entry, "t_seconds", 0)
    if m > 1 and t <= 0:
        # Half-formed window (M without T): with a ~10s tick cadence a
        # 0-second window can never accumulate M ticks — the detector
        # would be silently dead. Degrade to parity (on-sight), per the
        # module contract: malformed config = today's behavior, never
        # detection loss (P4 review finding 1).
        m, t = 1, 0
    k = _pos_int(confirm, "k", 0)
    n = _pos_int(confirm, "n", 0)
    if k > n:
        k = n = 0  # unsatisfiable config — treat as no confirm
    return EventingConfig(entry_m=m, entry_t_seconds=t, confirm_k=k, confirm_n=n)


def detection_keys_for(identifier: str) -> tuple[str, str]:
    """(detected_field, confidence_field) for a detector — from the
    catalog for built-ins (medication's field is `medication_taken`,
    not `medication_detected`), else the standard naming. The confirm
    gate/votes/suppression must use THIS, never string formatting
    (P4 review finding 2)."""
    from src.services.detection import catalog

    entry = catalog.by_key(identifier) if identifier else None
    if entry is not None:
        return entry.detected_key, entry.confidence_key
    return f"{identifier}_detected", f"{identifier}_confidence"


def cooldown_seconds_for(detector, default_seconds: int) -> int:
    """Per-detector point-shape re-fire spacing: the row value when
    set, else the legacy global default (parity for every row the P1
    backfill left NULL)."""
    v = getattr(detector, "cooldown_seconds", None)
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    return default_seconds


async def qualifying_tick_count(
    session: "AsyncSession",
    *,
    patient_id: UUID,
    detected_field: str,
    conf_field: str,
    threshold: float | None,
    since: datetime,
    before: datetime,
) -> int:
    """Count PROMOTABLE ticks (detected=true AND confidence ≥
    threshold) for one detector in [since, before) — patient-level, so
    evidence pools across every camera in the household. The single
    counting implementation for state entry AND (flag-on) absence:
    one query to soak, one to trust."""
    preds = [
        Analysis.patient_id == patient_id,
        Analysis.captured_at >= since,
        Analysis.captured_at < before,
        Analysis.detector_fields[detected_field].astext == "true",
    ]
    # threshold=None ⇒ any detected=true row counts (the absence
    # runner's legacy semantics — no confidence gate). States pass a
    # real threshold so sub-threshold blips never count toward entry.
    if threshold is not None:
        preds.append(
            cast(Analysis.detector_fields[conf_field].astext, Float) >= threshold
        )
    stmt = select(func.count()).select_from(Analysis).where(and_(*preds))
    return int((await session.execute(stmt)).scalar_one())


async def windowed_entry_verdict(
    session: "AsyncSession",
    *,
    patient_id: UUID,
    detected_field: str,
    conf_field: str,
    threshold: float,
    now: datetime,
    cfg: EventingConfig,
) -> tuple[str, int]:
    """Windowed-count entry as a COUNT-CROSSING edge. With prior =
    qualifying ticks in the T-window before now (the current tick is
    qualifying by construction and counts as +1):

        prior + 1 <  M   →  "held"     evidence still accumulating
        prior     <  M ≤ prior + 1  →  "fire"  the window just reached M
        prior     >= M   →  "ongoing"  the episode already entered —
                                       a saturated window means the Mth
                                       tick happened earlier; firing
                                       again here would spam one event
                                       per tick for a long episode

    Returns (verdict, count_including_current). Exit is implicit: a
    lull long enough for the window to drain below M re-arms entry —
    for M>1, T plays the anti-spam role episode_gap plays for M=1.
    (M=1 never reaches this function; the caller keeps the legacy
    rising edge for exact parity.)"""
    window = max(cfg.entry_t_seconds, 1)
    prior = await qualifying_tick_count(
        session,
        patient_id=patient_id,
        detected_field=detected_field,
        conf_field=conf_field,
        threshold=threshold,
        since=now - timedelta(seconds=window),
        before=now,
    )
    count = prior + 1
    if prior >= cfg.entry_m:
        return "ongoing", count
    if count >= cfg.entry_m:
        return "fire", count
    return "held", count
