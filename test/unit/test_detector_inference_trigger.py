"""Unit tests for `_coerce_inferred` — normalizing the compiler's JSON.

Replaces test_detector_inference_gap.py (2026-08-06). Three of its four
tests pinned `episode_gap_seconds`, which has been removed: it was inferred,
stored and served, and read by NOTHING — the live hysteresis is the
criticality-derived exit gap in `scheduling.decision`. The fourth test also
pinned the unknown-trigger fallback, which is still real, so it is kept here.
"""

from __future__ import annotations

from src.services.detector_compiler import _coerce_inferred


def _raw(alert_trigger: str = "leading", is_sustained: bool = False, **extra) -> dict:
    return {
        "alert_trigger": alert_trigger,
        "is_sustained": is_sustained,
        "category": "routine",
        "sensitivity": "low",
        "icon_name": "tv",
        **extra,
    }


def test_unknown_trigger_falls_back_to_leading() -> None:
    """A garbled trigger must degrade to today's behaviour, never raise."""
    assert _coerce_inferred(_raw(alert_trigger="nonsense")).alert_trigger == "leading"


def test_every_valid_trigger_survives_coercion() -> None:
    """record_only added 2026-08-06 — a routine state the caregiver wants to
    SEE, not be TOLD about. Before it existed, the validator rejected it and
    every custom detector was forced to `leading`, which is why `Watching TV`
    alerted three times in nineteen minutes."""
    for trigger in ("leading", "trailing", "duration", "absence", "record_only"):
        got = _coerce_inferred(_raw(alert_trigger=trigger, is_sustained=True))
        assert got.alert_trigger == trigger, f"{trigger} was coerced away"
