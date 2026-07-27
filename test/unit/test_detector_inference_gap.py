"""Unit tests for sustained-detector inference + episode gap.

`_coerce_inferred` normalizes the LLM's JSON. For a SUSTAINED detector it
must always produce an `episode_gap_seconds` (a bucket mapped to seconds,
defaulting when the model omits/garbles it), and leave it None for instant
/ schedule triggers.
"""

from __future__ import annotations

from src.services.detector_compiler import (
    _DEFAULT_GAP_BUCKET,
    _GAP_BUCKET_SECONDS,
    _coerce_inferred,
)


def _raw(alert_trigger: str = "leading", is_sustained: bool = False, **extra) -> dict:
    return {
        "alert_trigger": alert_trigger,
        "is_sustained": is_sustained,
        "category": "routine",
        "sensitivity": "low",
        "icon_name": "tv",
        **extra,
    }


def test_sustained_maps_bucket_to_seconds() -> None:
    for bucket, seconds in _GAP_BUCKET_SECONDS.items():
        c = _coerce_inferred(_raw(is_sustained=True, episode_gap=bucket))
        assert c.is_sustained is True
        assert c.episode_gap_seconds == seconds


def test_sustained_missing_or_garbled_bucket_defaults() -> None:
    default = _GAP_BUCKET_SECONDS[_DEFAULT_GAP_BUCKET]
    # No bucket given.
    assert _coerce_inferred(_raw(is_sustained=True)).episode_gap_seconds == default
    # Unknown bucket string.
    assert _coerce_inferred(_raw(is_sustained=True, episode_gap="forever")).episode_gap_seconds == default
    # Non-string.
    assert _coerce_inferred(_raw(is_sustained=True, episode_gap=123)).episode_gap_seconds == default


def test_instant_and_schedule_have_no_gap() -> None:
    assert _coerce_inferred(_raw(alert_trigger="leading", is_sustained=False)).episode_gap_seconds is None
    # schedule forces is_sustained=False, so a stray bucket is ignored.
    assert _coerce_inferred(
        _raw(alert_trigger="schedule", is_sustained=True, episode_gap="brief")
    ).episode_gap_seconds is None


def test_unknown_trigger_falls_back_to_leading_no_gap() -> None:
    c = _coerce_inferred(_raw(alert_trigger="nonsense", is_sustained=False, episode_gap="brief"))
    assert c.alert_trigger == "leading"
    assert c.episode_gap_seconds is None
