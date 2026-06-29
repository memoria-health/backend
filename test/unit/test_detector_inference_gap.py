"""Unit tests for continuous ("state") detector inference + episode gap.

`_coerce_inferred` normalizes the LLM's JSON. For the new `state` shape it
must always produce an `episode_gap_seconds` (a bucket mapped to seconds,
defaulting when the model omits/garbles it), and leave it None for the
momentary / scheduled shapes.
"""

from __future__ import annotations

from src.services.detector_compiler import (
    _DEFAULT_GAP_BUCKET,
    _GAP_BUCKET_SECONDS,
    _coerce_inferred,
)


def _raw(shape: str, **extra) -> dict:
    return {
        "shape": shape,
        "category": "routine",
        "sensitivity": "low",
        "icon_name": "tv",
        **extra,
    }


def test_state_maps_bucket_to_seconds() -> None:
    for bucket, seconds in _GAP_BUCKET_SECONDS.items():
        c = _coerce_inferred(_raw("state", episode_gap=bucket))
        assert c.shape == "state"
        assert c.episode_gap_seconds == seconds


def test_state_missing_or_garbled_bucket_defaults() -> None:
    default = _GAP_BUCKET_SECONDS[_DEFAULT_GAP_BUCKET]
    # No bucket given.
    assert _coerce_inferred(_raw("state")).episode_gap_seconds == default
    # Unknown bucket string.
    assert _coerce_inferred(_raw("state", episode_gap="forever")).episode_gap_seconds == default
    # Non-string.
    assert _coerce_inferred(_raw("state", episode_gap=123)).episode_gap_seconds == default


def test_non_state_shapes_have_no_gap() -> None:
    assert _coerce_inferred(_raw("point_in_time")).episode_gap_seconds is None
    # A bucket on a non-state shape is ignored.
    assert _coerce_inferred(_raw("scheduled", episode_gap="brief")).episode_gap_seconds is None


def test_unknown_shape_falls_back_to_point_in_time_no_gap() -> None:
    c = _coerce_inferred(_raw("nonsense", episode_gap="brief"))
    assert c.shape == "point_in_time"
    assert c.episode_gap_seconds is None
