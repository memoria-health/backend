"""Unit tests for the sensitivity → firing-threshold math.

Pure-function tests (no DB) for `event_promotion._effective_threshold`, which is
what makes the iOS sensitivity picker actually move the alert bar.
"""

from src.services.detection.event_promotion import _effective_threshold


def test_standard_base_yields_the_spec_numbers() -> None:
    # Every catalog detector's base is 0.70 today, so:
    #   low -> 0.85, standard -> 0.70, high -> 0.40
    assert round(_effective_threshold(0.70, "low"), 2) == 0.85
    assert round(_effective_threshold(0.70, "standard"), 2) == 0.70
    assert round(_effective_threshold(0.70, "high"), 2) == 0.40


def test_direction_low_is_stricter_high_is_looser() -> None:
    std = _effective_threshold(0.70, "standard")
    assert _effective_threshold(0.70, "low") > std    # higher bar = fewer alerts
    assert _effective_threshold(0.70, "high") < std   # lower bar = more alerts


def test_clamped_to_sane_band() -> None:
    # A low base can't push "high" below the floor, or "low" above the ceiling.
    assert _effective_threshold(0.10, "high") == 0.05   # 0.10 - 0.30 -> floor
    assert round(_effective_threshold(0.95, "low"), 2) == 0.95   # 0.95 + 0.15 -> ceil


def test_unknown_sensitivity_falls_back_to_standard() -> None:
    # Impossible via the DB enum / iOS picker, but must never crash — it falls
    # back to the standard bar rather than inventing one.
    assert round(_effective_threshold(0.70, "bogus"), 2) == 0.70
    assert round(_effective_threshold(0.70, "off"), 2) == 0.70
