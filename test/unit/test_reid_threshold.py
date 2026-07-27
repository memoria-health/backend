"""Adaptive threshold unit tests.

Pure math, no DB, no models. The threshold curve has to keep working
under three regimes:

  - Clear close face: high det_score, big bbox -> tight threshold (~0.5)
  - Wall-cam normal:  moderate det_score, moderate bbox -> ~0.40
  - Dim distant face: low det_score, small bbox -> ~0.20-0.30 floor

The match threshold is the safety mechanism for identity gating, so
the curve being right is load-bearing. A regression here means we
either false-positive non-patients (threshold too low) or
false-negative the patient under realistic wall-cam conditions
(threshold too high).
"""

from __future__ import annotations

import pytest

from src.services.identity.embeddings import effective_match_threshold


def test_clear_close_face_yields_tight_threshold():
    """det_score=0.9, 200x200 px = 40000 px. Strong signal; we want
    the match bar held at the upper-end of the band."""
    t = effective_match_threshold(0.40, det_score=0.9, face_pixel_area=40000)
    assert 0.45 <= t <= 0.55, f"clear-face threshold drifted: {t}"


def test_wall_cam_normal_lands_around_base():
    """det_score=0.7, 100x100 px = 10000 px. Matches today's
    hardcoded 0.40 lock threshold; we want the curve to NOT change
    the calibration point in steady state."""
    t = effective_match_threshold(0.40, det_score=0.7, face_pixel_area=10000)
    assert 0.35 <= t <= 0.45, f"wall-cam threshold drifted: {t}"


def test_low_quality_face_drops_toward_floor():
    """det_score=0.4, 50x50 px = 2500 px. This is the scenario-9
    extra-low-light failure from the experiment — must produce a
    threshold low enough that a 0.31 cosine match would lock."""
    t = effective_match_threshold(0.40, det_score=0.4, face_pixel_area=2500)
    assert 0.20 <= t <= 0.30, f"low-quality threshold drifted: {t}"


def test_threshold_floor_at_0_20():
    """Catastrophic face quality (det_score≈0, single pixel face)
    must not crash to ~0. The 0.20 floor prevents that."""
    t = effective_match_threshold(0.40, det_score=0.0, face_pixel_area=1)
    assert t >= 0.20, f"threshold went below floor: {t}"


def test_threshold_ceiling_at_0_55():
    """Extremely high signal (det_score=1.0, giant face) must not
    push the threshold above the ceiling either. We don't want
    99-percentile-only matches in production."""
    t = effective_match_threshold(0.50, det_score=1.0, face_pixel_area=1_000_000)
    assert t <= 0.55, f"threshold went above ceiling: {t}"


def test_zero_face_area_doesnt_crash():
    """The size_log term applies log10(area / 5000) which would
    diverge for area=0. Confirm the formula clamps safely."""
    t = effective_match_threshold(0.40, det_score=0.5, face_pixel_area=0)
    assert 0.20 <= t <= 0.55


@pytest.mark.parametrize(
    "det_score, area, expected_min, expected_max",
    [
        # Mid-quality, mid-size
        (0.5, 8000, 0.25, 0.40),
        # High det_score, mid-size (good selfie crop quality)
        (0.85, 12000, 0.38, 0.48),
        # Low det_score, big face (close but blurry)
        (0.3, 50000, 0.25, 0.42),
        # High det_score, tiny face (clear far-field)
        (0.85, 4000, 0.32, 0.44),
    ],
)
def test_threshold_curve_calibration_points(
    det_score: float, area: float, expected_min: float, expected_max: float,
):
    """Spot-check a few more points along the curve so a future tweak
    to the formula trips the test rather than silently shifting all
    calibration."""
    t = effective_match_threshold(0.40, det_score=det_score, face_pixel_area=area)
    assert expected_min <= t <= expected_max, (
        f"det_score={det_score} area={area}: got {t:.3f}, "
        f"expected [{expected_min}, {expected_max}]"
    )
