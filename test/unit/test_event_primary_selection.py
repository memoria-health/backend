"""Unit tests for one-event-per-clip primary selection.

When several detectors cross their firing threshold on a single clip
(e.g. a fall AND a general-emergency, or agitation AND calling-for-help),
only ONE event should fire. `_select_primary` decides which:

  1. the model's explicit `primary_detection` pick, when it names a
     detector that actually crossed threshold this clip;
  2. else the highest-confidence detection (safety beats routine on a tie,
     then identifier for determinism).
"""

from __future__ import annotations

from src.services.detection.event_promotion import _Candidate, _select_primary


def _c(ident: str, conf: float, category: str = "safety") -> _Candidate:
    # detector=None — _select_primary only reads identifier/confidence/category.
    return _Candidate(
        identifier=ident, confidence=conf, category=category, detector=None
    )


def test_none_when_no_candidates() -> None:
    assert _select_primary([], primary_hint="fall") is None


def test_single_candidate_wins_regardless_of_hint() -> None:
    c = _c("fall", 0.8)
    assert _select_primary([c], primary_hint="") is c
    assert _select_primary([c], primary_hint="something_else") is c


def test_model_hint_wins_over_higher_confidence() -> None:
    fall = _c("fall", 0.72)
    emergency = _c("general_emergency", 0.91)
    # emergency is more confident, but the model named fall as dominant.
    assert _select_primary([fall, emergency], primary_hint="fall") is fall


def test_falls_back_to_highest_confidence_without_a_usable_hint() -> None:
    fall = _c("fall", 0.72)
    emergency = _c("general_emergency", 0.91)
    assert _select_primary([fall, emergency], primary_hint="") is emergency
    # A stale/unknown hint that names no candidate is ignored, not honored.
    assert _select_primary([fall, emergency], primary_hint="wandering") is emergency
    assert _select_primary([fall, emergency], primary_hint=None) is emergency


def test_safety_breaks_a_confidence_tie() -> None:
    routine = _c("watching_tv", 0.80, category="routine")
    safety = _c("fall", 0.80, category="safety")
    assert _select_primary([routine, safety], primary_hint="") is safety
