"""Unit tests for one-event-per-clip primary selection.

When several detectors cross their firing threshold on a single clip
(e.g. a fall AND a general-emergency, or agitation AND calling-for-help),
only ONE event should fire. `_select_primary` ranks, in order:

  1. catalog SEVERITY (higher wins) — choking beats agitation no matter
     what the model hint or confidence say;
  2. the model's explicit `primary_detection` pick, breaking ties WITHIN
     the same severity, when it names a detector that actually crossed
     threshold this clip;
  3. confidence; 4. safety over routine; 5. identifier for determinism.

Identifiers not in the catalog severity map (all customs) share the
default severity, so tiers 2-5 are exercised with custom_* identifiers.
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


def test_severity_beats_confidence_and_hint() -> None:
    """fall (severity 95) outranks emergency (80) even when emergency is
    more confident AND the model names emergency as dominant — the
    catalog's stakes ordering is the top tier, deliberately.
    """
    fall = _c("fall", 0.72)
    emergency = _c("emergency", 0.91)
    assert _select_primary([fall, emergency], primary_hint="") is fall
    assert _select_primary([fall, emergency], primary_hint="emergency") is fall


def test_model_hint_breaks_ties_within_equal_severity() -> None:
    """Two customs share the default severity; the model's pick decides,
    overriding higher confidence.
    """
    pacing = _c("custom_pacing00", 0.72)
    shouting = _c("custom_shout000", 0.91)
    assert _select_primary([pacing, shouting], primary_hint="custom_pacing00") is pacing


def test_falls_back_to_highest_confidence_without_a_usable_hint() -> None:
    pacing = _c("custom_pacing00", 0.72)
    shouting = _c("custom_shout000", 0.91)
    assert _select_primary([pacing, shouting], primary_hint="") is shouting
    # A stale/unknown hint that names no candidate is ignored, not honored.
    assert _select_primary([pacing, shouting], primary_hint="wandering") is shouting
    assert _select_primary([pacing, shouting], primary_hint=None) is shouting


def test_safety_breaks_a_full_tie() -> None:
    """Equal (default) severity, equal confidence, no hint → safety wins
    over routine.
    """
    routine = _c("custom_tvwatch0", 0.80, category="routine")
    safety = _c("custom_nearfall", 0.80, category="safety")
    assert _select_primary([routine, safety], primary_hint="") is safety
