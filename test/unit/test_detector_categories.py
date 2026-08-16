"""Behaviour + Care categories, and the category/severity split.

The change these guard: category became SEMANTIC (four values, for
labelling and colour) and treatment moved onto `severity`, which the
scheduler already used for its cost exemptions.

The regression worth fearing is quiet. Nothing crashes if the compile
bias keeps keying on `category == "safety"` — a 3am wandering detector
filed under `behaviour` simply gets the conservative prompt and starts
missing things, and no test, log or alert says so. Hence the explicit
"high-severity non-safety category is still inclusive" case below.
"""

from __future__ import annotations

import pytest

from src.db.enums import DetectorCategoryEnum
from src.schemas.detector import DetectorCategory
from src.services.detector_compiler import (
    INCLUSIVE_BIAS_SEVERITY,
    _coerce_inferred,
    _is_high_stakes,
)

EXPECTED = {"safety", "routine", "behaviour", "care"}


# ── The enum, in both places it is declared ─────────────────────────


def test_python_enum_has_all_four_categories():
    assert {c.value for c in DetectorCategory} == EXPECTED


def test_sqlalchemy_enum_matches_the_pydantic_enum():
    """These are declared separately (src/db/enums.py vs schemas/detector.py).
    Drift means an INSERT that passes validation and then fails in Postgres
    with InvalidTextRepresentation."""
    assert set(DetectorCategoryEnum.enums) == EXPECTED


def test_behaviour_is_spelled_the_british_way():
    """Deliberate: matches the flyer and the compiler prompt. Pinned because
    an enum value is expensive to change once rows and clients use it —
    removing one needs a type rewrite, not a delta."""
    assert DetectorCategory.behaviour.value == "behaviour"
    assert "behavior" not in EXPECTED


# ── Treatment rides on severity, not category ───────────────────────


def test_threshold_matches_the_scheduler_emergency_constant():
    """One concept, two modules. The compiler cannot import the scheduler
    (cycle), so equality is asserted here instead — see the comment on
    INCLUSIVE_BIAS_SEVERITY."""
    from src.services.detection.scheduling.scheduler import _EMERGENCY_SEVERITY

    assert INCLUSIVE_BIAS_SEVERITY == _EMERGENCY_SEVERITY, (
        "compile bias and the scheduler's emergency exemption have drifted; "
        "they are the same 'a miss matters more than a false alarm' judgement"
    )


@pytest.mark.parametrize("category", sorted(EXPECTED))
def test_high_severity_is_inclusive_in_every_category(category):
    """THE regression this change exists to prevent: a high-stakes detector
    must get the inclusive prompt regardless of the label a caregiver's
    wording earned it."""
    assert _is_high_stakes(category=category, severity=95) is True


@pytest.mark.parametrize("category", sorted(EXPECTED))
def test_low_severity_is_conservative_even_when_labelled_safety(category):
    """The converse: labelling something 'safety' must not buy the inclusive
    prompt on its own, or the category becomes a backdoor to the old
    behaviour."""
    assert _is_high_stakes(category=category, severity=10) is False


def test_threshold_is_inclusive_at_the_boundary():
    assert _is_high_stakes(category="care", severity=INCLUSIVE_BIAS_SEVERITY) is True
    assert _is_high_stakes(category="care", severity=INCLUSIVE_BIAS_SEVERITY - 1) is False


def test_missing_severity_falls_back_to_the_old_category_check():
    """Callers not yet threaded through must keep EXACTLY their old
    behaviour — the dangerous direction is a high-stakes detector silently
    becoming conservative."""
    assert _is_high_stakes(category="safety", severity=None) is True
    assert _is_high_stakes(category="routine", severity=None) is False
    assert _is_high_stakes(category="behaviour", severity=None) is False


# ── Parsing the model's answer ──────────────────────────────────────


@pytest.mark.parametrize("category", sorted(EXPECTED))
def test_all_four_categories_survive_coercion(category):
    got = _coerce_inferred({"category": category, "alert_trigger": "leading"})
    assert got.category == category


def test_unknown_category_falls_back_to_safety():
    got = _coerce_inferred({"category": "vibes", "alert_trigger": "leading"})
    assert got.category == "safety"


def test_severity_defaults_to_the_catalog_middle_when_absent():
    """Not 0 and not 100: guessing 0 would make a real emergency
    conservative, guessing 100 would run every malformed response through
    the inclusive prompt."""
    got = _coerce_inferred({"category": "care", "alert_trigger": "leading"})
    assert got.severity == 50


@pytest.mark.parametrize(
    "raw,expected",
    [(0, 0), (100, 100), (150, 100), (-5, 0), (79.6, 79), ("high", 50), (None, 50)],
)
def test_severity_is_clamped_and_type_checked(raw, expected):
    got = _coerce_inferred(
        {"category": "safety", "alert_trigger": "leading", "severity": raw}
    )
    assert got.severity == expected


def test_bool_severity_is_rejected_not_coerced_to_one():
    """bool is a subclass of int in Python, so `True` would otherwise land as
    severity=1 — a silent demotion of whatever the model meant."""
    got = _coerce_inferred(
        {"category": "safety", "alert_trigger": "leading", "severity": True}
    )
    assert got.severity == 50


# ── The built-in catalog after reclassification ─────────────────────


def test_catalog_categories_are_all_valid():
    from src.services.detection import catalog

    assert {s.category for s in catalog.CATALOG} <= EXPECTED


def test_category_is_no_longer_a_near_constant():
    """The change exists because 20 of 25 built-ins were 'safety', which is
    what made the axis useless (scheduler.py:68). Guard the property, not the
    exact counts, so adding a detector does not fail this."""
    from collections import Counter

    from src.services.detection import catalog

    counts = Counter(s.category for s in catalog.CATALOG)
    biggest = counts.most_common(1)[0][1]
    assert biggest / len(catalog.CATALOG) < 0.8, (
        f"one category holds {biggest}/{len(catalog.CATALOG)} of the catalog; "
        f"it has collapsed back into a near-constant: {dict(counts)}"
    )


# test_reclassified_detectors_kept_their_severity deleted 2026-08-16:
# its subject was `distress`, removed with the whole child cohort (the
# app never displayed it). The reclassification-safety property it
# guarded lives on via the severity pins in test_catalog_v2.


def test_care_is_empty_and_that_is_deliberate():
    """No built-in watches for the Care examples the product pitches (help to
    the bathroom, company at meals, long stretches alone). Asserted so the gap
    is visible in test output rather than discovered by a caregiver, and so
    whoever fills it deletes this test on purpose."""
    from src.services.detection import catalog

    assert not [s for s in catalog.CATALOG if s.category == "care"]
