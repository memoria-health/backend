"""Pinned Home-tab widget specs on the catalog.

Contract (design spec, section 4): every caregiver-facing adult/both
built-in carries a pinned widget template; child-cohort and hidden
entries carry none (the app never displays them — no provisioning).
Templates are validated at import time like the rest of the catalog.
"""

import dataclasses

import pytest

from src.services.detection import catalog


ADULT_FACING = [
    e for e in catalog.CATALOG
    if e.cohort in ("adult", "both") and not e.hidden
]

EXPECTED = {
    "fall": "streak_tile",
    "agitation": "week_heatmap",
    "wandering": "hour_clock",
    "sleep": "sleep_bands",
    "activity": "trend_band",
    "medication": "dot_calendar",
    "calling_for_help": "streak_tile",
    "choking": "streak_tile",
    "meal_eaten": "dot_calendar",
    "stove_left_on": "dot_calendar",
    "emergency": "streak_tile",
    "possible_intruder": "hour_clock",
    "env_hazard_glass": "room_bars",
    "env_hazard_water": "room_bars",
    "fire_smoke": "streak_tile",
}


def test_every_adult_facing_entry_has_a_widget() -> None:
    assert {e.key for e in ADULT_FACING} == set(EXPECTED)
    for e in ADULT_FACING:
        assert e.widget is not None, e.key
        assert e.widget["widget"] == EXPECTED[e.key], e.key


def test_child_and_hidden_entries_have_no_widget() -> None:
    for e in catalog.CATALOG:
        if e.cohort == "child" or e.hidden:
            assert e.widget is None, e.key


def test_widget_templates_are_well_formed() -> None:
    for e in ADULT_FACING:
        w = e.widget
        assert w["widget"] in catalog.WIDGET_TYPES, e.key
        assert w["size"] in ("tile", "wide"), e.key
        assert isinstance(w["title"], str) and 0 < len(w["title"]) <= 24, e.key
        assert isinstance(w["params"], dict) and w["params"].get("metric"), e.key
        assert isinstance(w["min_days"], int) and w["min_days"] >= 0, e.key
        # ranges only on the ranged primitives (design spec section 3)
        if "ranges" in w:
            assert w["widget"] in (
                "trend_band", "count_bars", "duration_bars", "sleep_bands",
            ), e.key


def test_medication_adherence_rides_schedule_windows() -> None:
    med = next(e for e in catalog.CATALOG if e.key == "medication")
    assert med.widget["params"]["mode"] == "adherence"


def test_wandering_axis_domain_comes_from_when_scope() -> None:
    wan = next(e for e in catalog.CATALOG if e.key == "wandering")
    assert wan.widget["params"]["domain"] == "$when_scope"


def test_validator_rejects_bad_widget_type() -> None:
    fall = next(e for e in catalog.CATALOG if e.key == "fall")
    bad = dataclasses.replace(fall, widget={**fall.widget, "widget": "pie"})
    with pytest.raises(ValueError, match="widget"):
        catalog._validate_widget(bad)


def test_validator_rejects_widget_on_hidden_entry() -> None:
    setup = next(e for e in catalog.CATALOG if e.key == "setup_test")
    bad = dataclasses.replace(
        setup, widget={"widget": "streak_tile", "size": "tile",
                       "title": "x", "params": {"metric": "daysSince"},
                       "min_days": 0},
    )
    with pytest.raises(ValueError, match="hidden"):
        catalog._validate_widget(bad)
