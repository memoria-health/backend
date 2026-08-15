"""should_recompile: which PATCHed fields re-run the widget compiler.

Spec stability rule: a detector's widget only changes when a field that
feeds the compiler changes. `is_on`, sensitivity and description are
display/behaviour-only — recompiling on them would churn compiled_at
for nothing. `name` DOES recompile: custom titles derive from it.
"""

from types import SimpleNamespace
from uuid import uuid4

from src.services.dashboard.widget_compiler import (
    SHAPE_FIELDS, apply_widget, should_recompile,
)


def test_shape_fields_recompile() -> None:
    for f in ("name", "category", "alert_trigger", "is_sustained",
              "duration_seconds", "schedule_config", "when_scope",
              "where_rooms", "subject_scope", "severity"):
        assert should_recompile({f: "x"}), f


def test_display_fields_do_not_recompile() -> None:
    for f in ("is_on", "sensitivity", "icon_name", "description",
              "custom_prompt"):
        assert not should_recompile({f: "x"}), f


def test_mixed_patch_recompiles() -> None:
    assert should_recompile({"is_on": True, "when_scope": "ALL DAY"})


def test_apply_widget_failure_keeps_old_spec_and_never_raises() -> None:
    # severity as a string makes the rules comparison raise TypeError —
    # a stand-in for any future compiler bug. A widget bug must never
    # break a caregiver-facing CRUD write or roll back a compile.
    det = SimpleNamespace(
        id=uuid4(), name="X", identifier="custom_ab12cd34", source="custom",
        category="routine", alert_trigger="leading", is_sustained=False,
        duration_seconds=None, schedule_config=None, when_scope="ALWAYS",
        where_rooms=None, subject_scope="patient", severity="boom",
        dashboard_widget={"widget": "dot_calendar"},
    )
    apply_widget(det)
    assert det.dashboard_widget == {"widget": "dot_calendar"}


def test_apply_widget_sets_spec_on_success() -> None:
    det = SimpleNamespace(
        id=uuid4(), name="X", identifier="custom_ab12cd34", source="custom",
        category="routine", alert_trigger="leading", is_sustained=False,
        duration_seconds=None, schedule_config=None, when_scope="ALWAYS",
        where_rooms=None, subject_scope="patient", severity=40,
        dashboard_widget=None,
    )
    apply_widget(det)
    assert det.dashboard_widget["widget"] == "dot_calendar"


def test_shape_fields_is_the_documented_set() -> None:
    assert SHAPE_FIELDS == {
        "name", "category", "alert_trigger", "is_sustained",
        "duration_seconds", "schedule_config", "when_scope",
        "where_rooms", "subject_scope", "severity",
    }
