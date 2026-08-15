"""should_recompile: which PATCHed fields re-run the widget compiler.

Spec stability rule: a detector's widget only changes when a field that
feeds the compiler changes. `is_on`, sensitivity and description are
display/behaviour-only — recompiling on them would churn compiled_at
for nothing. `name` DOES recompile: custom titles derive from it.
"""

from src.services.dashboard.widget_compiler import SHAPE_FIELDS, should_recompile


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


def test_shape_fields_is_the_documented_set() -> None:
    assert SHAPE_FIELDS == {
        "name", "category", "alert_trigger", "is_sustained",
        "duration_seconds", "schedule_config", "when_scope",
        "where_rooms", "subject_scope", "severity",
    }
