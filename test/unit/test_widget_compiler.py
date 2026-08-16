"""widget_compiler: one Home-tab widget spec per detector, compiled once.

Built-ins resolve their pinned catalog template ($self/$when_scope/
$schedule become row values). Customs go through deterministic
trigger-shape rules; an optional llm_pick hook may swap the choice
WITHIN the primitive library, and anything invalid falls back to the
rules result. Hidden identifiers never compile.
"""

from types import SimpleNamespace
from uuid import uuid4

from src.services.dashboard.widget_compiler import compile_widget


def _det(**over):
    base = dict(
        id=uuid4(), name="Custom thing", identifier="custom_ab12cd34",
        source="custom", category="routine", alert_trigger="leading",
        is_sustained=False, duration_seconds=None, schedule_config=None,
        when_scope="ALWAYS", where_rooms=["Kitchen"], subject_scope="patient",
        severity=40,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ── built-ins: pinned template, resolved ─────────────────────────────

def test_preset_fall_resolves_catalog_template() -> None:
    d = _det(identifier="fall", source="preset", category="safety")
    spec = compile_widget(d)
    assert spec["widget"] == "streak_tile"
    assert spec["params"]["detector"] == str(d.id)
    assert spec["compiler"] == "catalog"
    assert spec["compiled_at"]


def test_preset_wandering_domain_from_when_scope() -> None:
    d = _det(identifier="wandering", source="preset", category="safety",
             when_scope="10 PM - 6 AM")
    spec = compile_widget(d)
    assert spec["params"]["domain"] == ["22:00", "06:00"]


def test_preset_wandering_always_drops_domain() -> None:
    d = _det(identifier="wandering", source="preset", when_scope="ALL DAY")
    assert "domain" not in compile_widget(d)["params"]


def test_preset_medication_windows_from_schedule() -> None:
    sched = {"cadence": "daily", "deadline": "20:00"}
    d = _det(identifier="medication", source="preset", schedule_config=sched)
    assert compile_widget(d)["params"]["windows"] == sched


def test_hidden_setup_test_compiles_to_none() -> None:
    d = _det(identifier="setup_test", source="preset")
    assert compile_widget(d) is None


# ── customs: trigger-shape rules ─────────────────────────────────────

def test_custom_record_only_counts_ticks_not_events() -> None:
    spec = compile_widget(_det(alert_trigger="record_only"))
    assert spec["widget"] == "count_bars"
    assert spec["params"]["metric"] == "detectedTicks"
    assert spec["compiler"] == "rules"


def test_custom_absence_with_deadline_is_adherence_dots() -> None:
    spec = compile_widget(_det(
        alert_trigger="absence",
        schedule_config={"cadence": "daily", "deadline": "09:00"},
    ))
    assert spec["widget"] == "dot_calendar"
    assert spec["params"]["mode"] == "adherence"


def test_custom_sustained_behaviour_is_heatmap() -> None:
    spec = compile_widget(_det(category="behaviour", is_sustained=True))
    assert spec["widget"] == "week_heatmap"


def test_custom_duration_is_duration_bars() -> None:
    spec = compile_widget(_det(alert_trigger="duration", duration_seconds=600))
    assert spec["widget"] == "duration_bars"
    assert "ranges" in spec


def test_custom_environment_multiroom_is_room_bars() -> None:
    spec = compile_widget(_det(subject_scope="environment", where_rooms=None))
    assert spec["widget"] == "room_bars"


def test_custom_high_severity_leading_is_streak() -> None:
    # 85 sits in the real emergency band [80,100] — custom severities
    # clamp to 100, so the bar must catch the whole band, not just 100.
    spec = compile_widget(_det(category="safety", severity=85))
    assert spec["widget"] == "streak_tile"


def test_severity_outranks_sustained_shape() -> None:
    # An agitation-shaped severe custom (leading + sustained) is still a
    # rare alarm: streak, not heatmap.
    spec = compile_widget(_det(category="behaviour", is_sustained=True,
                               severity=90))
    assert spec["widget"] == "streak_tile"


def test_severity_outranks_environment_scope() -> None:
    spec = compile_widget(_det(subject_scope="environment", where_rooms=None,
                               severity=95))
    assert spec["widget"] == "streak_tile"


def test_severity_does_not_hijack_non_leading_triggers() -> None:
    spec = compile_widget(_det(alert_trigger="record_only", severity=95))
    assert spec["widget"] == "count_bars"


def test_custom_night_window_leading_is_hour_clock() -> None:
    spec = compile_widget(_det(when_scope="11 PM - 5 AM"))
    assert spec["widget"] == "hour_clock"
    assert spec["params"]["domain"] == ["23:00", "05:00"]


def test_custom_fallback_is_occurrence_dots() -> None:
    spec = compile_widget(_det())
    assert spec["widget"] == "dot_calendar"
    assert spec["params"]["mode"] == "occurrence"


def test_titles_come_from_name_and_fit() -> None:
    spec = compile_widget(_det(name="Watches television very late at night"))
    assert spec["title"] and len(spec["title"]) <= 24


def test_every_custom_spec_carries_detector_param() -> None:
    d = _det()
    assert compile_widget(d)["params"]["detector"] == str(d.id)


# ── llm hook: constrained, validated, safe to fail ───────────────────

def test_llm_pick_can_swap_within_library() -> None:
    def pick(detector, candidate):
        return {**candidate, "widget": "count_bars",
                "ranges": ["W", "M", "3M"]}
    spec = compile_widget(_det(), llm_pick=pick)
    assert spec["widget"] == "count_bars"
    assert spec["compiler"] == "llm"


def test_llm_pick_invalid_type_falls_back_to_rules() -> None:
    spec = compile_widget(_det(), llm_pick=lambda d, c: {**c, "widget": "pie"})
    assert spec["widget"] == "dot_calendar"
    assert spec["compiler"] == "rules"


def test_llm_pick_exception_falls_back_to_rules() -> None:
    def boom(d, c):
        raise RuntimeError("provider down")
    spec = compile_widget(_det(), llm_pick=boom)
    assert spec["widget"] == "dot_calendar"
    assert spec["compiler"] == "rules"
