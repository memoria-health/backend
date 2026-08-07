"""M-of-N entry as an INTERNAL confidence gate by detection difficulty
(Aryaman 2026-07-26). Replaces the old user-duration-parsed entry preset:
the caregiver never dials M-of-N; it's derived from how hard the activity
is to see. Only SUSTAINED detectors accumulate entry evidence.
"""

from src.services.detection.scheduling.compiler_v2 import (
    _entry_confidence_for,
    classify_custom_dims,
)

TICK_SECONDS = 60  # scheduler_tick_seconds default


def _window_reachable(entry: dict) -> bool:
    """M must be physically reachable within the window at the tick rate."""
    e = entry["entry"]
    return e["t_seconds"] >= e["m"] * TICK_SECONDS


def test_easy_sustained_fires_on_sight():
    # Plain scene state, nothing hard to see → no corroboration gate.
    assert _entry_confidence_for(True, "scene", "none", "routine") is None
    assert _entry_confidence_for(True, "person", "none", "safety") is None


def test_hard_to_see_sustained_requires_corroboration():
    routine = _entry_confidence_for(True, "scene", "distance_class", "routine")
    assert routine == {"entry": {"m": 3, "t_seconds": 300}}
    assert _window_reachable(routine)


def test_safety_stakes_cap_corroboration_lower_than_routine():
    # An emergency must not wait as long as a routine annoyance.
    safety = _entry_confidence_for(True, "scene", "distance_class", "safety")
    routine = _entry_confidence_for(True, "scene", "distance_class", "routine")
    assert safety["entry"]["m"] < routine["entry"]["m"]
    assert _window_reachable(safety)


def test_rapid_and_fine_motion_buckets_corroborate():
    for bucket in ("rapid", "fine_motion"):
        cfg = _entry_confidence_for(True, bucket, "none", "routine")
        assert cfg == {"entry": {"m": 2, "t_seconds": 180}}
        assert _window_reachable(cfg)


def test_only_sustained_accumulates_entry_evidence():
    # Instant events fire on-sight; absence fires on its schedule — neither
    # uses entry M-of-N even when the activity is hard to see.
    assert _entry_confidence_for(False, "scene", "distance_class", "safety") is None


def test_duration_words_no_longer_drive_entry():
    # The old behavior snapped "for 10 minutes" to an entry preset. Now a
    # duration in the words is inert — an easy-to-see paced activity fires
    # on-sight regardless of any stated minutes.
    dims = classify_custom_dims(
        name="Pacing",
        user_text="paces around the living room for 10 minutes",
        is_sustained=True,
        category="routine",
    )
    assert dims["eventing"] is None


def test_fine_detail_sustained_gets_gate_via_classify():
    # "fidgets with a pill" → distance_class → a corroboration gate.
    dims = classify_custom_dims(
        name="Fidgeting",
        user_text="keeps fidgeting with a small pill bottle",
        is_sustained=True,
        category="routine",
    )
    assert dims["degradation_warning"] == "distance_class"
    assert dims["eventing"] == {"entry": {"m": 3, "t_seconds": 300}}


def test_instant_routine_gets_within_clip_confirm():
    """Instant non-safety detectors get the within-clip k-of-N confirm
    (2026-08-08, from the drinking-water FP episode: 35 fires/day at
    conf 0.9-1.0 on an instant routine custom, and no code path had
    ever set the confirm dial the executor already implements)."""
    got = _entry_confidence_for(False, "person", "none", "routine")
    assert got == {"confirm": {"k": 2, "n": 3}}
    assert "entry" not in got            # confirm and entry never mix
    # behaviour / care are non-safety stakes → same protection.
    assert _entry_confidence_for(False, "scene", "none", "behaviour") == \
        {"confirm": {"k": 2, "n": 3}}


def test_instant_safety_stays_on_sight():
    # A fall must not risk being outvoted by two noisier re-checks —
    # the same stakes carve-out the sustained branch makes.
    assert _entry_confidence_for(False, "scene", "none", "safety") is None
    assert _entry_confidence_for(False, "scene", "distance_class", "safety") is None


def test_classify_instant_custom_carries_confirm():
    dims = classify_custom_dims(
        name="Drinking water",
        user_text="Ary takes a drink from any vessel, raising it to his mouth",
        is_sustained=False,
        category="routine",
    )
    assert dims["eventing"] == {"confirm": {"k": 2, "n": 3}}


def test_continuation_field_prefers_state_active():
    """Sustained episode latch reads `<id>_state_active` when the schema
    declares it; presets and pre-split customs keep `_detected`."""
    from types import SimpleNamespace
    from src.services.detection.event_promotion import _continuation_field

    split = SimpleNamespace(
        identifier="custom_tv",
        schema_json={
            "custom_tv_detected": {"type": "boolean"},
            "custom_tv_state_active": {"type": "boolean"},
        },
    )
    assert _continuation_field(split, "custom_tv_detected") == \
        "custom_tv_state_active"
    presplit = SimpleNamespace(
        identifier="custom_old",
        schema_json={"custom_old_detected": {"type": "boolean"}},
    )
    assert _continuation_field(presplit, "custom_old_detected") == \
        "custom_old_detected"
    preset = SimpleNamespace(identifier="fall", schema_json=None)
    assert _continuation_field(preset, "fall_detected") == "fall_detected"


def test_schema_for_sustained_adds_state_active():
    from src.services.detector_compiler import _schema_for
    plain = _schema_for("custom_x")
    assert "custom_x_state_active" not in plain
    sustained = _schema_for("custom_x", is_sustained=True)
    assert sustained["custom_x_state_active"] == {"type": "boolean"}
    # The trio is untouched — dashboards key off it.
    for k in ("custom_x_detected", "custom_x_confidence", "custom_x_reasoning"):
        assert k in sustained


def test_meta_prompt_sustained_includes_state_field():
    from src.services.detector_compiler import _meta_prompt
    sustained = _meta_prompt(
        identifier="custom_x", name="Watching TV", description="d",
        user_text="watching tv", absence=False, category="routine",
        is_sustained=True,
    )
    assert "custom_x_state_active" in sustained
    assert "NEVER make it false" in sustained
    instant = _meta_prompt(
        identifier="custom_x", name="Drinks", description="d",
        user_text="drinks water", absence=False, category="routine",
    )
    assert "custom_x_state_active" not in instant
