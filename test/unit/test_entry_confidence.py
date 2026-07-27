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
