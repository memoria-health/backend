"""Phase E firing: the duration trigger reuses the windowed-count crossing
(fire once the activity has persisted ~duration_seconds); the trailing trigger
fires on the falling edge via the clock runner (DB-driven — soak-validated).
These unit tests cover the pure duration→entry mapping + fire-rule wiring.
"""

from src.services.detection.event_promotion import _apply_duration_entry
from src.services.detection.scheduling.decision import parse_eventing
from src.services.detection.scheduling.dims import _fire_rule

TICK = 60  # assumed scheduler cadence


def test_duration_entry_maps_seconds_to_reachable_window():
    cfg = parse_eventing(None)
    for seconds, expected_m in [(600, 10), (300, 5), (120, 2)]:
        c = _apply_duration_entry(cfg, seconds)
        assert c.entry_t_seconds == seconds
        assert c.entry_m == expected_m
        # M must be reachable within the window at the tick rate.
        assert c.entry_t_seconds >= c.entry_m * TICK


def test_short_duration_clamps_to_min_two():
    # A single tick isn't "sustained" — never require fewer than 2.
    c = _apply_duration_entry(parse_eventing(None), 30)
    assert c.entry_m == 2


def test_duration_takes_the_accumulate_crossing_path():
    # entry_m > 1 → the windowed-count crossing (fires once, re-arms on a lull).
    assert _fire_rule("duration", 10) == "accumulate"


def test_schedule_is_window_not_reactive():
    assert _fire_rule("absence", 1) == "window"
