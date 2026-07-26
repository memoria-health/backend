"""Reground-trigger decision logic (pure — no model, no GPU).

Contract: dets>live fires immediately (new-entry latency is the point);
dets<live needs 2 consecutive ticks (a detector blinking on a seated
person must not burn a 12s ground every cycle); None = trigger dead,
never fires, never disturbs the streak.
"""

from sam_service.person_trigger import should_force_reground


def test_more_persons_fires_immediately():
    assert should_force_reground(2, 1, 0) == (True, 0)
    assert should_force_reground(1, 0, 0) == (True, 0)


def test_fewer_persons_needs_two_consecutive_ticks():
    force, streak = should_force_reground(0, 1, 0)
    assert (force, streak) == (False, 1)
    assert should_force_reground(0, 1, streak) == (True, 0)


def test_match_resets_the_under_streak():
    assert should_force_reground(1, 1, 1) == (False, 0)


def test_dead_trigger_is_inert():
    assert should_force_reground(None, 1, 1) == (False, 1)
