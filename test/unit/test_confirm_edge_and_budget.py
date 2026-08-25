"""The confirmer re-checks a change, and cannot outspend its allowance.

Both rules exist because of one night. A confirmer was armed on the
sleep detector to check DISTURBANCES. The only lever is k-of-N on
`<ident>_detected`, and `sleep_detected` means ASLEEP — the ordinary
condition that holds for eight hours — so it voted three times on every
tick the person was asleep:

    03:26 -> 10:14 IST   905 ticks voted   confirmed 852   vetoed 53
                         33,258,129 input tokens

Nothing was wrong with the votes. The question was asked 905 times when
it should have been asked when the answer changed.
"""

from __future__ import annotations

from src.services.detection.scheduling.executor import (
    ConfirmBudget, _confirm_is_due,
)


# ── the edge ─────────────────────────────────────────────────────────

def test_a_state_that_has_not_moved_is_not_re_checked() -> None:
    """Asleep, still asleep, still asleep. One question, not three."""
    prev = {"sleep_detected": True}
    assert _confirm_is_due("sleep_detected", {"sleep_detected": True}, prev) is False


def test_the_moment_it_changes_is_checked() -> None:
    """Awake -> asleep, and asleep -> awake, are the disturbances."""
    assert _confirm_is_due(
        "sleep_detected", {"sleep_detected": True},
        {"sleep_detected": False}) is True


def test_going_false_is_not_a_fire_to_confirm() -> None:
    """A verdict that drops to false has nothing to suppress.

    The confirmer's only power is to flip a true to false. Voting on a
    value that is already false spends calls to change nothing.
    """
    assert _confirm_is_due(
        "sleep_detected", {"sleep_detected": False},
        {"sleep_detected": True}) is False


def test_the_first_tick_after_a_restart_is_checked() -> None:
    """No history means treat a true as new.

    Guessing the other way skips the check on the first real fire after
    every deploy, which is the one nobody would notice missing.
    """
    assert _confirm_is_due("fall_detected", {"fall_detected": True}, None) is True
    assert _confirm_is_due("fall_detected", {"fall_detected": False}, None) is False


def test_an_instant_detector_is_unaffected() -> None:
    """A fall is over by the next tick, so edge and level agree.

    This is why the old behaviour looked correct for years: every
    detector that mattered was momentary.
    """
    prev = {"fall_detected": False}
    assert _confirm_is_due("fall_detected", {"fall_detected": True}, prev) is True


# ── the budget ───────────────────────────────────────────────────────

def test_the_allowance_runs_out() -> None:
    b = ConfirmBudget(cap_per_hour=10)
    assert b.take(3) == 3
    assert b.take(3) == 3
    assert b.take(3) == 3
    # 9 spent; only 1 left, so a 3-vote round gets a partial grant that
    # the caller is required to refuse.
    assert b.take(3) == 1
    assert b.take(3) == 0
    assert b.spent_last_hour == 10


def test_a_zero_cap_grants_nothing() -> None:
    """The off switch, without a second flag to forget to set."""
    assert ConfirmBudget(cap_per_hour=0).take(1) == 0


def test_asking_for_nothing_costs_nothing() -> None:
    b = ConfirmBudget(cap_per_hour=5)
    assert b.take(0) == 0
    assert b.spent_last_hour == 0


# ── Scoping a confirm to a FIELD and its VALUES ────────────────────────
#
# The edge alone is not enough for a state detector. `sleep_detected`
# means ASLEEP, so its edge fires on every drift in and out of sleep —
# 1,712 over 7 days of real ticks. What a caregiver wants re-checked
# lives in `sleep_activity`, and only three of its six values are events:
#
#   none <-> awake_in_bed   1,286 of 1,510 transitions   noise
#   entering_bed  26 | leaving_bed 27 | sitting_up 18     ~10 a night
#
# So the confirm is scoped to a field and a set of its values, which is
# the primitive docs/plans/2026-08-24-sleep-detector-rework.md called
# missing.

SLEEP_EVENTS = ("entering_bed", "leaving_bed", "sitting_up")


def _due(now, prev, field="sleep_activity", values=SLEEP_EVENTS):
    return _confirm_is_due("sleep_detected", now, prev, field, values)


def test_entering_a_watched_value_is_an_edge():
    assert _due({"sleep_activity": "leaving_bed"},
                {"sleep_activity": "none"}) is True


def test_holding_a_watched_value_is_not_re_confirmed():
    """The 905-votes-a-night bug, in its new clothes. Someone can be
    mid-way out of bed across several ticks; that is one event."""
    assert _due({"sleep_activity": "leaving_bed"},
                {"sleep_activity": "leaving_bed"}) is False


def test_the_noisy_values_never_fire():
    """none <-> awake_in_bed is 85% of all activity transitions and is
    not an event. If this ever returns True the confirm is back to
    running most of the night."""
    assert _due({"sleep_activity": "awake_in_bed"},
                {"sleep_activity": "none"}) is False
    assert _due({"sleep_activity": "none"},
                {"sleep_activity": "awake_in_bed"}) is False
    assert _due({"sleep_activity": "stirring"},
                {"sleep_activity": "none"}) is False


def test_moving_between_two_watched_values_still_counts():
    """Sitting up and then getting out of bed is two things happening,
    not one."""
    assert _due({"sleep_activity": "leaving_bed"},
                {"sleep_activity": "sitting_up"}) is True


def test_no_previous_tick_counts_as_an_edge():
    assert _due({"sleep_activity": "entering_bed"}, None) is True
    assert _due({"sleep_activity": "none"}, None) is False


def test_without_a_scope_it_is_the_old_detected_edge():
    """Parity: every detector that names no field behaves exactly as
    before, so this cannot change a fall."""
    assert _confirm_is_due("fall_detected", {"fall_detected": True},
                           {"fall_detected": False}) is True
    assert _confirm_is_due("fall_detected", {"fall_detected": True},
                           {"fall_detected": True}) is False
