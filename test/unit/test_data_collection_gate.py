"""Task 6 — should_collect gate (pure).

Recording happens only when ALL of the existing signals agree:
the patient opted in (data_collection_enabled), there's motion, and
someone is present (gate_people_present, the detection layer's
already-computed presence verdict).

Privacy mode is a hard override on top: when privacy_mode_enabled is
True the gate is always closed, regardless of consent/motion/presence.
"""

from __future__ import annotations

import pytest

from src.services.data_collection.gate import should_collect


@pytest.mark.parametrize(
    "privacy,enabled,motion,present,expected",
    [
        # privacy OFF: the original truth table applies.
        (False, True, True, True, True),  # the only True case
        (False, False, True, True, False),  # not opted in
        (False, True, False, True, False),  # no motion
        (False, True, True, False, False),  # nobody present
        (False, False, False, False, False),
        (False, True, False, False, False),
        # privacy ON: always closed, even when everything else says collect.
        (True, True, True, True, False),  # privacy overrides a full green gate
        (True, False, True, True, False),
        (True, True, False, False, False),
    ],
)
def test_should_collect_truth_table(
    privacy: bool, enabled: bool, motion: bool, present: bool, expected: bool
) -> None:
    result = should_collect(
        privacy_mode_enabled=privacy,
        data_collection_enabled=enabled,
        motion_in_clip=motion,
        gate_people_present=present,
    )
    assert result is expected


def test_privacy_mode_takes_precedence_over_full_green_gate() -> None:
    # Opted in + motion + present would normally collect — privacy wins.
    assert (
        should_collect(
            privacy_mode_enabled=True,
            data_collection_enabled=True,
            motion_in_clip=True,
            gate_people_present=True,
        )
        is False
    )


def test_none_motion_is_treated_as_false() -> None:
    # motion_in_clip comes from a TTL cache that returns None when the
    # camera hasn't sent a recent motion heartbeat — must gate closed.
    assert (
        should_collect(
            privacy_mode_enabled=False,
            data_collection_enabled=True,
            motion_in_clip=None,
            gate_people_present=True,
        )
        is False
    )


def test_returns_a_real_bool() -> None:
    # Never leak a truthy non-bool (e.g. None) to callers.
    assert (
        should_collect(
            privacy_mode_enabled=False,
            data_collection_enabled=True,
            motion_in_clip=True,
            gate_people_present=True,
        )
        is True
    )
