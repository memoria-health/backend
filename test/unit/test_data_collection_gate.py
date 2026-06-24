"""Task 6 — should_collect gate (pure).

Recording happens only when ALL of the existing signals agree:
the patient opted in (data_collection_enabled), there's motion, and
someone is present (gate_people_present, the detection layer's
already-computed presence verdict).
"""

from __future__ import annotations

import pytest

from src.services.data_collection.gate import should_collect


@pytest.mark.parametrize(
    "enabled,motion,present,expected",
    [
        (True, True, True, True),  # the only True case
        (False, True, True, False),  # not opted in
        (True, False, True, False),  # no motion
        (True, True, False, False),  # nobody present
        (False, False, False, False),
        (True, False, False, False),
    ],
)
def test_should_collect_truth_table(
    enabled: bool, motion: bool, present: bool, expected: bool
) -> None:
    result = should_collect(
        data_collection_enabled=enabled,
        motion_in_clip=motion,
        gate_people_present=present,
    )
    assert result is expected


def test_none_motion_is_treated_as_false() -> None:
    # motion_in_clip comes from a TTL cache that returns None when the
    # camera hasn't sent a recent motion heartbeat — must gate closed.
    assert (
        should_collect(
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
            data_collection_enabled=True, motion_in_clip=True, gate_people_present=True
        )
        is True
    )
