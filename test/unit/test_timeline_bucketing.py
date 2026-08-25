"""Two identity services, one strip, and the phase between them.

One process per sensor means the dashboard has to merge samples from
several services onto one time axis. They sample on their own clocks,
so the merge buckets to the sample period — and the bucketing has to
survive the services being out of phase, because they always are.
"""

from __future__ import annotations

PERIOD = 10


def _bucket(ts: float) -> int:
    """The dashboard's rule (tools/soak_dashboard.py:identity_timeline)."""
    return int(float(ts) // PERIOD)


# Observed on the node, 2026-08-25. Four seconds apart, which is
# nothing — except that dividing the TV Room's stamps by 10 lands them
# exactly on .5, and Python rounds a tie to EVEN.
BEDROOM = [1787635399, 1787635409, 1787635419, 1787635429]
TV_ROOM = [1787635395, 1787635405, 1787635415, 1787635425]


def test_out_of_phase_services_land_in_the_same_buckets() -> None:
    """Every bucket that has one camera has the other.

    With `round`, ...395 and ...405 both became the same bucket and the
    one between them got no TV Room sample at all. The strip renders
    `row[cam] || 'empty'`, so the TV Room read as empty on every second
    tick — for as long as anyone watched — while the camera was ticking
    cleanly every 10 seconds and nothing was wrong with it.
    """
    bedroom = {_bucket(t) for t in BEDROOM}
    tv = {_bucket(t) for t in TV_ROOM}
    assert bedroom == tv, (
        f"buckets disagree: bedroom-only={sorted(bedroom - tv)}, "
        f"tv-only={sorted(tv - bedroom)}")


def test_no_sample_is_silently_dropped() -> None:
    """Distinct samples get distinct buckets.

    Two samples colliding is not only a gap elsewhere: the merge does
    `buckets[key].update(row)`, so the second one overwrites the first
    and that observation is gone.
    """
    assert len({_bucket(t) for t in TV_ROOM}) == len(TV_ROOM)
    assert len({_bucket(t) for t in BEDROOM}) == len(BEDROOM)


def test_the_tie_case_directly() -> None:
    """A stamp exactly on the half-period boundary."""
    assert _bucket(1787635395) != _bucket(1787635405), (
        "two samples a full period apart must never share a bucket")


def test_samples_inside_one_window_still_merge() -> None:
    """The whole point: same 10s window, same row."""
    assert _bucket(1787635390) == _bucket(1787635399)
