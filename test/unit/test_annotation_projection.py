"""Placing an episode in the footage.

The join is arithmetic, not a stored relationship: an episode lives on a
wall-clock timeline and a clip is a window on that same timeline. It only
works because clip duration became truthful (2026-08-04) — before that a
clip claiming 871s covered 906s of real life.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.services.annotations.projection import ClipSpan, project

_T0 = datetime(2026, 8, 5, 22, 0, 0, tzinfo=timezone.utc)


def _at(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _clip(key: str, start_s: float, dur: float, exact: bool = True) -> ClipSpan:
    return ClipSpan(key=key, started_at=_at(start_s), duration_seconds=dur,
                    timeline_exact=exact)


def test_episode_inside_one_clip() -> None:
    p = project(_at(30), _at(45), [_clip("a", 0, 900)])
    assert len(p.segments) == 1
    seg = p.segments[0]
    assert (seg.start_offset, seg.end_offset) == (30.0, 45.0)
    assert p.is_complete


def test_episode_spanning_several_rolls() -> None:
    """A night of sleep crosses many 15-minute clips. Each one gets its
    own segment, with offsets measured from that clip's own start."""
    clips = [_clip(f"c{i}", i * 900, 900) for i in range(4)]
    p = project(_at(850), _at(2000), clips)

    assert [s.key for s in p.segments] == ["c0", "c1", "c2"]
    assert p.segments[0].start_offset == 850.0
    assert p.segments[1].start_offset == 0.0     # starts at c1's beginning
    assert p.segments[2].end_offset == 200.0     # 2000 - 1800
    assert p.is_complete


def test_a_gap_with_no_recording_is_reported() -> None:
    """Recording is gated on presence, so an episode routinely covers
    stretches with no footage. Silently returning only what exists would
    make four minutes look like a whole night."""
    clips = [_clip("a", 0, 100), _clip("b", 400, 100)]
    p = project(_at(50), _at(450), clips)

    assert not p.is_complete
    assert p.gaps == ((_at(100), _at(400)),)
    assert p.covered_seconds == 100.0  # 50 from a, 50 from b


def test_an_episode_with_no_footage_at_all() -> None:
    p = project(_at(50), _at(80), [_clip("a", 500, 100)])
    assert p.segments == ()
    assert p.gaps == ((_at(50), _at(80)),)


def test_still_running_episode_projects_up_to_now() -> None:
    """ended_at is NULL all night for sleep; it must still place."""
    p = project(_at(30), None, [_clip("a", 0, 900)], now=_at(120))
    assert len(p.segments) == 1
    assert p.segments[0].end_offset == 120.0


def test_overlapping_clips_do_not_double_count() -> None:
    """Roll boundaries can overlap by a moment; the same wall-clock must
    not appear twice."""
    p = project(_at(0), _at(200), [_clip("a", 0, 120), _clip("b", 100, 120)])
    assert p.covered_seconds == 200.0
    assert p.is_complete
    assert p.segments[1].start_offset == 20.0  # b resumes where a stopped


def test_unreliable_clips_are_flagged_through() -> None:
    """A clip the recorder could not keep up with has approximate
    offsets. The caller has to be able to see that."""
    p = project(_at(10), _at(20), [_clip("a", 0, 900, exact=False)])
    assert p.segments[0].timeline_exact is False


def test_zero_length_and_inverted_spans_are_ignored() -> None:
    assert project(_at(50), _at(50), [_clip("a", 0, 900)]).segments == ()
    assert project(_at(60), _at(50), [_clip("a", 0, 900)]).segments == ()
    assert project(_at(10), _at(20), [_clip("a", 0, 0)]).segments == ()
