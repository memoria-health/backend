"""CameraState — track lifecycle from per-batch pod observations.

The pod's persistent session carries stable ids; the state mirror just
tracks births, refreshes, and deaths (a track absent from a batch's
output is gone — SAM3 is the authority on existence, and the session
clears departed masks). Solo = exactly one live track on the camera.
"""

from __future__ import annotations

from identity.state import MISSES_BEFORE_DEATH, CameraState


def _batch(tracks: dict[int, tuple[float, float, float, float]]):
    """tid -> bbox for one batch result."""
    return tracks


def test_birth_and_refresh():
    st = CameraState(camera_id="cam-a")
    events = st.apply(_batch({7: (0, 0, 50, 100)}), now=100.0)
    assert events.births == [7]
    assert st.tracks[7].bbox == (0, 0, 50, 100)
    assert st.tracks[7].born_at == 100.0

    events = st.apply(_batch({7: (10, 0, 60, 100)}), now=101.0)
    assert events.births == [] and events.deaths == []
    assert st.tracks[7].bbox == (10, 0, 60, 100)
    assert st.tracks[7].born_at == 100.0      # birth time survives refresh
    assert st.tracks[7].batches_seen == 2


def test_death_on_sustained_absence():
    """Death takes MISSES_BEFORE_DEATH consecutive absences, not one.

    Was one. That single batch was the loudest source of instability in
    the system: the seat, the identity lock and every consumer below
    them are keyed on the track id, so one noisy batch destroyed a face
    match that had cost 20-60s to earn (measured 2026-08-25).
    """
    st = CameraState(camera_id="cam-a")
    st.apply(_batch({7: (0, 0, 50, 100), 9: (60, 0, 110, 100)}), now=100.0)
    events = st.apply(_batch({9: (65, 0, 115, 100)}), now=101.0)
    assert events.deaths == [], "one absent batch must not be a death"
    assert 7 in st.tracks, "the track is carried through the gap"
    for i in range(MISSES_BEFORE_DEATH - 1):
        events = st.apply(_batch({9: (65, 0, 115, 100)}), now=102.0 + i)
    assert events.deaths == [7]
    assert 7 not in st.tracks and 9 in st.tracks


def test_empty_batches_eventually_kill_all():
    """An empty room still empties, it just takes the full count."""
    st = CameraState(camera_id="cam-a")
    st.apply(_batch({7: (0, 0, 50, 100)}), now=100.0)
    for i in range(MISSES_BEFORE_DEATH):
        events = st.apply(_batch({}), now=101.0 + i)
    assert events.deaths == [7]
    assert st.tracks == {}


def test_solo_detection():
    st = CameraState(camera_id="cam-a")
    st.apply(_batch({7: (0, 0, 50, 100)}), now=100.0)
    assert st.solo
    st.apply(_batch({7: (0, 0, 50, 100), 9: (60, 0, 110, 100)}), now=101.0)
    assert not st.solo
    st.apply(_batch({}), now=102.0)
    assert not st.solo                        # empty is not solo


def test_track_age():
    st = CameraState(camera_id="cam-a")
    st.apply(_batch({7: (0, 0, 50, 100)}), now=100.0)
    st.apply(_batch({7: (0, 0, 50, 100)}), now=104.0)
    assert st.tracks[7].age(now=105.0) == 5.0
