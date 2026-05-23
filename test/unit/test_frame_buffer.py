"""Unit tests for `FrameBuffer`.

Pure data structure — these tests do not touch LiveKit at all. Every
frame is a synthetic ``b"\x00" * (1.5 * w * h)`` blob with an
explicit monotonic timestamp. The buffer's correctness lives entirely
in three behaviors:

  1. The time-based downsampler accepts frames at no more than
     `target_fps`, dropping anything that arrives sooner than the
     minimum gap.
  2. The time-based trim removes frames older than `window_seconds`
     relative to the most-recently-pushed frame.
  3. `snapshot_last_seconds` returns the trailing window relative to
     an explicitly-pinned `now`.
"""

from __future__ import annotations

import pytest

from src.services.detection.frame_buffer import FrameBuffer, FrameSnapshot


def _blob(width: int = 16, height: int = 16) -> bytes:
    """A throwaway I420 byte string of the right size. The buffer
    doesn't inspect contents — only length-by-reference matters for
    the test's purposes (verifying we held onto the same bytes).
    """
    return b"\x00" * (width * height * 3 // 2)


def test_init_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError):
        FrameBuffer(window_seconds=0, target_fps=5)


def test_init_rejects_nonpositive_fps() -> None:
    with pytest.raises(ValueError):
        FrameBuffer(window_seconds=15, target_fps=0)


def test_first_frame_is_always_accepted() -> None:
    buf = FrameBuffer(window_seconds=15, target_fps=5)
    accepted = buf.push(width=16, height=16, data=_blob(), captured_at=100.0)
    assert accepted is True
    assert len(buf) == 1


def test_frames_within_min_gap_are_dropped() -> None:
    """Min gap at 5fps is 0.2s. A frame 0.15s after the previous
    one must be dropped; a frame 0.25s after must be accepted.
    """
    buf = FrameBuffer(window_seconds=15, target_fps=5)
    assert buf.push(width=16, height=16, data=_blob(), captured_at=100.0) is True
    assert buf.push(width=16, height=16, data=_blob(), captured_at=100.15) is False
    assert buf.push(width=16, height=16, data=_blob(), captured_at=100.25) is True
    assert len(buf) == 2


def test_frames_older_than_window_get_trimmed_on_push() -> None:
    """Each push call drops frames older than window_seconds relative
    to the freshly-accepted frame's timestamp.
    """
    buf = FrameBuffer(window_seconds=2.0, target_fps=10)
    # Pump in 5 frames at 100.0, 100.2, 100.4, 100.6, 100.8.
    for i in range(5):
        accepted = buf.push(width=16, height=16, data=_blob(), captured_at=100.0 + 0.2 * i)
        assert accepted is True
    assert len(buf) == 5

    # A frame at 103.0 makes the cutoff 101.0; all earlier frames drop.
    buf.push(width=16, height=16, data=_blob(), captured_at=103.0)
    assert len(buf) == 1
    only = buf.snapshot_last_seconds(10, now=103.0)
    assert len(only) == 1
    assert only[0].captured_at == pytest.approx(103.0)


def test_snapshot_returns_only_within_window_and_oldest_first() -> None:
    buf = FrameBuffer(window_seconds=15, target_fps=5)
    # Five frames, evenly spaced 0.5s apart from t=100.0.
    for i in range(5):
        buf.push(width=16, height=16, data=_blob(), captured_at=100.0 + 0.5 * i)
    # At now=102.0, the cutoff for a 1s window is 101.0 → frames at
    # 101.0 and 101.5 (i=2, i=3) and 102.0 (i=4) qualify; 100.0 and
    # 100.5 do not.
    window = buf.snapshot_last_seconds(1.0, now=102.0)
    assert [f.captured_at for f in window] == pytest.approx([101.0, 101.5, 102.0])


def test_snapshot_with_nonpositive_seconds_returns_empty() -> None:
    buf = FrameBuffer(window_seconds=15, target_fps=5)
    buf.push(width=16, height=16, data=_blob(), captured_at=100.0)
    assert buf.snapshot_last_seconds(0) == []
    assert buf.snapshot_last_seconds(-1) == []


def test_capacity_is_window_times_fps_plus_one() -> None:
    """The plan promises a hard memory bound; this test pins the
    capacity formula so a future refactor can't silently bloat the
    buffer.
    """
    buf = FrameBuffer(window_seconds=15, target_fps=5)
    assert buf.capacity == 76  # 15 * 5 + 1


def test_continuous_pushing_stays_within_window() -> None:
    """Push at the target rate for much longer than the window and
    verify the buffer length stabilizes at ~window_seconds * target_fps.
    This is the real memory-bound assertion (the deque's `maxlen` is
    just a defense-in-depth cap for impossible cases).
    """
    buf = FrameBuffer(window_seconds=5.0, target_fps=10)
    # Push at 1.5x the target gap so each push is accepted by the
    # fps gate (no floating-point ambiguity at the boundary).
    gap = 0.15  # 50% headroom over the 0.1s min gap at 10 fps
    for i in range(200):
        buf.push(width=16, height=16, data=_blob(), captured_at=100.0 + gap * i)
    # Steady state: each push trims anything older than `now - 5s`.
    # At 1/0.15 ≈ 6.67 accepted-frames per second × 5s ≈ 33-34 frames.
    assert 30 <= len(buf) <= 35


def test_data_is_stored_as_is_not_referenced() -> None:
    """The buffer holds the exact bytes the caller passed in.
    Important because the SDK's frame.data memoryview is invalidated
    after the next frame — the room-worker must copy to bytes before
    push, and the buffer's storage must preserve that copy.
    """
    buf = FrameBuffer(window_seconds=15, target_fps=5)
    sentinel = b"\x42" * 384  # 16x16 I420
    buf.push(width=16, height=16, data=sentinel, captured_at=100.0)
    frames = buf.snapshot_last_seconds(15, now=100.0)
    assert len(frames) == 1
    assert frames[0].data is sentinel    # identity, not just equality
    assert frames[0].width == 16
    assert frames[0].height == 16
    assert isinstance(frames[0], FrameSnapshot)
