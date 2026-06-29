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


def _push_pre_then_post_roll(
    buf: FrameBuffer, *, window: float, post_roll: float, fps: int
) -> float:
    """Drive `buf` exactly like the live tick → post-roll sequence:
    push frames at `fps` from t=0 through the event (pre-end at t=window)
    and keep recording through the post-roll to t=window+post_roll.
    Returns the pre-end timestamp (last frame the pre.mp4 snapshot saw).
    """
    step = 1.0 / fps
    end = window + post_roll
    t = 0.0
    # Tiny payload — FrameBuffer stores bytes verbatim and never inspects
    # size (only frames_to_mp4 validates I420 dimensions).
    blob = _blob(2, 2)
    while t <= end + 1e-9:
        buf.push(width=2, height=2, data=blob, captured_at=round(t, 6))
        t += step
    return window


def test_post_roll_snapshot_is_contiguous_with_pre_end() -> None:
    """Seam regression (the prod config: 11s analysis window, 15s post-roll).

    clip_archival builds the incident clip by concatenating pre.mp4 (the
    last `window` seconds at tick time) with post.mp4 (snapshot_last_seconds
    `post_roll` taken `post_roll` after the pre window's last frame). For
    the concat to be seamless, the post snapshot's first frame must be
    contiguous with the pre window's last frame.

    With the DERIVED retention (max(window, post_roll)+margin), it is.
    """
    from src.settings import get_settings

    fps, window, post_roll = 4, 11.0, 15.0
    s = get_settings().model_copy(
        update={
            "livekit_worker_window_seconds": int(window),
            "clip_post_event_wait_seconds": post_roll,
            "livekit_worker_target_fps": fps,
        }
    )
    retention = s.frame_buffer_retention_seconds  # 17.0
    buf = FrameBuffer(window_seconds=retention, target_fps=fps)
    pre_end = _push_pre_then_post_roll(buf, window=window, post_roll=post_roll, fps=fps)

    # Anchored exactly as clip_archival does: now = pre_end + post_roll.
    post = buf.snapshot_last_seconds(post_roll, now=pre_end + post_roll)
    assert post, "post-roll snapshot must not be empty"
    step = 1.0 / fps
    gap = post[0].captured_at - pre_end
    # Contiguous: first post frame is at (or one frame after) pre's end —
    # NOT several seconds later. Pre-fix this gap was ~4s (post_roll-window).
    assert gap <= step + 1e-9, (
        f"seam gap of {gap:.2f}s: pre ended at {pre_end}, "
        f"post starts at {post[0].captured_at}"
    )


def test_old_window_sized_retention_drops_post_roll_head() -> None:
    """Documents the bug the derived retention fixes: sizing the buffer to
    only the analysis window (11s) — what shipped — evicts the first ~4s of
    a 15s post-roll, which is the visible jump at the concat seam.
    """
    fps, window, post_roll = 4, 11.0, 15.0
    buf = FrameBuffer(window_seconds=window, target_fps=fps)  # OLD behavior
    pre_end = _push_pre_then_post_roll(buf, window=window, post_roll=post_roll, fps=fps)

    post = buf.snapshot_last_seconds(post_roll, now=pre_end + post_roll)
    assert post, "post-roll snapshot must not be empty"
    gap = post[0].captured_at - pre_end
    # The hole: post can't start before (pre_end + post_roll - window).
    assert gap >= (post_roll - window) - 1e-6  # ~4s gap, the seam jump


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
