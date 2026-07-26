"""Spine batching — contiguous, never front-drops, evenly decimated.

The 2026-07-23 lesson: `new[-N:]` (keeping only the tail while the pod
catches up) silently opens blind gaps between windows → the tracker
renumbers on any movement. plan_batch must select frames spread across
the WHOLE unshipped interval, and every batch must start with the
previous batch's final frame (the carry-seed overlap).
"""

from __future__ import annotations

from src.services.identity.batching import plan_batch


def _frames(n: int, fps: float = 15.0, t0: float = 100.0):
    """[(idx, ts), ...] — a steady camera delivering n frames."""
    return [(i, t0 + i / fps) for i in range(n)]


def test_first_batch_takes_from_start_no_overlap_seed():
    frames = _frames(12)
    sel = plan_batch(frames, last_shipped=None, target_fps=15.0, max_batch=32)
    assert sel[0] == frames[0]
    assert sel[-1] == frames[-1]


def test_never_front_drops_when_backlog_grows():
    # Pod stalled: 60 unshipped frames (4s at 15fps). The selection must
    # SPAN the backlog (oldest unshipped is included via decimation),
    # not just take the newest tail.
    frames = _frames(60)
    sel = plan_batch(frames, last_shipped=None, target_fps=15.0, max_batch=10)
    idxs = [i for i, _ in sel]
    assert len(sel) <= 10
    assert idxs[0] == 0                      # front preserved
    assert idxs[-1] == 59                    # newest included
    # roughly even spread: no gap more than ~2x the ideal stride
    ideal = 59 / (len(idxs) - 1)
    gaps = [b - a for a, b in zip(idxs, idxs[1:])]
    assert max(gaps) <= 2 * ideal + 1


def test_overlap_seed_is_previous_final_frame():
    frames = _frames(30)
    first = plan_batch(frames[:15], last_shipped=None, target_fps=15.0, max_batch=32)
    last = first[-1]
    second = plan_batch(
        frames, last_shipped=last, target_fps=15.0, max_batch=32,
    )
    # Batch 2 starts with the exact frame batch 1 ended on — EdgeTAM's
    # carry seed matches its first frame.
    assert second[0] == last
    # ...and continues strictly forward from there.
    assert [i for i, _ in second[1:]] == [i for i, _ in frames[15:]]


def test_decimates_to_target_fps():
    frames = _frames(30, fps=15.0)           # 2s of 15fps
    sel = plan_batch(frames, last_shipped=None, target_fps=5.0, max_batch=32)
    # ~2s at 5fps → ~10-11 frames, never the full 30
    assert 8 <= len(sel) <= 12


def test_no_new_frames_returns_empty():
    frames = _frames(10)
    last = frames[-1]
    assert plan_batch(frames, last_shipped=last, target_fps=15.0, max_batch=8) == []


def test_empty_input():
    assert plan_batch([], last_shipped=None, target_fps=15.0, max_batch=8) == []


def test_max_batch_one_takes_newest_only():
    frames = _frames(30)
    sel = plan_batch(frames, last_shipped=None, target_fps=15.0, max_batch=1)
    assert sel == [frames[-1]]
