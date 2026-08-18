"""plan_follow_rects — the follow-crop geometry (pure)."""

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.scheduling.artifacts import plan_follow_rects
from src.services.identity.observations import (
    BBOX_SPACE_NATIVE, TrackObservation,
)


def _frames(n=8, w=1920, h=1080, dt=0.25):
    return [FrameSnapshot(width=w, height=h, data=b"x" * 32,
                          captured_at=100.0 + i * dt) for i in range(n)]


def _track(series):
    return TrackObservation(
        track_id=1, label="UNKNOWN", confidence=0.0,
        bbox_series=tuple(series), face_locked=False, pre_labeled=False,
        bbox_space=BBOX_SPACE_NATIVE,
    )


def test_rects_follow_a_moving_person_with_constant_dims() -> None:
    # Person walks left -> right across ~2s of node time.
    series = [(200.0 + i * 0.25, (100 + i * 150, 400, 220 + i * 150, 800))
              for i in range(8)]
    frames = _frames()
    rects = plan_follow_rects(frames, _track(series), sample_width=640)
    assert rects is not None and len(rects) == len(frames)
    ws = {r[2] - r[0] for r in rects}
    hs = {r[3] - r[1] for r in rects}
    assert len(ws) == 1 and len(hs) == 1          # constant output dims
    centers = [(r[0] + r[2]) / 2 for r in rects]
    assert centers == sorted(centers)             # pans with the person
    assert centers[-1] - centers[0] > 300         # actually moved
    for r in rects:
        assert 0 <= r[0] and r[2] <= 1920 and 0 <= r[1] and r[3] <= 1080
        assert r[0] % 2 == 0 and r[1] % 2 == 0


def test_uncovered_window_degrades_whole_call() -> None:
    # Series covers only the last second of an 8-frame window.
    series = [(200.0, (500, 400, 620, 800))]
    frames = _frames(n=8, dt=1.5)                 # 10.5s window
    assert plan_follow_rects(frames, _track(series), sample_width=640) is None


def test_single_fresh_box_still_crops_short_window() -> None:
    series = [(200.0, (500, 400, 620, 800))]
    frames = _frames(n=6, dt=0.25)                # 1.25s window
    rects = plan_follow_rects(frames, _track(series), sample_width=640)
    assert rects is not None and len(set(rects)) == 1
