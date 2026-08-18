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


def test_rects_follow_a_moving_person() -> None:
    # Person walks left -> right across ~2s of node time.
    series = [(200.0 + i * 0.25, (100 + i * 150, 400, 220 + i * 150, 800))
              for i in range(8)]
    frames = _frames()
    plan = plan_follow_rects(frames, _track(series), sample_width=640)
    assert plan is not None
    rects, (ow, oh) = plan
    assert len(rects) == len(frames)
    assert ow % 2 == 0 and oh % 2 == 0 and ow >= 2 and oh >= 2
    centers = [(r[0] + r[2]) / 2 for r in rects]
    assert centers == sorted(centers)             # pans with the person
    assert centers[-1] - centers[0] > 300         # actually moved
    aspect = ow / oh
    for r in rects:
        assert 0 <= r[0] and r[2] <= 1920 and 0 <= r[1] and r[3] <= 1080
        assert r[0] % 2 == 0 and r[1] % 2 == 0
        # Every rect keeps the output aspect (distortion-free zoom).
        assert abs(((r[2] - r[0]) / (r[3] - r[1])) - aspect) < 0.08


def test_zoom_tightens_when_person_shrinks() -> None:
    # Near (huge box) then far (small box): late rects must be smaller
    # than the max box — the old fixed-dims viewport kept them equal.
    series = ([(200.0 + i * 0.25, (400, 100, 1400, 1000)) for i in range(4)]
              + [(201.0 + i * 0.25, (900, 400, 1020, 700)) for i in range(4)])
    frames = _frames()
    plan = plan_follow_rects(frames, _track(series), sample_width=640)
    assert plan is not None
    rects, (ow, oh) = plan
    first_area = (rects[0][2] - rects[0][0]) * (rects[0][3] - rects[0][1])
    last_area = (rects[-1][2] - rects[-1][0]) * (rects[-1][3] - rects[-1][1])
    assert last_area < first_area * 0.7           # actually zoomed in
    assert ow >= rects[-1][2] - rects[-1][0]      # upscale target


def test_crop_follow_resizes_every_frame_to_output_dims() -> None:
    # Real I420 bytes through the actual crop+zoom path — geometry
    # tests alone missed a missing import in the resize half.
    from src.services.detection.scheduling.artifacts import _crop_follow
    w, h = 64, 48
    data = bytes(w * h) + bytes(w * h // 4) * 2
    frames = [FrameSnapshot(width=w, height=h, data=data,
                            captured_at=100.0 + i * 0.25) for i in range(4)]
    rects = [(0, 0, 32, 24), (8, 8, 40, 32), (16, 8, 48, 32), (0, 0, 64, 48)]
    out = _crop_follow(frames, rects, (32, 24))
    assert len(out) == 4
    assert all((f.width, f.height) == (32, 24) for f in out)


def test_uncovered_window_degrades_whole_call() -> None:
    # Series covers only the last second of an 8-frame window.
    series = [(200.0, (500, 400, 620, 800))]
    frames = _frames(n=8, dt=1.5)                 # 10.5s window
    assert plan_follow_rects(frames, _track(series), sample_width=640) is None


def test_single_fresh_box_still_crops_short_window() -> None:
    series = [(200.0, (500, 400, 620, 800))]
    frames = _frames(n=6, dt=0.25)                # 1.25s window
    plan = plan_follow_rects(frames, _track(series), sample_width=640)
    assert plan is not None
    rects, _dims = plan
    assert len(set(rects)) == 1
