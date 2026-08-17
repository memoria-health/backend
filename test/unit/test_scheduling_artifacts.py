"""Unit: person-crop artifact builder (detection v2 P3 T2).

The crop GEOMETRY (pad/min-side/clamp/plane-slicing) is already pinned
by the P0 rapid_artifact tests — these tests pin what's new: the
extracted→native scaling seam, the None fallbacks, and one real
end-to-end remux.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest

import src.services.detection.scheduling.artifacts as artifacts
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.scheduling.artifacts import (
    build_person_crop_mp4,
    person_crop_box,
)
from src.services.identity.observations import TrackObservation

W, H = 160, 120


def _frame(t: float) -> FrameSnapshot:
    y = np.full((H, W), 60, dtype=np.uint8)
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return FrameSnapshot(
        width=W, height=H, data=y.tobytes() + uv.tobytes(), captured_at=t,
    )


def _window(n: int = 8) -> list[FrameSnapshot]:
    t0 = time.monotonic() - n / 4.0
    return [_frame(t0 + i / 4.0) for i in range(n)]


def _track(series) -> TrackObservation:
    return TrackObservation(
        track_id=1, label="UNKNOWN", confidence=0.0,
        bbox_series=tuple(series), face_locked=False, pre_labeled=False,
    )


def test_box_scales_extracted_space_to_native() -> None:
    # Boxes recorded on 80px-wide extracted frames; native is 160 ⇒ ×2.
    track = _track([(0, (10.0, 10.0, 30.0, 50.0)), (1, (12.0, 10.0, 32.0, 50.0))])
    box = person_crop_box(_window(), track=track, sample_width=80)
    assert box is not None
    x0, y0, x1, y1 = box
    # Union in native px is (20,20,64,100) + 20% pad, min-side 64,
    # even-aligned — bounds prove the ×2 scaling happened (an unscaled
    # box would sit left of x=35).
    assert x1 > 60 and y1 > 90
    assert x0 % 2 == 0 and y0 % 2 == 0 and x1 % 2 == 0 and y1 % 2 == 0
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H


def test_no_boxes_or_empty_window_yield_none() -> None:
    assert person_crop_box([], track=_track([(0, (1, 1, 9, 9))]), sample_width=480) is None
    assert person_crop_box(_window(), track=_track([]), sample_width=480) is None


@pytest.mark.asyncio
async def test_builder_returns_none_when_no_crop(monkeypatch) -> None:
    out = await build_person_crop_mp4(
        _window(), track=_track([]), sample_width=480, fps=4,
    )
    assert out is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
@pytest.mark.asyncio
async def test_builder_produces_playable_mp4() -> None:
    track = _track([(i, (20.0, 20.0, 100.0, 100.0)) for i in range(8)])
    out = await build_person_crop_mp4(
        _window(), track=track, sample_width=480, fps=4,
    )
    assert out is not None and len(out) > 200
    assert b"ftyp" in out[:64]


@pytest.mark.asyncio
async def test_remux_failure_degrades_to_none(monkeypatch) -> None:
    async def boom(*a, **k):
        raise artifacts.Mp4RemuxError("no frames")

    monkeypatch.setattr(artifacts, "frames_to_mp4", boom)
    track = _track([(0, (20.0, 20.0, 100.0, 100.0))])
    out = await build_person_crop_mp4(
        _window(), track=track, sample_width=480, fps=4,
    )
    assert out is None


# ── _shrink_i420 / bounded-res tint (OOM guard, soak 2026-07-14) ─────


def _big_frame(w: int = 1280, h: int = 720) -> FrameSnapshot:
    y = np.full((h, w), 60, dtype=np.uint8)
    uv = np.full((h // 2) * (w // 2) * 2, 128, dtype=np.uint8)
    return FrameSnapshot(
        width=w, height=h, data=y.tobytes() + uv.tobytes(),
        captured_at=time.monotonic(),
    )


def test_shrink_noop_at_or_below_max_width() -> None:
    f = _frame(time.monotonic())          # 160w — already small
    assert artifacts._shrink_i420(f, 640) is f


def test_shrink_downscales_even_aligned_i420() -> None:
    f = _big_frame(1280, 720)
    s = artifacts._shrink_i420(f, 640)
    assert (s.width, s.height) == (640, 360)
    assert len(s.data) == 640 * 360 * 3 // 2      # valid I420 payload
    assert s.captured_at == f.captured_at


async def test_tint_window_bounds_native_resolution() -> None:
    tinted = await artifacts.tint_window([_big_frame() for _ in range(4)])
    assert tinted and all(t.width <= artifacts._TINT_MAX_WIDTH for t in tinted)
    assert all(len(t.data) == t.width * t.height * 3 // 2 for t in tinted)


# ── bbox coordinate space (2026-08-07 regression) ────────────────────
# The sidecar identity provider returns boxes in NATIVE pixels; the
# retired in-process reid brain returned them downscaled to
# `reid_inference_sample_width`. For ~24h the consumer applied the
# downscaled assumption to native boxes and multiplied every one by
# native/sample_width. On a 1920x1080 frame with sample_width=640 that
# is 3x: a patient standing at x=333..666 produced the crop
# (520, 0, 1920, 1080) — their legs plus 1400px of empty room — and
# every person-bucket detector answered "no person is visible" while
# the uncropped scene call described the patient plainly.
#
# These pin BOTH grids, because the bug was not a wrong constant: it
# was a consumer that could only guess which grid it was handed.

def _native_track(series) -> TrackObservation:
    from src.services.identity.observations import BBOX_SPACE_NATIVE
    return TrackObservation(
        track_id=1, label="PATIENT", confidence=0.9,
        bbox_series=tuple(series), face_locked=True, pre_labeled=False,
        bbox_space=BBOX_SPACE_NATIVE,
    )


def test_native_bbox_is_not_rescaled() -> None:
    """A native-space box must land on the person, not 3x away."""
    window = _window()
    box = (40.0, 30.0, 70.0, 90.0)           # already in 160x120 space
    got = person_crop_box(
        window, track=_native_track([(0, box)]), sample_width=40,
    )
    assert got is not None
    x0, y0, x1, y1 = got
    # Pad/min-side may GROW the box; it must never slide off the
    # subject. The true box has to sit inside the crop.
    assert x0 <= box[0] and y0 <= box[1]
    assert x1 >= box[2] and y1 >= box[3]


def test_extracted_bbox_still_scales_up() -> None:
    """The legacy grid keeps its exact behaviour — 40w -> 160w is 4x."""
    window = _window()                        # native 160x120
    box = (10.0, 5.0, 20.0, 25.0)             # in a 40px-wide grid
    got = person_crop_box(
        window, track=_track([(0, box)]), sample_width=40,
    )
    assert got is not None
    x0, _, x1, _ = got
    # 4x scaling puts the box at x=40..80; pad only widens it.
    assert x0 <= 40 and x1 >= 80


def test_native_and_extracted_disagree_when_grids_differ() -> None:
    """The tag must actually change the answer — otherwise it is
    decoration and the next provider swap breaks silently again."""
    window = _window()
    box = (40.0, 30.0, 70.0, 90.0)
    native = person_crop_box(
        window, track=_native_track([(0, box)]), sample_width=40,
    )
    extracted = person_crop_box(
        window, track=_track([(0, box)]), sample_width=40,
    )
    assert native != extracted
