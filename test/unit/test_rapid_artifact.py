"""Unit tests for the RAPID artifact builder (detection v2, P0).

The validated recipe: motion overlay computed on NATIVE full frames first,
then a STATIC per-person crop (union of the person's boxes, padded,
even-aligned), encoded at native fps, video-only. A moving crop window
would frame-diff stationary scenery into phantom red — hence static union.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.motion_overlay import overlay_motion
from src.services.detection.rapid_artifact import (
    build_rapid_artifacts,
    crop_i420,
    union_box,
)

W, H = 160, 120
_FPS = 8.0


def _i420(y: np.ndarray, u_val: int = 128, v_val: int = 128) -> bytes:
    h, w = y.shape
    u = np.full((h // 2, w // 2), u_val, dtype=np.uint8)
    v = np.full((h // 2, w // 2), v_val, dtype=np.uint8)
    return y.tobytes() + u.tobytes() + v.tobytes()


def _flicker_frames(n: int = 8, *, x0: int = 60, y0: int = 40, size: int = 24
                    ) -> list[FrameSnapshot]:
    frames = []
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        y[y0:y0 + size, x0:x0 + size] = 190 if i % 2 else 60
        frames.append(FrameSnapshot(
            width=W, height=H, data=_i420(y), captured_at=100.0 + i / _FPS))
    return frames


def test_union_box_pads_clamps_and_even_aligns() -> None:
    box = union_box([(10, 10, 50, 50), (20, 20, 70, 60)], W, H)
    assert box is not None
    x0, y0, x1, y1 = box
    # Union is (10,10,70,60); 20% pad pushes past the left edge → clamp to 0.
    assert x0 == 0 and y0 == 0
    assert x1 > 70 and y1 > 60          # padding applied on the far side
    assert all(v % 2 == 0 for v in box)  # I420 even alignment
    assert x1 <= W and y1 <= H


def test_union_box_enforces_min_side_and_empty() -> None:
    assert union_box([], W, H) is None
    box = union_box([(80, 60, 84, 64)], W, H, min_side=64)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 - x0 >= 64 and y1 - y0 >= 64


def test_crop_i420_slices_all_three_planes() -> None:
    y = np.full((H, W), 60, dtype=np.uint8)
    y[20:84, 40:104] = 200
    frame = FrameSnapshot(
        width=W, height=H, data=_i420(y, u_val=90, v_val=210), captured_at=1.0)
    (crop,) = crop_i420([frame], (40, 20, 104, 84))
    assert crop.width == 64 and crop.height == 64
    yc = np.frombuffer(crop.data[: 64 * 64], dtype=np.uint8).reshape(64, 64)
    assert int(yc.min()) == 200 and int(yc.max()) == 200
    uc = np.frombuffer(crop.data[64 * 64: 64 * 64 + 32 * 32], dtype=np.uint8)
    vc = np.frombuffer(crop.data[64 * 64 + 32 * 32:], dtype=np.uint8)
    assert uc.size == 32 * 32 and int(uc[0]) == 90
    assert vc.size == 32 * 32 and int(vc[0]) == 210


def test_crop_i420_skips_mismatched_shapes() -> None:
    """Reference shape is the LAST frame (current publish state after a
    renegotiation) — an odd-shaped frame at index 0 must be the one dropped,
    not the majority."""
    good = _flicker_frames(n=2)
    odd = FrameSnapshot(width=40, height=30,
                        data=_i420(np.full((30, 40), 60, dtype=np.uint8)),
                        captured_at=99.9)
    out = crop_i420([odd, good[0], good[1]], (40, 20, 104, 84))
    assert len(out) == 2
    out2 = crop_i420([good[0], odd, good[1]], (40, 20, 104, 84))
    assert len(out2) == 2


def test_union_box_exact_arithmetic_pin() -> None:
    """One hand-computed case pinning pad→clamp→even order exactly:
    box (20,20,60,60) in 160x120, pad 20% of 40 = 8 each side →
    (12,12,68,68), already ≥ min_side 40, all even."""
    assert union_box([(20, 20, 60, 60)], W, H, pad_ratio=0.2, min_side=40) == \
        (12, 12, 68, 68)


def test_union_box_corner_and_tiny_frame() -> None:
    # Box in the far corner: min-side growth + slide keeps it inside.
    box = union_box([(150, 112, 158, 118)], W, H, min_side=64)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x1 <= W and y1 <= H and x0 >= 0 and y0 >= 0
    assert x1 - x0 >= 64 and y1 - y0 >= 64
    # Frame smaller than min_side: saturates to the full frame, no crash.
    tiny = union_box([(2, 2, 30, 20)], 40, 30, min_side=64)
    assert tiny == (0, 0, 40, 30)


async def test_encode_failure_skips_person_not_tick(monkeypatch) -> None:
    """One person's ffmpeg failure must not discard the other's artifact."""
    import src.services.detection.rapid_artifact as ra

    calls = {"n": 0}

    async def flaky(frames, *, fps, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ra.Mp4RemuxError("boom")
        return b"\x00\x00\x00\x18ftypfake"

    monkeypatch.setattr(ra, "frames_to_mp4", flaky)
    arts = await build_rapid_artifacts(
        _flicker_frames(n=8),
        [[(50, 30, 100, 90)], [(0, 0, 40, 60)]],
        fps=8,
    )
    assert len(arts) == 1
    assert calls["n"] == 2


def test_tint_survives_static_crop() -> None:
    """The composition the builder performs: overlay on native frames, then
    the static crop — the red chroma from the in-box flicker must be present
    in the cropped frames (V pushed toward 240; blend lands ~195)."""
    tinted = overlay_motion(_flicker_frames())
    cropped = crop_i420(tinted, (40, 20, 120, 100))
    cw, ch = cropped[0].width, cropped[0].height
    v_max = 0
    for f in cropped[1:]:
        ysz, csz = cw * ch, (cw // 2) * (ch // 2)
        v = np.frombuffer(f.data[ysz + csz:], dtype=np.uint8)
        v_max = max(v_max, int(v.max()))
    assert v_max > 180


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
async def test_build_rapid_artifacts_one_mp4_per_person() -> None:
    frames = _flicker_frames(n=8)
    persons = [
        [(50, 30, 100, 90)] * 8,               # around the flicker block
        [(0, 0, 40, 60), (2, 2, 42, 62)],      # a second, separate person
    ]
    arts = await build_rapid_artifacts(frames, persons, fps=8)
    assert len(arts) == 2
    for mp4 in arts:
        assert mp4[4:8] == b"ftyp"
        assert len(mp4) > 400


async def test_build_rapid_artifacts_empty_inputs() -> None:
    assert await build_rapid_artifacts([], [[(0, 0, 10, 10)]], fps=8) == []
    assert await build_rapid_artifacts(_flicker_frames(), [], fps=8) == []
    assert await build_rapid_artifacts(_flicker_frames(), [[]], fps=8) == []
