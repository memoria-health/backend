"""Unit tests for the RAPID pre-gate signal (detection v2, P0).

`rapid_energy` must be a cheap, model-free discriminator between
sustained IN-PLACE rapid motion (convulsion-like: the same pixels churn
in most consecutive frame diffs) and everything cheaper to reject:
stillness, slow drift, and TRANSLATING motion (walking — lots of change,
but each pixel is active in only a couple of frame pairs).

Frames are synthetic I420 blobs: a Y plane we control + flat UV planes.
All sequences simulate the 8 fps native rate the buffer retains when the
RAPID path is on.
"""

from __future__ import annotations

import numpy as np

import pytest

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.signals.rapid_energy import rapid_energy, rapid_gate

W, H = 80, 60
_FPS = 8.0


def _i420(y: np.ndarray) -> bytes:
    """Wrap a (H, W) uint8 Y plane into an I420 byte blob (UV flat 128)."""
    assert y.shape == (H, W) and y.dtype == np.uint8
    uv = np.full((H // 2) * W // 2 * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _frames(planes: list[np.ndarray]) -> list[FrameSnapshot]:
    return [
        FrameSnapshot(
            width=W, height=H, data=_i420(p), captured_at=100.0 + i / _FPS
        )
        for i, p in enumerate(planes)
    ]


def _flat(value: int = 60) -> np.ndarray:
    return np.full((H, W), value, dtype=np.uint8)


def _flicker_frames(
    n: int = 12, *, x0: int = 30, y0: int = 20, size: int = 16
) -> list[FrameSnapshot]:
    """A `size`×`size` block alternating dark/bright every frame — the
    in-place oscillation signature of convulsive shaking."""
    planes = []
    for i in range(n):
        p = _flat()
        p[y0 : y0 + size, x0 : x0 + size] = 180 if i % 2 else 60
        planes.append(p)
    return _frames(planes)


def _translating_frames(n: int = 12, *, size: int = 16) -> list[FrameSnapshot]:
    """The same-sized bright block sliding 5 px per frame (no wraparound —
    revisiting pixels would fake persistence) — walking-like translation:
    comparable per-pair activity, near-zero per-pixel churn."""
    planes = []
    for i in range(n):
        p = _flat()
        x0 = i * 5
        assert x0 + size <= W, "fixture must not wrap"
        p[22 : 22 + size, x0 : x0 + size] = 180
        planes.append(p)
    return _frames(planes)


def test_stillness_scores_zero_and_gate_stays_closed() -> None:
    frames = _frames([_flat() for _ in range(12)])
    e = rapid_energy(frames)
    assert e.active_fraction == pytest.approx(0.0)
    assert e.score == pytest.approx(0.0)
    assert rapid_gate(frames) is False


def test_in_place_flicker_scores_high_and_opens_gate() -> None:
    frames = _flicker_frames()
    e = rapid_energy(frames)
    assert e.score > 0.6          # most active pixels churn in most pairs
    assert e.active_fraction > 0.01
    assert rapid_gate(frames) is True


def test_translating_motion_concentrates_far_less_than_flicker() -> None:
    """Discrimination contract (eval-tuned 2026-07-13): translation scores a
    FRACTION of in-place churn. The gate deliberately errs open — vigorous
    locomotion may open it (the flash model is the second line of defense,
    1/30 FP measured); what must never open it is stillness, specks and
    weak churn (the tests below). Recall > savings."""
    flicker = rapid_energy(_flicker_frames())
    translating = rapid_energy(_translating_frames())
    # Comparable total activity, wildly different churn concentration.
    assert translating.active_fraction > 0.01
    assert translating.score < flicker.score / 3


def test_tiny_active_area_is_floored_out() -> None:
    """A 2×2 oscillating speck (sensor noise / a status LED) concentrates
    churn perfectly but must fail the activity floor."""
    frames = _flicker_frames(size=2)
    e = rapid_energy(frames)
    assert e.active_fraction < 0.004  # under the default gate floor
    assert rapid_gate(frames) is False


def test_region_scopes_the_signal() -> None:
    """With a region that excludes the flicker block, the gate stays
    closed; a region containing it opens the gate (person-box scoping)."""
    frames = _flicker_frames(x0=50, y0=30, size=16)
    assert rapid_gate(frames, region=(0, 0, 40, 60)) is False
    assert rapid_gate(frames, region=(40, 20, 80, 60)) is True


def test_too_few_frames_is_never_rapid() -> None:
    assert rapid_gate(_flicker_frames(n=3)) is False


def _i420_sized(y: np.ndarray) -> bytes:
    h, w = y.shape
    uv = np.full((h // 2) * (w // 2) * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def test_downscale_path_preserves_discrimination() -> None:
    """Real camera frames exceed the 160px downscale width — the resize
    branch must run and keep the in-place-flicker signal intact."""
    w, h = 320, 240
    frames = []
    for i in range(12):
        p = np.full((h, w), 60, dtype=np.uint8)
        p[80:144, 120:184] = 180 if i % 2 else 60
        frames.append(
            FrameSnapshot(
                width=w, height=h, data=_i420_sized(p), captured_at=100.0 + i / 8
            )
        )
    e = rapid_energy(frames)
    assert e.score > 0.6
    assert rapid_gate(frames) is True


def test_mixed_resolution_frames_are_filtered_not_fatal() -> None:
    """A publisher renegotiation mid-window injects odd-shaped frames; the
    dominant-shape filter must drop them without changing the verdict."""
    frames = _flicker_frames()
    odd = np.full((30, 40), 60, dtype=np.uint8)
    frames.insert(
        3, FrameSnapshot(width=40, height=30, data=_i420_sized(odd), captured_at=100.9)
    )
    assert rapid_energy(frames).score == pytest.approx(
        rapid_energy(_flicker_frames()).score
    )
    assert rapid_gate(frames) is True


def test_corrupt_short_frame_data_is_skipped_not_fatal() -> None:
    """Truncated frame bytes (right shape, short data) must be excluded,
    never raise; an all-corrupt window scores zero."""
    frames = _flicker_frames()
    frames.insert(
        2, FrameSnapshot(width=W, height=H, data=b"\x00" * 10, captured_at=100.6)
    )
    assert rapid_gate(frames) is True
    all_bad = [
        FrameSnapshot(width=W, height=H, data=b"", captured_at=100.0 + i)
        for i in range(6)
    ]
    e = rapid_energy(all_bad)
    assert e.score == 0.0
    assert rapid_gate(all_bad) is False
