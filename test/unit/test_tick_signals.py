"""Unit: Layer-1 tick signals (detection v2 P2 T3).

Synthetic I420 windows with motion planted inside / outside fabricated
track boxes. Frames are 160×120 (below the 320 map width, so map space
== native space == extracted space when sample_width ≥ 160 — factor 1,
which keeps the geometry in the tests readable).
"""

from __future__ import annotations

import time

import numpy as np

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.signals.tick_signals import (
    TickSignals,
    compute_tick_signals,
)
from src.services.identity.observations import TrackObservation

W, H = 160, 120
_SAMPLE_WIDTH = 480  # > W ⇒ extracted width = min(480, 160) = 160


def _frame(y: np.ndarray, t: float) -> FrameSnapshot:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return FrameSnapshot(
        width=W, height=H, data=y.tobytes() + uv.tobytes(), captured_at=t,
    )


def _window(
    *, flicker_box: tuple[int, int, int, int] | None = None, n: int = 12,
) -> list[FrameSnapshot]:
    """n frames at 4 fps; `flicker_box` (x0, y0, x1, y1) alternates
    bright/dark every frame — persistent, high-contrast churn."""
    t0 = time.monotonic() - n / 4.0
    frames = []
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        if flicker_box is not None:
            x0, y0, x1, y1 = flicker_box
            y[y0:y1, x0:x1] = 200 if i % 2 else 60
        frames.append(_frame(y, t0 + i / 4.0))
    return frames


def _track(
    tid: int,
    box: tuple[float, float, float, float],
    *,
    label: str = "UNKNOWN",
    n_frames: int = 12,
    face_px_max: float = 0.0,
) -> TrackObservation:
    return TrackObservation(
        track_id=tid,
        label=label,
        confidence=0.0,
        bbox_series=tuple((i, box) for i in range(n_frames)),
        face_locked=False,
        pre_labeled=False,
        face_px_max=face_px_max,
    )


# ─── Presence ────────────────────────────────────────────────────


def test_tracking_did_not_run_yields_unknowable_presence() -> None:
    sig = compute_tick_signals(
        _window(), tracks=None, attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert sig.person_present is None
    assert sig.people_count is None
    assert sig.patient_present is None
    assert sig.track_motion == ()


def test_presence_and_peak_concurrency_from_series() -> None:
    t1 = TrackObservation(
        track_id=1, label="UNKNOWN", confidence=0.0,
        bbox_series=((0, (10, 10, 40, 90)), (1, (10, 10, 40, 90))),
        face_locked=False, pre_labeled=False,
    )
    t2 = TrackObservation(
        track_id=2, label="UNKNOWN", confidence=0.0,
        bbox_series=((1, (60, 10, 100, 90)),),  # overlaps t1 only on frame 1
        face_locked=False, pre_labeled=False,
    )
    sig = compute_tick_signals(
        _window(), tracks=(t1, t2), attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert sig.person_present is True
    assert sig.people_count == 2   # concurrent on frame 1


def test_ran_and_saw_nobody_is_a_real_false() -> None:
    sig = compute_tick_signals(
        _window(), tracks=(), attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert sig.person_present is False
    assert sig.people_count == 0


# ─── Motion: per-track vs environment ────────────────────────────


def test_motion_inside_track_box_scores_that_track_not_env() -> None:
    box = (40, 30, 90, 90)
    frames = _window(flicker_box=(50, 40, 80, 80))  # inside the track box
    sig = compute_tick_signals(
        frames, tracks=(_track(1, box),), attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    (tm,) = sig.track_motion
    assert tm.track_id == 1
    assert tm.amount > 0.1, "in-box flicker must register as track motion"
    assert tm.churn > 0.5, "synthetic flicker is maximally persistent"
    assert sig.env_motion < 0.02, "in-box motion must not leak into env"


def test_motion_outside_all_boxes_is_env_motion() -> None:
    box = (10.0, 10.0, 50.0, 90.0)
    frames = _window(flicker_box=(100, 20, 150, 100))  # far from the box
    sig = compute_tick_signals(
        frames, tracks=(_track(1, box),), attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    (tm,) = sig.track_motion
    assert tm.amount < 0.02, "env flicker must not count as track motion"
    assert sig.env_motion > 0.05, "outside-box change must register as env"


def test_still_scene_scores_zero_everywhere() -> None:
    sig = compute_tick_signals(
        _window(), tracks=(_track(1, (10, 10, 50, 90)),),
        attribution_ran=False, sample_width=_SAMPLE_WIDTH,
    )
    (tm,) = sig.track_motion
    assert tm.amount == 0.0
    assert sig.env_motion == 0.0


def test_no_tracks_env_motion_reads_whole_frame() -> None:
    """Without person boxes there is nothing to mask: env_motion
    degrades to whole-frame motion — exactly the closed-state wake
    signal T4 needs."""
    frames = _window(flicker_box=(50, 40, 80, 80))
    sig = compute_tick_signals(
        frames, tracks=None, attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert sig.env_motion > 0.01


# ─── Attribution-dependent signals ───────────────────────────────


def test_patient_present_only_when_attribution_ran() -> None:
    tracks = (_track(1, (10, 10, 50, 90), label="PATIENT"),)
    no_attr = compute_tick_signals(
        _window(), tracks=tracks, attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert no_attr.patient_present is None
    with_attr = compute_tick_signals(
        _window(), tracks=tracks, attribution_ran=True,
        sample_width=_SAMPLE_WIDTH,
    )
    assert with_attr.patient_present is True


def test_face_valid_scales_extracted_px_to_native() -> None:
    # Frames are 160 wide = extracted width ⇒ factor 1.0 here, so the
    # floor applies to face_px_max directly; the scaling math itself is
    # exercised via a tightened floor.
    big = (_track(1, (10, 10, 50, 90), face_px_max=80.0),)
    sig = compute_tick_signals(
        _window(), tracks=big, attribution_ran=True,
        sample_width=_SAMPLE_WIDTH, face_min_px=60,
    )
    assert sig.face_valid is True
    small = (_track(1, (10, 10, 50, 90), face_px_max=30.0),)
    sig2 = compute_tick_signals(
        _window(), tracks=small, attribution_ran=True,
        sample_width=_SAMPLE_WIDTH, face_min_px=60,
    )
    assert sig2.face_valid is False
    # No usable frames ⇒ no native scale ⇒ unknowable, never False.
    sig3 = compute_tick_signals(
        [], tracks=big, attribution_ran=True,
        sample_width=_SAMPLE_WIDTH, face_min_px=60,
    )
    assert sig3.face_valid is None
    # Empty room ⇒ no face question to answer ⇒ None, not False.
    sig4 = compute_tick_signals(
        _window(), tracks=(), attribution_ran=True,
        sample_width=_SAMPLE_WIDTH, face_min_px=60,
    )
    assert sig4.face_valid is None


# ─── Stash shape ─────────────────────────────────────────────────


def test_stash_fields_omits_unknowable_and_rounds() -> None:
    sig = compute_tick_signals(
        _window(), tracks=None, attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    d = sig.stash_fields()
    assert "sig_person_present" not in d
    assert "sig_patient_present" not in d
    assert "sig_face_valid" not in d
    assert "sig_env_motion" in d and "sig_frames" in d


def test_stash_fields_is_json_serializable_with_tracks() -> None:
    """The stash feeds analyses.detector_fields (JSONB). A single
    numpy scalar leaking through (np.float32 amount, np.bool_ verdict)
    would pass every equality assert and then kill session.commit() in
    prod — pin serializability on the richest shape: tracks + motion +
    face_valid (T3/T4 review finding)."""
    import json

    tracks = (
        _track(1, (40, 30, 90, 90), label="PATIENT", face_px_max=80.0),
        _track(2, (100, 20, 150, 100)),
    )
    sig = compute_tick_signals(
        _window(flicker_box=(50, 40, 80, 80)), tracks=tracks,
        attribution_ran=True, sample_width=_SAMPLE_WIDTH, face_min_px=60,
    )
    d = sig.stash_fields()
    encoded = json.dumps(d)   # raises TypeError on any numpy leak
    assert "sig_tracks" in d and "sig_face_valid" in d
    assert json.loads(encoded)["sig_patient_present"] is True


def test_stash_fields_caps_track_list_at_eight() -> None:
    tracks = tuple(
        _track(i, (float(i), 10.0, float(i) + 8.0, 90.0)) for i in range(11)
    )
    sig = compute_tick_signals(
        _window(flicker_box=(20, 20, 40, 40)), tracks=tracks,
        attribution_ran=False, sample_width=_SAMPLE_WIDTH,
    )
    d = sig.stash_fields()
    assert len(d["sig_tracks"]) == 8
    assert d["sig_tracks_truncated"] == 11
    assert d["sig_person_present"] is True
    assert d["sig_people_count"] == 11
    assert isinstance(sig, TickSignals)


# ─── JPEG-in-buffer path (2026-07-19 soak bug) ───────────────────


def _jpeg_window(n: int = 12, *, with_luma: bool = True) -> list[FrameSnapshot]:
    """Same flickering window as `_window`, but frames stored the way
    the live buffer stores them since JPEG-in-buffer: JPEG bytes + the
    320x180 luma thumbnail. Whole-frame flicker so the env pass sees it."""
    from src.services.detection import frame_codec

    t0 = time.monotonic() - n / 4.0
    frames = []
    for i in range(n):
        y = np.full((H, W), 200 if i % 2 else 60, dtype=np.uint8)
        uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
        jpeg, luma = frame_codec.compress_for_buffer(
            W, H, y.tobytes() + uv.tobytes(), quality=88,
        )
        frames.append(FrameSnapshot(
            width=W, height=H, data=jpeg, captured_at=t0 + i / 4.0,
            luma=luma if with_luma else b"",
        ))
    return frames


def test_env_motion_sees_jpeg_frames_via_luma_thumbnail() -> None:
    # The live-soak blindness: _map_plane read JPEG bytes as a raw Y
    # plane, failed the size guard, and skipped EVERY frame — env
    # motion pinned at 0.0 while the person visibly moved.
    sig = compute_tick_signals(
        _jpeg_window(), tracks=None, attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert sig.env_motion > 0.5
    assert sig.frames_used >= 10


def test_jpeg_frames_without_luma_skip_not_misread() -> None:
    sig = compute_tick_signals(
        _jpeg_window(with_luma=False), tracks=None, attribution_ran=False,
        sample_width=_SAMPLE_WIDTH,
    )
    assert sig.env_motion == 0.0
    assert sig.frames_used == 0
