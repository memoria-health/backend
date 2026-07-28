"""Task 10 — camera_ingest → recorder handoff.

The deep tick body (LLM, re-ID) is covered by the e2e test; here we
verify just the seam: _drive_recorder forwards the per-tick decision to
the recorder, and is an inert no-op when the feature isn't wired or the
tick bailed before learning the patient.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.data_collection.session import RecorderCtx
from src.services.detection.camera_ingest import CameraIngest, _RecorderDecision


class _FakeRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def on_tick(
        self, camera_id, collect, *, ctx, lead_in_frames, lead_in_audio
    ) -> None:
        self.calls.append((camera_id, collect, ctx, lead_in_frames, lead_in_audio))


def _settings() -> SimpleNamespace:
    # frame_buffer_retention_seconds is a derived property on the real
    # Settings (max(window, post_roll) + margin = max(11, 15) + 2 = 17);
    # the fake just supplies the resolved value the buffers are sized with.
    return SimpleNamespace(
        livekit_worker_window_seconds=11,
        livekit_worker_target_fps=4,
        frame_buffer_target_fps=4,
        clip_fps=4,
        clip_post_event_wait_seconds=15.0,
        frame_buffer_retention_seconds=17.0,
        rapid_fps=8,
    )


def _ingest(recorder) -> CameraIngest:
    return CameraIngest(
        camera_id=uuid4(),
        patient_id=uuid4(),
        sessionmaker=None,
        motion_cache=None,
        settings=_settings(),
        reid=None,
        recorder=recorder,
    )


@pytest.mark.asyncio
async def test_drive_recorder_forwards_the_decision() -> None:
    rec = _FakeRecorder()
    ing = _ingest(rec)
    ctx = RecorderCtx(
        caregiver_id=uuid4(),
        patient_id=ing.patient_id,
        camera_id=ing.camera_id,
        room="Bedroom",
        gate="poi",
        confidence=0.8,
    )
    decision = _RecorderDecision(
        collect=True, ctx=ctx, lead_frames=["f"], lead_audio=b"a"
    )

    await ing._drive_recorder(decision)

    assert len(rec.calls) == 1
    cam_id, collect, got_ctx, frames, audio = rec.calls[0]
    assert cam_id == ing.camera_id
    assert collect is True
    assert got_ctx is ctx
    assert frames == ["f"]
    assert audio == b"a"


@pytest.mark.asyncio
async def test_drive_recorder_is_noop_without_a_recorder() -> None:
    ing = _ingest(None)
    ctx = RecorderCtx(
        caregiver_id=uuid4(), patient_id=uuid4(), camera_id=uuid4(), room="x",
        gate="person", confidence=None,
    )
    await ing._drive_recorder(_RecorderDecision(collect=True, ctx=ctx))  # no raise


@pytest.mark.asyncio
async def test_drive_recorder_is_noop_before_patient_loaded() -> None:
    rec = _FakeRecorder()
    ing = _ingest(rec)
    await ing._drive_recorder(_RecorderDecision())  # ctx is None
    assert rec.calls == []
