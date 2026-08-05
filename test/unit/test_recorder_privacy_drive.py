"""Privacy wiring at the per-tick recorder drive.

When a tick's decision carries ``privacy=True``, ``_drive_recorder``
must call ``abort_camera`` (drop the in-flight clip, no upload) instead
of the normal ``on_tick`` linger/finalize route. With privacy off it
drives ``on_tick`` exactly as before.

The drive is exercised directly with a stub ``self`` so we don't have to
stand up a full CameraIngest (DB, LiveKit, models) just to test one
branch.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.detection.camera_ingest import CameraIngest, _RecorderDecision


class _FakeRecorder:
    def __init__(self) -> None:
        self.on_tick_calls: list[tuple] = []
        self.abort_calls: list = []

    async def on_tick(self, camera_id, collect, *, ctx, lead_in_frames,
                      lead_in_audio, sensitive=None):
        self.on_tick_calls.append((camera_id, collect))

    async def abort_camera(self, camera_id):
        self.abort_calls.append(camera_id)


def _stub(rec: _FakeRecorder, cam):
    return SimpleNamespace(_recorder=rec, camera_id=cam, _log_tag="cam")


@pytest.mark.asyncio
async def test_drive_aborts_when_privacy_on() -> None:
    rec, cam = _FakeRecorder(), uuid4()
    decision = _RecorderDecision(collect=False, ctx=object(), privacy=True)

    await CameraIngest._drive_recorder(_stub(rec, cam), decision)

    assert rec.abort_calls == [cam]
    assert rec.on_tick_calls == []  # never takes the linger/upload route


@pytest.mark.asyncio
async def test_drive_uses_on_tick_when_privacy_off() -> None:
    rec, cam = _FakeRecorder(), uuid4()
    decision = _RecorderDecision(collect=True, ctx=object(), privacy=False)

    await CameraIngest._drive_recorder(_stub(rec, cam), decision)

    assert rec.on_tick_calls == [(cam, True)]
    assert rec.abort_calls == []


@pytest.mark.asyncio
async def test_drive_noop_without_recorder_or_ctx() -> None:
    rec, cam = _FakeRecorder(), uuid4()
    # No ctx yet (early-return tick) → neither path runs.
    await CameraIngest._drive_recorder(
        _stub(rec, cam), _RecorderDecision(privacy=True, ctx=None)
    )
    assert rec.abort_calls == [] and rec.on_tick_calls == []
