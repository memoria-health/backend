"""Task 12 — end-to-end recorder path (real ffmpeg, mocked B2).

Drives a synthetic presence session through the REAL RecorderManager
with the REAL streaming encoder + REAL A/V mux, and only B2 stubbed.
Asserts exactly one clip is produced, muxed with audio, keyed
correctly, and uploaded once. Skipped without ffmpeg/ffprobe.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.data_collection.manager import RecorderManager
from src.services.data_collection.mux import mux_av
from src.services.data_collection.session import RecorderCtx
from src.services.data_collection.streaming_encoder import StreamingMp4Writer
from src.services.detection.frame_buffer import FrameSnapshot

_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

_UTC = datetime(2026, 6, 24, 9, 0, 0, tzinfo=timezone.utc)
_W, _H = 320, 240


def _blob() -> bytes:
    return b"\x80" * (_W * _H * 3 // 2)


def _ffprobe(path: Path) -> list[dict]:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-i", str(path)],
        capture_output=True, check=True,
    )
    return json.loads(proc.stdout)["streams"]


class _CapturingUploader:
    """Copies the finished clip aside (before the manager deletes it)
    so the test can ffprobe it, and records the call."""

    def __init__(self, dest: Path) -> None:
        self.dest = dest
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, path, key, *, metadata, settings, region=None) -> str:
        shutil.copy(path, self.dest)
        self.calls.append((key, dict(metadata)))
        return key


async def _noop_sleep(_s: float) -> None:
    return None


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@_ffmpeg
@pytest.mark.asyncio
async def test_presence_session_produces_one_uploaded_av_clip(tmp_path) -> None:
    clock = _Clock()
    saved = tmp_path / "uploaded.mp4"
    uploader = _CapturingUploader(saved)
    settings = SimpleNamespace(
        b2_key_id="k",
        b2_app_key="a",
        b2_bucket="memoria-data",
        data_collection_max_concurrent=8,
    )
    mgr = RecorderManager(
        settings=settings,
        temp_dir=tmp_path,
        encoder_factory=lambda p: StreamingMp4Writer(p, width=_W, height=_H, fps=15),
        muxer=mux_av,
        uploader=uploader,
        clock=clock,
        utcnow=lambda: _UTC,
        roll_seconds=900.0,
        linger_seconds=45.0,
        retry_base_delay=0.0,
        sleep=_noop_sleep,
    )
    cam = uuid4()
    ctx = RecorderCtx(
        caregiver_id=uuid4(),
        patient_id=uuid4(),
        room="Living Room",
        gate="poi",
        confidence=0.91,
    )

    # Tick 1: flag-on + motion + presence → start, seeding the 4fps
    # lead-in (analysis window + matching audio).
    lead = [
        FrameSnapshot(width=_W, height=_H, data=_blob(), captured_at=float(i))
        for i in range(4)
    ]
    await mgr.on_tick(
        cam, True, ctx=ctx, lead_in_frames=lead, lead_in_audio=b"\x00\x00" * 800
    )

    # Live forward: feed real frames + audio over ~0.6s of wall-clock.
    for _ in range(20):
        mgr.feed_video(cam, _W, _H, _blob())
        mgr.feed_audio(cam, b"\x00\x00" * 160)
        await asyncio.sleep(0.03)

    # Presence drops → linger → finalize.
    await mgr.on_tick(cam, False, ctx=ctx)
    clock.advance(46.0)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.drain()

    # Exactly one clip uploaded, with the expected key + metadata.
    assert len(uploader.calls) == 1
    key, meta = uploader.calls[0]
    assert key == (
        f"data/v1/{ctx.caregiver_id}/{ctx.patient_id}"
        f"/2026-06-24/090000Z__living-room.mp4"
    )
    assert meta["gate"] == "poi"
    assert meta["patient_id"] == str(ctx.patient_id)

    # The uploaded file is a real MP4 carrying BOTH video and audio.
    kinds = {s["codec_type"] for s in _ffprobe(saved)}
    assert "video" in kinds
    assert "audio" in kinds

    # No temp files left behind in the recorder dir (only our saved copy).
    leftovers = [p.name for p in tmp_path.iterdir() if p != saved]
    assert leftovers == []

    assert mgr.active_count == 0
