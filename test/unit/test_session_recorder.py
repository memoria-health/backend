"""Task 7 — SessionRecorder state machine (fakes, no ffmpeg/real time).

Drives IDLE → RECORDING → LINGERING → IDLE off on_tick(collect), with
an injected clock + encoder factory so transitions, lead-in seeding,
roll, and linger are tested deterministically.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from src.services.data_collection.session import (
    RecorderCtx,
    SessionRecorder,
    SessionState,
)
from src.services.detection.frame_buffer import FrameSnapshot

_UTC = datetime(2026, 6, 24, 9, 0, 0, tzinfo=timezone.utc)


def _blob(w: int = 320, h: int = 240) -> bytes:
    return b"\x80" * (w * h * 3 // 2)


class _FakeEncoder:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.started = False
        self.closed = False
        self.fed: list[tuple[int, int, int]] = []

    async def start(self) -> None:
        self.started = True

    def feed(self, w: int, h: int, data: bytes) -> None:
        self.fed.append((w, h, len(data)))

    async def aclose(self) -> Path:
        self.closed = True
        return self.path


class _Factory:
    def __init__(self) -> None:
        self.created: list[_FakeEncoder] = []

    def __call__(self, path: Path) -> _FakeEncoder:
        e = _FakeEncoder(path)
        self.created.append(e)
        return e


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _ctx() -> RecorderCtx:
    return RecorderCtx(
        caregiver_id=uuid4(),
        patient_id=uuid4(),
        room="Bedroom",
        gate="person",
        confidence=0.7,
    )


def _make(tmp_path, clock, factory, finalized, *, roll=900.0, linger=45.0):
    async def on_finalized(fs) -> None:
        finalized.append(fs)

    return SessionRecorder(
        camera_id=uuid4(),
        ctx=_ctx(),
        temp_dir=tmp_path,
        encoder_factory=factory,
        on_finalized=on_finalized,
        clock=clock,
        utcnow=lambda: _UTC,
        roll_seconds=roll,
        linger_seconds=linger,
    )


@pytest.mark.asyncio
async def test_collect_starts_session_and_seeds_lead_in(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized)
    lead = [FrameSnapshot(width=320, height=240, data=_blob(), captured_at=999.0)]

    await rec.on_tick(collect=True, lead_in_frames=lead)

    assert rec.state is SessionState.RECORDING
    enc = factory.created[0]
    assert enc.started is True
    assert enc.fed == [(320, 240, len(_blob()))]  # lead-in frame seeded


@pytest.mark.asyncio
async def test_presence_drop_enters_linger_without_finalizing(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized)
    await rec.on_tick(collect=True)
    await rec.on_tick(collect=False)
    assert rec.state is SessionState.LINGERING
    assert finalized == []


@pytest.mark.asyncio
async def test_presence_resumes_within_linger_keeps_same_session(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized, linger=45.0)
    await rec.on_tick(collect=True)
    await rec.on_tick(collect=False)  # lingering
    clock.advance(10.0)  # < linger
    await rec.on_tick(collect=True)  # presence back
    assert rec.state is SessionState.RECORDING
    assert len(factory.created) == 1  # same encoder, no new file
    assert finalized == []


@pytest.mark.asyncio
async def test_linger_expiry_finalizes(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized, linger=45.0)
    await rec.on_tick(collect=True)
    await rec.on_tick(collect=False)  # lingering
    clock.advance(46.0)  # > linger
    await rec.on_tick(collect=False)

    assert rec.state is SessionState.IDLE
    assert len(finalized) == 1
    fs = finalized[0]
    assert fs.video_path == factory.created[0].path
    assert fs.session_start_utc == _UTC
    assert factory.created[0].closed is True


@pytest.mark.asyncio
async def test_roll_finalizes_and_starts_new_session(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized, roll=900.0)
    await rec.on_tick(collect=True)  # session 1 at t=1000
    clock.advance(901.0)  # past roll
    await rec.on_tick(collect=True)  # still collecting → roll over

    assert len(finalized) == 1  # session 1 finalized
    assert len(factory.created) == 2  # session 2 started
    assert rec.state is SessionState.RECORDING
    assert factory.created[0].closed is True
    assert factory.created[1].started is True


@pytest.mark.asyncio
async def test_feed_video_routes_only_when_active(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized)

    rec.feed_video(320, 240, _blob())  # IDLE → no-op
    assert factory.created == []

    await rec.on_tick(collect=True)
    rec.feed_video(320, 240, _blob())
    assert (320, 240, len(_blob())) in factory.created[0].fed


@pytest.mark.asyncio
async def test_abort_drops_temps_without_finalizing(tmp_path) -> None:
    """Privacy abort: tear down the in-flight session, delete BOTH temp
    files, and NEVER call on_finalized — footage captured before privacy
    turned on must never reach the upload path."""
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized)

    await rec.on_tick(collect=True)
    assert rec.state is SessionState.RECORDING
    enc = factory.created[0]
    # audio.pcm is opened on disk by _start_session; simulate the encoder
    # having written the video file as well.
    video_path = enc.path
    video_path.write_bytes(b"video-bytes")
    audio_path = rec._audio_path
    assert audio_path is not None and audio_path.exists()

    await rec.abort()

    assert rec.state is SessionState.IDLE
    assert finalized == []  # on_finalized (upload path) NEVER invoked
    assert enc.closed is True  # encoder torn down
    assert not video_path.exists()  # temp video discarded
    assert not audio_path.exists()  # temp audio discarded


@pytest.mark.asyncio
async def test_audio_lead_in_and_appends_then_persists_on_finalize(tmp_path) -> None:
    clock, factory, finalized = _Clock(), _Factory(), []
    rec = _make(tmp_path, clock, factory, finalized, linger=45.0)

    await rec.on_tick(collect=True, lead_in_audio=b"\x01\x02")
    rec.feed_audio(b"\x03\x04")
    await rec.on_tick(collect=False)  # lingering
    clock.advance(46.0)
    await rec.on_tick(collect=False)  # finalize

    fs = finalized[0]
    assert fs.audio_path is not None
    assert Path(fs.audio_path).read_bytes() == b"\x01\x02\x03\x04"
