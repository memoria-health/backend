"""A recorded clip must last as long as the session it recorded.

This is the assertion the rest of the data-collection suite was missing.
`test_data_collection_e2e` runs real ffmpeg but only checks that a video
stream and an audio stream exist, so a clip could come out at a third of
its true length and every test still passed — which is exactly what was
happening (measured 2026-08-04: a 20s session produced 18.7s of video on
an idle loop, 6.6s under load).

The old pacer counted its own wakeups and assumed each one meant 1/fps
of elapsed time. `asyncio.sleep` sleeps *at least* that long, never
exactly, and the event loop shared with detection stretches it further,
so the count always ran behind real time. Video came out compressed and
pulled away from the real-time audio it was muxed with.

These tests drive a *simulated* clock, so they assert the arithmetic
without waiting real seconds; `test_duration_survives_a_starved_loop`
uses real time and a genuinely blocked loop as the end-to-end check.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.services.data_collection.mux import mux_av
from src.services.data_collection.streaming_encoder import StreamingMp4Writer

_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH",
)

_W, _H, _FPS = 64, 48, 15


def _frame() -> bytes:
    return b"\x80" * (_W * _H * 3 // 2)


class _FakeClock:
    """Hand-advanced monotonic clock."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeStdin:
    def __init__(self) -> None:
        self.writes = 0

    def write(self, data: bytes) -> None:
        self.writes += 1

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = _FakeStdin()

    async def wait(self) -> int:
        return 0


def _writer_with_fake_proc(tmp_path: Path, clock: _FakeClock):
    w = StreamingMp4Writer(
        tmp_path / "v.mp4", width=_W, height=_H, fps=_FPS, clock=clock
    )
    proc = _FakeProc()
    w._proc = proc  # type: ignore[assignment]
    return w, proc


async def test_late_wake_is_made_up_not_lost(tmp_path) -> None:
    """A pacer wake that lands late must still account for all the time
    that passed, not contribute a single frame."""
    clock = _FakeClock()
    w, proc = _writer_with_fake_proc(tmp_path, clock)
    w.feed(_W, _H, _frame())

    # First fill defines position 0 and writes exactly one frame.
    await w._fill_to_clock(w._latest, _FPS)
    assert w.written == 1

    # The loop stalls for a full second: 15 frames' worth of time passed,
    # so 15 more frames are owed even though this is one wake.
    clock.advance(1.0)
    await w._fill_to_clock(w._latest, _FPS)
    assert w.written == 16
    assert w.duration_seconds == pytest.approx(1.0, abs=0.07)
    assert proc.stdin.writes == 16


async def test_early_wake_writes_nothing(tmp_path) -> None:
    """Waking sooner than a frame is due must not pad the timeline."""
    clock = _FakeClock()
    w, _ = _writer_with_fake_proc(tmp_path, clock)
    w.feed(_W, _H, _frame())
    await w._fill_to_clock(w._latest, _FPS)
    assert w.written == 1

    clock.advance(0.001)  # nowhere near 1/15s
    await w._fill_to_clock(w._latest, _FPS)
    assert w.written == 1


async def test_catch_up_burst_is_bounded_and_carried(tmp_path) -> None:
    """A long stall must not become one huge blocking write; the excess
    carries to the next wake rather than vanishing."""
    clock = _FakeClock()
    w, _ = _writer_with_fake_proc(tmp_path, clock)
    w.feed(_W, _H, _frame())
    await w._fill_to_clock(w._latest, _FPS)

    clock.advance(10.0)  # 150 frames owed
    await w._fill_to_clock(w._latest, _FPS)  # burst cap is _FPS
    assert w.written == 1 + _FPS

    # Still owed, and the next wake keeps working through it.
    await w._fill_to_clock(w._latest, _FPS)
    assert w.written == 1 + 2 * _FPS


async def test_duration_tracks_the_clock_over_a_long_session(tmp_path) -> None:
    """Fifteen simulated minutes must encode as fifteen minutes."""
    clock = _FakeClock()
    w, _ = _writer_with_fake_proc(tmp_path, clock)
    w.feed(_W, _H, _frame())
    await w._fill_to_clock(w._latest, _FPS)

    # Wake irregularly, the way a loaded event loop does.
    for step in [0.05, 0.9, 0.066, 0.3, 0.02, 2.5] * 100:
        clock.advance(step)
        await w._fill_to_clock(w._latest, _FPS * 30)

    assert w.duration_seconds == pytest.approx(clock.now - 1000.0, abs=0.1)


@_ffmpeg
async def test_duration_survives_a_starved_loop(tmp_path) -> None:
    """End-to-end, with real ffmpeg and a real, genuinely blocked loop.

    Before the fix this produced roughly a third of its true length.
    """
    seconds = 4.0
    w = StreamingMp4Writer(tmp_path / "v.mp4", width=_W, height=_H, fps=_FPS)
    await w.start()

    stop = asyncio.Event()

    async def hog() -> None:
        """Block the loop in slices, like a per-frame convert does."""
        while not stop.is_set():
            deadline = asyncio.get_running_loop().time() + 0.02
            while asyncio.get_running_loop().time() < deadline:
                pass
            await asyncio.sleep(0)

    hogs = [asyncio.create_task(hog()) for _ in range(3)]
    loop = asyncio.get_running_loop()
    started = loop.time()
    while loop.time() - started < seconds:
        w.feed(_W, _H, _frame())
        await asyncio.sleep(0.01)
    elapsed = loop.time() - started
    stop.set()
    for h in hogs:
        h.cancel()
    await w.aclose()

    out = tmp_path / "final.mp4"
    await mux_av(w.out_path, None, out, duration_seconds=w.duration_seconds)
    probed = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-of", "json", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    assert float(probed["format"]["duration"]) == pytest.approx(elapsed, rel=0.1)
    assert w.deficit_frames == 0


# ── audio lined up against video ────────────────────────────────────

_BYTES_PER_SECOND = 16000 * 2  # 16 kHz, one channel, two bytes a sample


class _StubEncoder:
    """Encoder that reports a timeline, like the real one."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.first_frame_mono: float | None = None
        self.duration_seconds = 0.0

    async def start(self) -> None:
        self.path.write_bytes(b"")

    def feed(self, w: int, h: int, data: bytes) -> None:
        return None

    async def aclose(self) -> Path:
        self.path.write_bytes(b"video")
        return self.path


def _recorder(tmp_path: Path, clock: _FakeClock, captured: list):
    from uuid import uuid4

    from datetime import datetime, timezone

    from src.services.data_collection.session import RecorderCtx, SessionRecorder

    enc = _StubEncoder(tmp_path / "v.mp4")
    ctx = RecorderCtx(
        caregiver_id=uuid4(), patient_id=uuid4(), camera_id=uuid4(),
        room="Bedroom", gate="person", confidence=0.9,
    )
    rec = SessionRecorder(
        camera_id=ctx.camera_id,
        ctx=ctx,
        temp_dir=tmp_path,
        encoder_factory=lambda p: enc,
        on_finalized=lambda fs: captured.append(fs) or _noop(),
        clock=clock,
        utcnow=lambda: datetime(2026, 8, 4, tzinfo=timezone.utc),
        linger_seconds=0.0,
    )
    return rec, enc


async def _noop() -> None:
    return None


async def test_audio_stall_is_filled_with_silence(tmp_path) -> None:
    """A dropout in the audio stream must not slide everything after it
    earlier for the rest of the clip."""
    clock = _FakeClock()
    captured: list = []
    rec, _ = _recorder(tmp_path, clock, captured)
    await rec.on_tick(True)

    chunk = b"\x01\x02" * 1600  # 3200 bytes == 0.1s of audio
    rec.feed_audio(chunk, captured_at=1000.0)
    # Stream drops out for two seconds, then resumes.
    rec.feed_audio(chunk, captured_at=1002.0)

    # The second chunk belongs 2s in, and only 0.1s of audio existed, so
    # 1.9s of silence goes in between.
    assert rec._audio_bytes == pytest.approx(
        len(chunk) * 2 + int(1.9 * _BYTES_PER_SECOND), abs=4
    )
    assert rec._audio_path.exists()


async def test_small_jitter_is_not_treated_as_a_gap(tmp_path) -> None:
    """Ordinary network jitter must not accumulate padding."""
    clock = _FakeClock()
    captured: list = []
    rec, _ = _recorder(tmp_path, clock, captured)
    await rec.on_tick(True)

    chunk = b"\x01\x02" * 1600  # 0.2s
    rec.feed_audio(chunk, captured_at=1000.0)
    rec.feed_audio(chunk, captured_at=1000.25)  # 50ms of jitter
    assert rec._audio_bytes == len(chunk) * 2


async def test_offset_measures_both_stream_starts(tmp_path) -> None:
    """The offset handed to the muxer is the real gap between the first
    frame and the first audio sample."""
    clock = _FakeClock()
    captured: list = []
    rec, enc = _recorder(tmp_path, clock, captured)
    await rec.on_tick(True)

    rec.feed_audio(b"\x00" * 3200, captured_at=1000.0)  # audio first
    enc.first_frame_mono = 1000.4  # video's opening frame, 0.4s later
    enc.duration_seconds = 12.5

    # First falsy tick starts the linger; the next one expires it
    # (linger_seconds=0) and finalizes.
    await rec.on_tick(False)
    clock.advance(1.0)
    await rec.on_tick(False)

    assert len(captured) == 1
    assert captured[0].audio_offset_seconds == pytest.approx(0.4)
    assert captured[0].video_duration_seconds == pytest.approx(12.5)


@_ffmpeg
async def test_mux_trims_audio_that_started_early(tmp_path) -> None:
    """A positive offset must actually shorten the audio, and the output
    must respect the duration cap rather than the longest stream."""
    video = tmp_path / "v.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=black:s={_W}x{_H}:r={_FPS}",
         "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True,
    )
    audio = tmp_path / "a.pcm"
    audio.write_bytes(b"\x00" * int(5 * _BYTES_PER_SECOND))  # 5s of audio

    out = tmp_path / "final.mp4"
    await mux_av(
        video, audio, out, audio_offset_seconds=2.0, duration_seconds=3.0
    )
    probed = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(out)],
            capture_output=True, text=True, check=True,
        ).stdout
    )
    streams = {s["codec_type"]: s for s in probed["streams"]}
    assert set(streams) == {"video", "audio"}
    # 5s of audio, 2s trimmed off the front, capped at 3s -> 3s.
    assert float(streams["audio"]["duration"]) == pytest.approx(3.0, abs=0.15)
    assert float(streams["video"]["duration"]) == pytest.approx(3.0, abs=0.15)
