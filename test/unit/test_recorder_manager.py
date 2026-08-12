"""Task 8 — RecorderManager (fakes: no ffmpeg, no B2, no real time).

Owns one SessionRecorder per actively-collecting camera; on finalize it
muxes, builds the B2 key, uploads with metadata, and deletes temps.
Injected muxer/uploader/clock make the whole path deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.data_collection.manager import RecorderManager
from src.services.data_collection.session import RecorderCtx
from src.services.detection.frame_buffer import FrameSnapshot

_UTC = datetime(2026, 6, 24, 9, 0, 0, tzinfo=timezone.utc)


def _blob(w: int = 320, h: int = 240) -> bytes:
    return b"\x80" * (w * h * 3 // 2)


class _FakeEncoder:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.fed: list[tuple[int, int, int]] = []

    async def start(self) -> None:
        self.path.write_bytes(b"")  # create the temp so cleanup is real

    def feed(self, w: int, h: int, data: bytes) -> None:
        self.fed.append((w, h, len(data)))

    async def aclose(self) -> Path:
        self.path.write_bytes(b"video-bytes")
        return self.path


class _Factory:
    def __init__(self) -> None:
        self.created: list[_FakeEncoder] = []

    def __call__(self, path: Path) -> _FakeEncoder:
        e = _FakeEncoder(path)
        self.created.append(e)
        return e


class _FakeMuxer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, object, Path]] = []

    async def __call__(self, video, audio, out, **kw) -> Path:
        self.calls.append((Path(video), audio, Path(out)))
        Path(out).write_bytes(b"final")
        return Path(out)


class _FakeUploader:
    def __init__(self, fail_times: int = 0) -> None:
        self.calls: list[tuple[Path, str, dict]] = []
        self.regions: list[str] = []
        self.fail_times = fail_times

    async def __call__(self, path, key, *, metadata, settings, region="US") -> str:
        self.calls.append((Path(path), key, dict(metadata)))
        self.regions.append(region)
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("transient upload error")
        return key


class _Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


async def _noop_sleep(_seconds: float) -> None:
    return None


def _settings(max_concurrent: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        b2_key_id="k",
        b2_app_key="a",
        b2_bucket="memoria-data",
        data_collection_max_concurrent=max_concurrent,
        data_collection_redaction_enabled=False,
    )


def _make_manager(tmp_path, *, settings=None, muxer=None, uploader=None, clock=None):
    return RecorderManager(
        settings=settings or _settings(),
        temp_dir=tmp_path,
        encoder_factory=_Factory(),
        muxer=muxer or _FakeMuxer(),
        uploader=uploader or _FakeUploader(),
        clock=clock or _Clock(),
        utcnow=lambda: _UTC,
        roll_seconds=900.0,
        linger_seconds=45.0,
        retry_base_delay=0.0,
        sleep=_noop_sleep,
    )


def _ctx(room="Living Room", gate="poi", conf=0.9) -> RecorderCtx:
    return RecorderCtx(
        caregiver_id=uuid4(),
        patient_id=uuid4(),
        camera_id=uuid4(),
        room=room,
        gate=gate,
        confidence=conf,
    )


@pytest.mark.asyncio
async def test_upload_routes_to_patient_region(tmp_path) -> None:
    """Data-residency wiring: the recorder resolves the patient's region via
    the injected region_lookup and threads it to the uploader, so an India
    patient's research clip lands in the India bucket. This exercises the
    patient-row → upload call path end-to-end through the manager."""
    clock = _Clock()
    muxer, uploader = _FakeMuxer(), _FakeUploader()

    async def _region_in(_patient_id) -> str:
        return "IN"

    mgr = RecorderManager(
        settings=_settings(),
        temp_dir=tmp_path,
        encoder_factory=_Factory(),
        muxer=muxer,
        uploader=uploader,
        clock=clock,
        utcnow=lambda: _UTC,
        roll_seconds=900.0,
        linger_seconds=45.0,
        retry_base_delay=0.0,
        sleep=_noop_sleep,
        region_lookup=_region_in,
    )
    cam = uuid4()
    ctx = _ctx()
    lead = [FrameSnapshot(width=320, height=240, data=_blob(), captured_at=999.0)]

    await mgr.on_tick(cam, True, ctx=ctx, lead_in_frames=lead, lead_in_audio=b"\x00\x00")
    mgr.feed_video(cam, 320, 240, _blob())
    mgr.feed_audio(cam, b"\x01\x02")
    await mgr.on_tick(cam, False, ctx=ctx)  # lingering
    clock.advance(46.0)
    await mgr.on_tick(cam, False, ctx=ctx)  # finalize
    await mgr.drain()

    assert len(uploader.calls) == 1
    assert uploader.regions == ["IN"]


@pytest.mark.asyncio
async def test_full_session_muxes_uploads_with_key_and_cleans_up(tmp_path) -> None:
    clock = _Clock()
    muxer, uploader = _FakeMuxer(), _FakeUploader()
    mgr = RecorderManager(
        settings=_settings(),
        temp_dir=tmp_path,
        encoder_factory=(factory := _Factory()),
        muxer=muxer,
        uploader=uploader,
        clock=clock,
        utcnow=lambda: _UTC,
        roll_seconds=900.0,
        linger_seconds=45.0,
        retry_base_delay=0.0,
        sleep=_noop_sleep,
    )
    cam = uuid4()
    ctx = _ctx()
    lead = [FrameSnapshot(width=320, height=240, data=_blob(), captured_at=999.0)]

    await mgr.on_tick(cam, True, ctx=ctx, lead_in_frames=lead, lead_in_audio=b"\x00\x00")
    mgr.feed_video(cam, 320, 240, _blob())
    mgr.feed_audio(cam, b"\x01\x02")
    await mgr.on_tick(cam, False, ctx=ctx)  # lingering
    clock.advance(46.0)
    await mgr.on_tick(cam, False, ctx=ctx)  # finalize
    await mgr.drain()  # await the fire-and-forget upload

    assert len(muxer.calls) == 1
    assert len(uploader.calls) == 1
    _, key, meta = uploader.calls[0]
    assert key == (
        f"data/v1/{ctx.caregiver_id}/{ctx.patient_id}"
        f"/2026-06-24/090000Z__living-room.mp4"
    )
    assert meta["patient_id"] == str(ctx.patient_id)
    assert meta["gate"] == "poi"
    assert meta["room"] == "Living Room"
    # All temp files removed after upload.
    assert not factory.created[0].path.exists()
    assert mgr.active_count == 0


@pytest.mark.asyncio
async def test_concurrent_session_cap_blocks_new_camera(tmp_path) -> None:
    mgr = _make_manager(tmp_path, settings=_settings(max_concurrent=2))
    cams = [uuid4() for _ in range(3)]
    ctx = _ctx()
    for c in cams[:2]:
        await mgr.on_tick(c, True, ctx=ctx)
    assert mgr.active_count == 2

    await mgr.on_tick(cams[2], True, ctx=ctx)  # over the cap
    assert mgr.active_count == 2  # third was not started


@pytest.mark.asyncio
async def test_feed_routes_to_the_right_camera(tmp_path) -> None:
    factory = _Factory()
    mgr = RecorderManager(
        settings=_settings(),
        temp_dir=tmp_path,
        encoder_factory=factory,
        muxer=_FakeMuxer(),
        uploader=_FakeUploader(),
        clock=_Clock(),
        utcnow=lambda: _UTC,
    )
    cam_a, cam_b = uuid4(), uuid4()
    ctx = _ctx()
    await mgr.on_tick(cam_a, True, ctx=ctx)
    await mgr.on_tick(cam_b, True, ctx=ctx)

    mgr.feed_video(cam_a, 320, 240, _blob())
    enc_a, enc_b = factory.created[0], factory.created[1]
    assert len(enc_a.fed) == 1
    assert len(enc_b.fed) == 0


@pytest.mark.asyncio
async def test_feed_for_unknown_camera_is_noop(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    # No session for this camera — must not raise.
    mgr.feed_video(uuid4(), 320, 240, _blob())
    mgr.feed_audio(uuid4(), b"\x00")


@pytest.mark.asyncio
async def test_finalize_camera_flushes_in_flight_session(tmp_path) -> None:
    muxer, uploader = _FakeMuxer(), _FakeUploader()
    mgr = _make_manager(tmp_path, muxer=muxer, uploader=uploader)
    cam = uuid4()
    await mgr.on_tick(cam, True, ctx=_ctx())  # session in flight
    assert mgr.active_count == 1

    await mgr.finalize_camera(cam)  # camera removed mid-session
    await mgr.drain()

    assert mgr.active_count == 0
    assert len(muxer.calls) == 1
    assert len(uploader.calls) == 1


@pytest.mark.asyncio
async def test_finalize_camera_without_session_is_noop(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    await mgr.finalize_camera(uuid4())  # must not raise
    assert mgr.active_count == 0


@pytest.mark.asyncio
async def test_abort_camera_drops_session_without_mux_or_upload(tmp_path) -> None:
    """Privacy abort: an in-flight session is discarded — no mux, no
    upload, temp files removed."""
    muxer, uploader = _FakeMuxer(), _FakeUploader()
    mgr = _make_manager(tmp_path, muxer=muxer, uploader=uploader)
    cam = uuid4()
    await mgr.on_tick(cam, True, ctx=_ctx())  # session recording
    assert mgr.active_count == 1
    assert list(tmp_path.iterdir()) != []  # temps exist while recording

    await mgr.abort_camera(cam)
    await mgr.drain()

    assert mgr.active_count == 0
    assert len(muxer.calls) == 0  # never muxed
    assert len(uploader.calls) == 0  # never uploaded
    assert list(tmp_path.iterdir()) == []  # temps discarded


@pytest.mark.asyncio
async def test_abort_camera_without_session_is_noop(tmp_path) -> None:
    mgr = _make_manager(tmp_path)
    await mgr.abort_camera(uuid4())  # must not raise
    assert mgr.active_count == 0


async def _run_one_session(mgr, clock) -> None:
    cam = uuid4()
    await mgr.on_tick(cam, True, ctx=_ctx())
    await mgr.on_tick(cam, False, ctx=_ctx())  # linger
    clock.advance(46.0)
    await mgr.on_tick(cam, False, ctx=_ctx())  # finalize
    await mgr.drain()


@pytest.mark.asyncio
async def test_upload_retries_then_gives_up_but_still_cleans_up(tmp_path) -> None:
    clock = _Clock()
    uploader = _FakeUploader(fail_times=99)  # never succeeds
    mgr = _make_manager(tmp_path, uploader=uploader, clock=clock)
    await _run_one_session(mgr, clock)

    assert len(uploader.calls) == 3  # default upload_retries, then gives up
    assert mgr.active_count == 0
    # Temp files are cleaned up even when every upload attempt failed.
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_upload_retries_then_succeeds(tmp_path) -> None:
    clock = _Clock()
    uploader = _FakeUploader(fail_times=1)  # first attempt fails, second wins
    mgr = _make_manager(tmp_path, uploader=uploader, clock=clock)
    await _run_one_session(mgr, clock)

    assert len(uploader.calls) == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_aclose_finalizes_in_flight_sessions_on_shutdown(tmp_path) -> None:
    muxer, uploader = _FakeMuxer(), _FakeUploader()
    mgr = _make_manager(tmp_path, muxer=muxer, uploader=uploader)
    await mgr.on_tick(uuid4(), True, ctx=_ctx())  # recording, never lingered out
    assert mgr.active_count == 1

    await mgr.aclose()  # graceful shutdown

    assert mgr.active_count == 0
    assert len(muxer.calls) == 1
    assert len(uploader.calls) == 1


@pytest.mark.asyncio
async def test_b2_not_configured_skips_upload_but_cleans_up(tmp_path) -> None:
    clock = _Clock()
    uploader = _FakeUploader()
    settings = SimpleNamespace(
        b2_key_id="", b2_app_key="", data_collection_max_concurrent=8
    )
    mgr = _make_manager(tmp_path, settings=settings, uploader=uploader, clock=clock)
    await _run_one_session(mgr, clock)

    assert uploader.calls == []  # never attempted without creds
    assert list(tmp_path.iterdir()) == []  # still cleaned up
