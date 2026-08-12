"""Clip audio, sensitive-verdict priors, and the counters that make
collection observable.

Audio: research clips carry their audio unconditionally (2026-07-28
product call — visuals are censored, sound stays; redaction pixelates
video spans and never touches the track). Capture itself is severed at
the publisher by `patients.audio_capture_enabled` when audio is illegal
to record.

Sensitive verdicts (2026-08-12): a True verdict no longer switches
encoders — it is NOTED as a clip-relative offset and rides
FinalizedSession.sensitive_offsets into the finalize-time redaction
pass as a span prior.

Counters: without them, collection can stop completely (cap reached, B2
unconfigured, every upload failing) and nothing anywhere says so.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.services.data_collection.manager import RecorderManager
from src.services.data_collection.session import RecorderCtx

_UTC = datetime(2026, 8, 5, 9, 0, 0, tzinfo=timezone.utc)


class _Enc:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.first_frame_mono = 100.0
        self.duration_seconds = 5.0
        self.ok = True
        self.deficit_seconds = 0.0
        self.dropped = 0

    async def start(self) -> None:
        self.path.write_bytes(b"")

    def feed(self, w, h, data) -> None: ...

    async def aclose(self) -> Path:
        self.path.write_bytes(b"video")
        return self.path


class _Muxer:
    def __init__(self) -> None:
        self.audio_args: list = []

    async def __call__(self, video, audio, out, **kw) -> Path:
        self.audio_args.append(audio)
        Path(out).write_bytes(b"final")
        return Path(out)


class _Uploader:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list = []
        self.fail = fail

    async def __call__(self, path, key, **kw) -> str:
        if self.fail:
            raise RuntimeError("b2 down")
        self.calls.append(key)
        return key


def _settings(**over):
    base = dict(
        b2_key_id="k", b2_app_key="s", b2_bucket="b",
        b2_endpoint_url="e", b2_region="r",
        data_collection_max_concurrent=8,
        data_collection_redaction_enabled=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _mgr(tmp_path, *, settings=None, muxer=None, uploader=None,
         span_detector=None):
    kw = {}
    if span_detector is not None:
        kw["span_detector"] = span_detector
    return RecorderManager(
        settings=settings or _settings(),
        temp_dir=tmp_path,
        encoder_factory=_Enc,
        muxer=muxer or _Muxer(),
        uploader=uploader or _Uploader(),
        clock=lambda: 0.0,
        utcnow=lambda: _UTC,
        linger_seconds=0.0,
        retry_base_delay=0.0,
        sleep=lambda _s: _noop(),
    **kw)


async def _noop() -> None:
    return None


def _ctx(cam):
    return RecorderCtx(
        caregiver_id=uuid4(), patient_id=uuid4(), camera_id=cam,
        room="Bathroom", gate="person", confidence=0.9,
    )


async def _run(mgr, cam, ctx, *, sensitive=None):
    await mgr.on_tick(cam, True, ctx=ctx, sensitive=sensitive)
    mgr.feed_audio(cam, b"\x01\x02" * 100, captured_at=0.0)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.drain()


async def test_clip_audio_is_always_kept(tmp_path) -> None:
    muxer = _Muxer()
    mgr = _mgr(tmp_path, muxer=muxer)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam), sensitive=True)
    assert muxer.audio_args and muxer.audio_args[0] is not None


async def test_sensitive_verdict_becomes_a_redaction_prior(tmp_path) -> None:
    """A True verdict during a session must surface in the priors handed
    to the span detector — that is the entire remaining effect of the
    tick-time sensitivity screen on recording."""
    seen: dict = {}

    async def _detector(_path, *, duration, prior_offsets, settings):
        seen["priors"] = list(prior_offsets)
        return SimpleNamespace(
            spans=[], chunk_hits=0, prior_hits=len(prior_offsets),
            chunks_screened=0,
        )

    mgr = _mgr(
        tmp_path,
        settings=_settings(data_collection_redaction_enabled=True),
        span_detector=_detector,
    )
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam), sensitive=True)
    assert seen["priors"] == [0.0]  # frozen clock: verdict at t=0


async def test_counters_track_a_healthy_upload(tmp_path) -> None:
    uploader = _Uploader()
    mgr = _mgr(tmp_path, uploader=uploader)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert mgr.stats["sessions_started"] == 1
    assert mgr.stats["clips_uploaded"] == 1
    assert mgr.stats["clips_short"] == 0


async def test_counters_track_a_failed_upload(tmp_path) -> None:
    """The case that silently loses data today — it must at least be
    visible."""
    mgr = _mgr(tmp_path, uploader=_Uploader(fail=True))
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert mgr.stats["clips_uploaded"] == 0
    assert mgr.stats["dropped_upload_failed"] == 1


async def test_counters_track_b2_not_configured(tmp_path) -> None:
    mgr = _mgr(tmp_path, settings=_settings(b2_key_id="", b2_app_key=""))
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert mgr.stats["dropped_no_b2"] == 1
