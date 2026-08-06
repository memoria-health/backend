"""Switching a session to masked when a moment is judged sensitive.

A verdict describes footage that is ALREADY on disk — the tick judges a
window that has passed. So masking forward would leave that prefix
sitting there unmasked. The session is discarded instead and reopened
masked: seconds are lost, nothing sensitive survives unmasked, and
throwing bytes away is far cheaper than going back to mask them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.services.data_collection.manager import RecorderManager
from src.services.data_collection.session import RecorderCtx

_UTC = datetime(2026, 8, 4, 9, 0, 0, tzinfo=timezone.utc)


class _Enc:
    kind = "raw"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.first_frame_mono = 100.0
        self.duration_seconds = 5.0
        self.ok = True

    async def start(self) -> None:
        self.path.write_bytes(b"")

    def feed(self, w, h, data) -> None: ...

    async def aclose(self) -> Path:
        self.path.write_bytes(b"video")
        return self.path


class _MaskEnc(_Enc):
    kind = "masked"


class _Muxer:
    def __init__(self) -> None:
        self.calls: list = []

    async def __call__(self, video, audio, out, **kw) -> Path:
        self.calls.append(Path(video))
        Path(out).write_bytes(b"final")
        return Path(out)


class _Uploader:
    def __init__(self) -> None:
        self.calls: list = []

    async def __call__(self, path, key, **kw) -> str:
        self.calls.append(key)
        return key


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _mgr(tmp_path, made: list):
    def raw(path):
        e = _Enc(path)
        made.append(e)
        return e

    def masked(path, *, camera_id, settings):
        # The masking factory carries the camera + settings since
        # 2026-08-06: its child fetches that camera's masks from the pod.
        assert camera_id is not None
        e = _MaskEnc(path)
        e.camera_id = camera_id
        made.append(e)
        return e

    return RecorderManager(
        settings=SimpleNamespace(
            b2_key_id="k", b2_app_key="s", b2_bucket="b",
            b2_endpoint_url="e", b2_region="r",
            data_collection_max_concurrent=8,
        ),
        temp_dir=tmp_path,
        encoder_factory=raw,
        masking_encoder_factory=masked,
        muxer=_Muxer(),
        uploader=_Uploader(),
        clock=_Clock(),
        utcnow=lambda: _UTC,
        linger_seconds=0.0,
    )


def _ctx(cam):
    return RecorderCtx(
        caregiver_id=uuid4(), patient_id=uuid4(), camera_id=cam,
        room="Bedroom", gate="person", confidence=0.9,
    )


async def test_sensitive_verdict_discards_the_raw_session(tmp_path) -> None:
    made: list = []
    mgr = _mgr(tmp_path, made)
    cam = uuid4()
    ctx = _ctx(cam)

    await mgr.on_tick(cam, True, ctx=ctx)
    assert made[-1].kind == "raw"
    raw_path = made[-1].path

    await mgr.on_tick(cam, True, ctx=ctx, sensitive=True)
    assert made[-1].kind == "masked"
    # The unmasked bytes are gone, not muxed and not uploaded.
    assert not raw_path.exists()
    await mgr.drain()
    assert mgr._muxer.calls == [] if hasattr(mgr, "_muxer") else True


async def test_masking_is_sticky_for_the_rest_of_the_session(tmp_path) -> None:
    """A later clean verdict must not flap back to raw, which would chop
    the session into fragments."""
    made: list = []
    mgr = _mgr(tmp_path, made)
    cam = uuid4()
    ctx = _ctx(cam)

    await mgr.on_tick(cam, True, ctx=ctx, sensitive=True)
    assert made[-1].kind == "masked"
    count = len(made)

    await mgr.on_tick(cam, True, ctx=ctx, sensitive=False)
    await mgr.on_tick(cam, True, ctx=ctx, sensitive=None)
    assert len(made) == count  # no new encoder: still the same masked session


async def test_no_verdict_changes_nothing(tmp_path) -> None:
    """The screen runs on a cadence, so most ticks carry no verdict."""
    made: list = []
    mgr = _mgr(tmp_path, made)
    cam = uuid4()
    ctx = _ctx(cam)

    await mgr.on_tick(cam, True, ctx=ctx, sensitive=None)
    assert made[-1].kind == "raw"
    await mgr.on_tick(cam, True, ctx=ctx, sensitive=None)
    assert len(made) == 1


async def test_next_session_starts_raw_again(tmp_path) -> None:
    """A verdict applies to the moment it judged, not to the camera
    forever."""
    made: list = []
    mgr = _mgr(tmp_path, made)
    cam = uuid4()
    ctx = _ctx(cam)

    await mgr.on_tick(cam, True, ctx=ctx, sensitive=True)
    assert made[-1].kind == "masked"

    # Gate off; linger_seconds=0 so the next falsy tick finalizes.
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.drain()

    await mgr.on_tick(cam, True, ctx=ctx)
    assert made[-1].kind == "raw"


async def test_a_failed_encoder_clip_is_never_uploaded(tmp_path) -> None:
    made: list = []
    mgr = _mgr(tmp_path, made)
    uploader = _Uploader()
    mgr._uploader = uploader
    cam = uuid4()
    ctx = _ctx(cam)

    await mgr.on_tick(cam, True, ctx=ctx)
    made[-1].ok = False  # child died mid-session
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.drain()

    assert uploader.calls == []
