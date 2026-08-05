"""The note gets uploaded beside the clip — and never at the clip's cost.

The video is the irreplaceable thing. A sidecar can be regenerated from
the database at any time, so nothing in that path may lose us a clip.
"""

from __future__ import annotations

import json
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
    async def __call__(self, video, audio, out, **kw) -> Path:
        Path(out).write_bytes(b"final")
        return Path(out)


class _Uploader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []

    async def __call__(self, path, key, **kw) -> str:
        body = Path(path).read_bytes() if Path(path).exists() else None
        self.calls.append((key, kw.get("content_type", ""), body))
        return key


def _mgr(tmp_path, uploader, builder):
    return RecorderManager(
        settings=SimpleNamespace(
            b2_key_id="k", b2_app_key="s", b2_bucket="b",
            b2_endpoint_url="e", b2_region="r",
            data_collection_max_concurrent=8,
        ),
        temp_dir=tmp_path,
        encoder_factory=_Enc,
        masking_encoder_factory=_Enc,
        muxer=_Muxer(),
        uploader=uploader,
        clock=lambda: 0.0,
        utcnow=lambda: _UTC,
        linger_seconds=0.0,
        sidecar_builder=builder,
    )


def _ctx(cam):
    return RecorderCtx(
        caregiver_id=uuid4(), patient_id=uuid4(), camera_id=cam,
        room="Bedroom", gate="person", confidence=0.9,
    )


async def _run(mgr, cam, ctx):
    await mgr.on_tick(cam, True, ctx=ctx)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.on_tick(cam, False, ctx=ctx)
    await mgr.drain()


async def test_sidecar_lands_beside_the_clip(tmp_path) -> None:
    async def builder(fs, key, meta):
        return {"schema_version": "1", "clip": {"key": key}}

    up = _Uploader()
    mgr = _mgr(tmp_path, up, builder)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))

    assert len(up.calls) == 2
    clip_key, _, _ = up.calls[0]
    side_key, ctype, body = up.calls[1]
    # Same name, next to each other in a listing.
    assert clip_key.endswith(".mp4")
    assert side_key == clip_key[:-4] + ".json"
    assert ctype == "application/json"
    assert json.loads(body)["clip"]["key"] == clip_key
    assert mgr.stats["sidecars_uploaded"] == 1


async def test_the_clip_goes_up_first(tmp_path) -> None:
    """Order matters: a note about a clip that failed to upload is
    worse than no note."""
    async def builder(fs, key, meta):
        return {"schema_version": "1"}

    up = _Uploader()
    mgr = _mgr(tmp_path, up, builder)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert up.calls[0][0].endswith(".mp4")


async def test_a_failing_builder_never_costs_the_clip(tmp_path) -> None:
    async def builder(fs, key, meta):
        raise RuntimeError("db down")

    up = _Uploader()
    mgr = _mgr(tmp_path, up, builder)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))

    assert len(up.calls) == 1                      # the clip still landed
    assert up.calls[0][0].endswith(".mp4")
    assert mgr.stats["clips_uploaded"] == 1
    assert mgr.stats["sidecars_failed"] == 1


async def test_no_builder_means_no_sidecar_not_a_failure(tmp_path) -> None:
    up = _Uploader()
    mgr = _mgr(tmp_path, up, None)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert len(up.calls) == 1
    assert mgr.stats["sidecars_failed"] == 0


async def test_no_sidecar_when_the_clip_never_uploaded(tmp_path) -> None:
    """A note with no clip is orphaned data in the bucket."""
    calls: list = []

    async def builder(fs, key, meta):
        calls.append(key)
        return {"schema_version": "1"}

    class _Failing:
        async def __call__(self, path, key, **kw):
            raise RuntimeError("b2 down")

    mgr = _mgr(tmp_path, _Failing(), builder)
    mgr._retry_base_delay = 0.0
    mgr._sleep = lambda _s: _noop()
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert calls == []


async def _noop() -> None:
    return None


async def test_temp_sidecar_file_is_cleaned_up(tmp_path) -> None:
    async def builder(fs, key, meta):
        return {"schema_version": "1"}

    mgr = _mgr(tmp_path, _Uploader(), builder)
    cam = uuid4()
    await _run(mgr, cam, _ctx(cam))
    assert list(tmp_path.glob("*.json")) == []
