"""Finalize-time redaction (2026-08-12): span math and the manager's
fail-closed wiring.

The invariant under test: never UPLOAD an unredacted clip. Spans found
→ the redactor runs and the REDACTED file uploads; any failure in
detection or redaction → the clip is dropped entirely (raw never
leaves the box on an error path).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from src.services.data_collection.manager import RecorderManager
from src.services.data_collection.redaction import merge_spans
from src.services.data_collection.session import RecorderCtx

_UTC = datetime(2026, 8, 12, 9, 0, 0, tzinfo=timezone.utc)


# ── merge_spans: pure ───────────────────────────────────────────────

def test_merge_pads_clamps_and_merges() -> None:
    assert merge_spans(
        [(10.0, 12.0), (13.0, 15.0), (100.0, 101.0)],
        pad_seconds=2.0, duration=900.0,
    ) == [(8.0, 17.0), (98.0, 103.0)]


def test_merge_clamps_to_clip_bounds() -> None:
    assert merge_spans(
        [(0.5, 1.0), (899.0, 950.0)], pad_seconds=3.0, duration=900.0,
    ) == [(0.0, 4.0), (896.0, 900.0)]


def test_merge_empty_and_zero_duration() -> None:
    assert merge_spans([], pad_seconds=3.0, duration=900.0) == []
    assert merge_spans([(1.0, 2.0)], pad_seconds=3.0, duration=0.0) == []


def test_point_priors_become_padded_windows() -> None:
    # A tick verdict is a point (t, t); padding turns it into a window.
    assert merge_spans([(60.0, 60.0)], pad_seconds=3.0, duration=900.0) == [
        (57.0, 63.0)
    ]


# ── manager wiring: fakes ───────────────────────────────────────────

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
        self.paths: list[Path] = []
        self.metadata: list[dict] = []

    async def __call__(self, path, key, *, metadata, **kw) -> str:
        self.paths.append(Path(path))
        self.metadata.append(dict(metadata))
        return key


def _settings():
    return SimpleNamespace(
        b2_key_id="k", b2_app_key="s", b2_bucket="b",
        b2_endpoint_url="e", b2_region="r",
        data_collection_max_concurrent=8,
        data_collection_redaction_enabled=True,
    )


def _plan(spans):
    return SimpleNamespace(
        spans=spans, chunk_hits=len(spans),
        chunks_screened=1,
    )


def _mgr(tmp_path, *, span_detector, redactor, uploader=None):
    return RecorderManager(
        settings=_settings(),
        temp_dir=tmp_path,
        encoder_factory=_Enc,
        muxer=_Muxer(),
        uploader=uploader or _Uploader(),
        clock=lambda: 0.0,
        utcnow=lambda: _UTC,
        linger_seconds=0.0,
        retry_base_delay=0.0,
        sleep=_noop,
        span_detector=span_detector,
        redactor=redactor,
    )


async def _noop(_s=None) -> None:
    return None


def _ctx(cam):
    return RecorderCtx(
        caregiver_id=uuid4(), patient_id=uuid4(), camera_id=cam,
        room="Bedroom", gate="person", confidence=0.9,
    )


async def _run(mgr, cam):
    await mgr.on_tick(cam, True, ctx=_ctx(cam))
    await mgr.on_tick(cam, False, ctx=_ctx(cam))
    await mgr.on_tick(cam, False, ctx=_ctx(cam))
    await mgr.drain()


async def test_spans_found_uploads_the_redacted_file(tmp_path) -> None:
    calls = {}

    async def _detector(_p, **kw):
        return _plan([(1.0, 2.0)])

    async def _redactor(in_path, out_path, *, spans, **kw):
        calls["spans"] = spans
        Path(out_path).write_bytes(b"redacted")

    uploader = _Uploader()
    mgr = _mgr(
        tmp_path, span_detector=_detector, redactor=_redactor,
        uploader=uploader,
    )
    cam = uuid4()
    await _run(mgr, cam)

    assert calls["spans"] == [(1.0, 2.0)]
    assert uploader.paths and uploader.paths[0].name.endswith(".redacted.mp4")
    meta = uploader.metadata[0]
    assert meta["redacted"] == "true"
    assert meta["sensitive_spans"] == "[[1.0, 2.0]]"
    assert mgr.stats["clips_redacted"] == 1
    assert list(tmp_path.iterdir()) == []  # every temp cleaned


async def test_no_spans_uploads_the_plain_file(tmp_path) -> None:
    async def _detector(_p, **kw):
        return _plan([])

    async def _redactor(*a, **kw):
        raise AssertionError("redactor must not run with no spans")

    uploader = _Uploader()
    mgr = _mgr(
        tmp_path, span_detector=_detector, redactor=_redactor,
        uploader=uploader,
    )
    cam = uuid4()
    await _run(mgr, cam)

    assert uploader.paths and uploader.paths[0].name.endswith(".final.mp4")
    assert uploader.metadata[0]["redacted"] == "false"
    assert mgr.stats["clips_redacted"] == 0
    assert list(tmp_path.iterdir()) == []


async def test_detection_failure_drops_the_clip(tmp_path) -> None:
    async def _detector(_p, **kw):
        raise RuntimeError("screen down")

    async def _redactor(*a, **kw): ...

    uploader = _Uploader()
    mgr = _mgr(
        tmp_path, span_detector=_detector, redactor=_redactor,
        uploader=uploader,
    )
    cam = uuid4()
    await _run(mgr, cam)

    assert uploader.paths == []  # raw never left the box
    assert mgr.stats["dropped_redaction"] == 1
    assert list(tmp_path.iterdir()) == []


async def test_redaction_failure_drops_the_clip(tmp_path) -> None:
    async def _detector(_p, **kw):
        return _plan([(0.0, 1.0)])

    async def _redactor(*a, **kw):
        raise RuntimeError("censor died")

    uploader = _Uploader()
    mgr = _mgr(
        tmp_path, span_detector=_detector, redactor=_redactor,
        uploader=uploader,
    )
    cam = uuid4()
    await _run(mgr, cam)

    assert uploader.paths == []
    assert mgr.stats["dropped_redaction"] == 1
    assert list(tmp_path.iterdir()) == []
