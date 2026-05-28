"""Tests for `services.clip_annotate.annotate_clip_for_display`.

We don't exercise the real InsightFace path here (slow + opaque). The
function's failure-mode guarantees are what matter: any exception or
missing precondition returns the input bytes unchanged, so the
caregiver never loses a clip to display-layer issues.

The "real" annotation behavior (boxes follow the patient frame by
frame) needs real video + a real gallery to verify, and is left to
the manual smoke test in section 7 of the plan.
"""

from __future__ import annotations

import shutil
from uuid import uuid4

import numpy as np
import pytest

from src.db.session import get_sessionmaker
from src.services import clip_annotate
from src.services.clip_annotate import annotate_clip_for_display
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.mp4_remux import frames_to_mp4
from src.settings import Settings, get_settings


_pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)


def _i420_blob(width: int, height: int) -> bytes:
    return b"\x80" * (width * height * 3 // 2)


async def _make_test_clip() -> bytes:
    """A minimal 1s MP4 — enough for cv2 to open and iterate."""
    import time
    base = time.monotonic()
    frames = [
        FrameSnapshot(
            width=160, height=120,
            data=_i420_blob(160, 120),
            captured_at=base + i / 5,
        )
        for i in range(5)
    ]
    return await frames_to_mp4(frames, fps=5)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_annotation_disabled_returns_input_unchanged() -> None:
    """Settings flag off → no work done, no decode, original bytes
    returned. The annotation step is fully opt-out."""
    base = get_settings()
    overrides = base.model_dump()
    overrides["reid_clip_annotation_enabled"] = False
    settings = Settings(**overrides)

    clip = await _make_test_clip()
    out = await annotate_clip_for_display(
        clip, patient_id=uuid4(),
        sessionmaker=get_sessionmaker(), settings=settings,
    )
    assert out is clip, "disabled flag must short-circuit before any work"


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_no_enrollment_returns_input_unchanged() -> None:
    """A patient with no ready embeddings has no gallery to match
    against. Return input unchanged rather than draw nothing on every
    frame (cheaper, and the resulting MP4 is byte-identical so any
    downstream byte-comparing test stays stable)."""
    base = get_settings()
    overrides = base.model_dump()
    overrides["reid_clip_annotation_enabled"] = True
    settings = Settings(**overrides)

    clip = await _make_test_clip()
    out = await annotate_clip_for_display(
        clip, patient_id=uuid4(),  # uuid that has no enrollment
        sessionmaker=get_sessionmaker(), settings=settings,
    )
    assert out == clip, "no enrollment must short-circuit annotation"


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_internal_failure_returns_input_unchanged(monkeypatch) -> None:
    """If the cv2/ffmpeg layer crashes mid-annotation, the original
    clip is returned. The caregiver never loses a clip to a display-
    layer bug."""
    base = get_settings()
    overrides = base.model_dump()
    overrides["reid_clip_annotation_enabled"] = True
    settings = Settings(**overrides)

    # Force has_ready_enrollment to return True so we proceed past the
    # short-circuit. Then break the gallery loader so the cv2/ffmpeg
    # path tries to run with an empty gallery (which is also safe to
    # short-circuit on, returning input).
    async def fake_has_enrollment(*a, **kw):
        return True
    monkeypatch.setattr(clip_annotate, "has_ready_enrollment", fake_has_enrollment)
    async def fake_load(*a, **kw):
        raise RuntimeError("simulated gallery load failure")
    monkeypatch.setattr(clip_annotate, "_load_gallery", fake_load)

    clip = await _make_test_clip()
    out = await annotate_clip_for_display(
        clip, patient_id=uuid4(),
        sessionmaker=get_sessionmaker(), settings=settings,
    )
    assert out == clip, "internal failure must return raw clip, never raise"


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_empty_gallery_returns_input_unchanged(monkeypatch) -> None:
    """Defense in depth: even if has_ready_enrollment lies and returns
    True, an empty gallery vector must short-circuit (we'd have
    nothing to match against and would burn no boxes anywhere)."""
    base = get_settings()
    overrides = base.model_dump()
    overrides["reid_clip_annotation_enabled"] = True
    settings = Settings(**overrides)

    async def fake_has_enrollment(*a, **kw):
        return True
    monkeypatch.setattr(clip_annotate, "has_ready_enrollment", fake_has_enrollment)
    async def fake_load(*a, **kw):
        return np.empty((0, 512), dtype=np.float32)
    monkeypatch.setattr(clip_annotate, "_load_gallery", fake_load)

    clip = await _make_test_clip()
    out = await annotate_clip_for_display(
        clip, patient_id=uuid4(),
        sessionmaker=get_sessionmaker(), settings=settings,
    )
    assert out == clip
