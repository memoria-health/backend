"""Unit: `label_clip` gating (detection v2 P2 T1).

Body tracking must ALWAYS run — `reid_enabled` and enrollment gate face
ATTRIBUTION only. These tests pin the new contract at the `label_clip`
seam: the inner pipeline is invoked on every call (no early-out), the
gallery it receives reflects attribution eligibility, and the
timeout/error fallbacks report eligibility truthfully.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from src.services.reid.inference import ReIDInference
from src.services.reid.result import IdentityResult


def _settings(*, reid_enabled: bool, timeout: float = 5.0) -> SimpleNamespace:
    return SimpleNamespace(
        reid_enabled=reid_enabled,
        reid_timeout_seconds=timeout,
    )


def _sentinel() -> IdentityResult:
    return IdentityResult(
        reid_ran=True,
        patient_in_clip=False,
        confidence=0.0,
        best_frame_index=None,
        face_count=0,
        people_present_reid=True,
        people_count_reid=1,
    )


def _make(reid_enabled: bool, gallery_rows: int | None) -> tuple[ReIDInference, dict]:
    """ReIDInference with `_get_gallery` and `_label_clip_inner` stubbed.

    gallery_rows=None means "_get_gallery must not be called" (the
    flag-off path skips the DB entirely).
    """
    svc = ReIDInference(
        sessionmaker=None,  # never touched: gallery + inner are stubbed
        settings=_settings(reid_enabled=reid_enabled),
        body_gallery=None,
    )
    seen: dict = {"inner_calls": [], "gallery_calls": 0}

    async def fake_get_gallery(patient_id):
        seen["gallery_calls"] += 1
        if gallery_rows is None:
            raise AssertionError("_get_gallery must not be called when reid_enabled=False")
        return np.zeros((gallery_rows, 512), dtype=np.float32)

    async def fake_inner(*, mp4_bytes, patient_id, gallery, camera_id=None, **_):
        seen["inner_calls"].append(gallery)
        return _sentinel()

    svc._get_gallery = fake_get_gallery  # type: ignore[method-assign]
    svc._label_clip_inner = fake_inner  # type: ignore[method-assign]
    return svc, seen


@pytest.mark.asyncio
async def test_flag_off_still_runs_tracking_with_empty_gallery() -> None:
    """reid_enabled=False no longer kills the pipeline: the inner run
    happens anyway, gallery-less, and the DB is never consulted."""
    svc, seen = _make(reid_enabled=False, gallery_rows=None)
    result = await svc.label_clip(mp4_bytes=b"x", patient_id=uuid4())
    assert len(seen["inner_calls"]) == 1
    assert seen["inner_calls"][0].size == 0
    assert seen["gallery_calls"] == 0
    assert result.reid_ran is True  # the sentinel passed through


@pytest.mark.asyncio
async def test_flag_on_but_no_enrollment_runs_gallery_less() -> None:
    svc, seen = _make(reid_enabled=True, gallery_rows=0)
    await svc.label_clip(mp4_bytes=b"x", patient_id=uuid4())
    assert len(seen["inner_calls"]) == 1
    assert seen["inner_calls"][0].size == 0
    assert seen["gallery_calls"] == 1


@pytest.mark.asyncio
async def test_flag_on_and_enrolled_passes_gallery_through() -> None:
    svc, seen = _make(reid_enabled=True, gallery_rows=3)
    await svc.label_clip(mp4_bytes=b"x", patient_id=uuid4())
    assert len(seen["inner_calls"]) == 1
    assert seen["inner_calls"][0].shape == (3, 512)


@pytest.mark.asyncio
async def test_timeout_reports_eligibility_truthfully() -> None:
    """Timeout with an enrolled, flag-on patient: empty result must say
    faces_enrolled=True (eligible, but the tick failed) and never claim
    the pipeline or attribution ran."""
    svc, _ = _make(reid_enabled=True, gallery_rows=2)
    svc._settings = _settings(reid_enabled=True, timeout=0.01)

    async def slow_inner(*, mp4_bytes, patient_id, gallery, **_):
        await asyncio.sleep(1.0)

    svc._label_clip_inner = slow_inner  # type: ignore[method-assign]
    result = await svc.label_clip(mp4_bytes=b"x", patient_id=uuid4())
    assert result.reid_ran is False
    assert result.attribution_ran is False
    assert result.faces_enrolled is True


@pytest.mark.asyncio
async def test_error_with_flag_off_reports_not_enrolled() -> None:
    svc, _ = _make(reid_enabled=False, gallery_rows=None)

    async def boom(*, mp4_bytes, patient_id, gallery, **_):
        raise RuntimeError("pipeline exploded")

    svc._label_clip_inner = boom  # type: ignore[method-assign]
    result = await svc.label_clip(mp4_bytes=b"x", patient_id=uuid4())
    assert result.reid_ran is False
    assert result.attribution_ran is False
    assert result.faces_enrolled is False


def test_empty_result_never_claims_attribution() -> None:
    assert IdentityResult.empty().attribution_ran is False
    assert IdentityResult.empty(faces_enrolled=True).attribution_ran is False
