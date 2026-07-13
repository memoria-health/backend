"""Tick-level tests for the upstream identity-suppression gate
(camera_ingest), detection v2 P2 T1.

The gate skips the LLM when everyone in frame is verified not-patient.
v2 adds `attribution_ran` to the condition: with always-on tracking the
pipeline now runs (reid_ran=True) even when identity labels were
impossible (reid_enabled off / no enrollment), so a tick where
attribution could not run must NEVER suppress — no matter what the
track labels claim (body-gallery pre-labels can mark tracks OTHER with
zero in-clip face evidence).

Uses the same real-tick harness as test_rapid_seizure_path: real DB
rows, real frame buffer + ffmpeg, VLM seam faked.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import src.services.detection.camera_ingest as ci
from src.db.analysis import Analysis
from src.db.session import get_sessionmaker
from src.repos import face_embedding_repo
from src.services.detection.camera_ingest import CameraIngest, _RecorderDecision
from src.services.detection.motion_cache import MotionCache
from src.services.reid.inference import IdentityResult
from src.settings import get_settings
from test.fakes.fake_vlm import FakeVLM
from test.integration.test_reid_safety_invariant import (
    _ready_embedding,
    _setup_patient_with_camera,
)

W, H = 160, 120

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


class _FixedReID:
    """Returns one crafted IdentityResult for every tick."""

    def __init__(self, result: IdentityResult) -> None:
        self._result = result

    async def label_clip(self, **kwargs):
        return self._result


def _i420(y: np.ndarray) -> bytes:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _push_still_frames(ingest: CameraIngest, n: int = 24) -> None:
    t0 = time.monotonic() - n / 8.0
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        ingest.frame_buffer.push(
            width=W, height=H, data=_i420(y), captured_at=t0 + i / 8.0,
        )


async def _seed_ready_enrollment(session: AsyncSession, patient_id) -> None:
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="x.jpg",
    )
    await face_embedding_repo.mark_ready(
        session, embedding_id=row.id,
        embedding=_ready_embedding(7), det_score=0.9,
    )
    await session.commit()


def _make_ingest(camera_id, patient_id, monkeypatch, *, identity, main):
    settings = get_settings().model_copy(update={
        "motion_overlay_enabled": False,
    })
    monkeypatch.setattr(ci, "run_vlm", main)
    cache = MotionCache(ttl_seconds=60)
    cache.put(camera_id, in_motion=True)
    return CameraIngest(
        camera_id=camera_id, patient_id=patient_id,
        sessionmaker=get_sessionmaker(), motion_cache=cache,
        settings=settings, reid=_FixedReID(identity),
    )


async def _last_analysis(session: AsyncSession, patient_id) -> Analysis:
    rows = (await session.execute(
        select(Analysis).where(Analysis.patient_id == patient_id)
        .order_by(Analysis.created_at)
    )).scalars().all()
    assert rows, "tick must persist an analysis row"
    return rows[-1]


@pytest.mark.asyncio
async def test_verified_other_with_attribution_suppresses_llm(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """The suppression path itself: attribution ran, faces were seen,
    none matched, enrollment exists → LLM skipped, synthetic row says
    identity_not_patient."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _seed_ready_enrollment(session, patient_id)

    identity = IdentityResult(
        reid_ran=True,
        patient_in_clip=False,
        confidence=0.21,
        best_frame_index=None,
        face_count=3,
        people_present_reid=True,
        people_count_reid=1,
        attribution_ran=True,
        faces_enrolled=True,
    )
    main = FakeVLM()
    ingest = _make_ingest(camera_id, patient_id, monkeypatch,
                          identity=identity, main=main)
    _push_still_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert main.calls == [], "verified not-patient must skip the LLM"
    row = await _last_analysis(session, patient_id)
    assert row.provider == "skipped"
    assert row.detector_fields["skipped_reason"] == "identity_not_patient"


@pytest.mark.asyncio
async def test_no_attribution_never_suppresses_even_with_other_labels(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """THE T1 SAFETY TEST. Tracking ran but attribution could not
    (flag off / gallery-less tick) — yet the track labels claim
    everyone is OTHER (the body-gallery pre-label shape). Enrollment
    even exists in the DB. The gate must refuse to suppress: without
    attribution this tick, 'verified other' is not verifiable."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _seed_ready_enrollment(session, patient_id)

    identity = IdentityResult(
        reid_ran=True,
        patient_in_clip=False,
        confidence=0.0,
        best_frame_index=None,
        face_count=0,
        people_present_reid=True,
        people_count_reid=1,
        attribution_ran=False,          # ← the new gate input
        has_only_verified_other_tracks=True,   # poisoned labels
        faces_enrolled=False,
    )
    main = FakeVLM()
    ingest = _make_ingest(camera_id, patient_id, monkeypatch,
                          identity=identity, main=main)
    _push_still_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 1, (
        "attribution_ran=False must bypass identity suppression — "
        "suppressing here silences detection for unenrolled/flag-off "
        "patients, the exact failure v2 always-on tracking forbids"
    )
    row = await _last_analysis(session, patient_id)
    assert row.provider != "skipped"
