"""Full-tick integration test for the RAPID seizure path (v2 P0.5).

Drives a REAL `CameraIngest._run_tick_inner` — real DB rows, real frame
buffer, real ffmpeg artifacts, real promotion — with the VLM seam faked
(`test/fakes/fake_vlm.py`). This is the first test that exercises the tick
end to end: gates → main call (without seizure) → RAPID call (with it) →
merged fields → event row.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import src.services.detection.camera_ingest as ci
from src.db.camera_detector import CameraDetector
from src.db.detector import Detector
from src.db.event import Event
from src.db.session import get_sessionmaker
from src.services.detection import catalog
from src.services.detection.camera_ingest import CameraIngest, _RecorderDecision
from src.services.detection.motion_cache import MotionCache
from src.services.reid.inference import IdentityResult
from src.settings import get_settings
from test.fakes.fake_vlm import FakeRapidProvider, FakeVLM
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera

W, H = 160, 120

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


class _FakeReID:
    async def label_clip(self, **kwargs):
        return IdentityResult.empty(faces_enrolled=False)


def _i420(y: np.ndarray) -> bytes:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _push_frames(ingest: CameraIngest, *, flicker: bool, n: int = 24) -> None:
    t0 = time.monotonic() - n / 8.0
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        if flicker:
            y[40:70, 60:90] = 190 if i % 2 else 60
        ingest.frame_buffer.push(
            width=W, height=H, data=_i420(y), captured_at=t0 + i / 8.0,
        )


async def _seed_seizure(session: AsyncSession, patient_id, camera_id) -> Detector:
    entry = catalog.by_key("seizure")
    now = datetime.now(timezone.utc)
    det = Detector(
        id=uuid4(), created_at=now, updated_at=now, patient_id=patient_id,
        name=entry.name, icon_name=entry.icon_name, description=entry.description,
        category=entry.category, source="preset", shape=entry.shape,
        where_scope=entry.where_scope, when_scope=entry.when_scope,
        sensitivity="standard", is_on=True, custom_prompt=None,
        identifier=entry.key, prompt_fragment=entry.prompt_fragment,
        schema_json=entry.schema_json, is_builtin=True,
    )
    session.add(det)
    await session.flush()
    session.add(CameraDetector(
        camera_id=camera_id, detector_id=det.id, created_at=now,
    ))
    await session.commit()
    return det


async def _make_ingest(camera_id, patient_id, monkeypatch, *, main, rapid):
    settings = get_settings().model_copy(update={
        "rapid_seizure_enabled": True,
        "motion_overlay_enabled": False,
    })
    monkeypatch.setattr(ci, "run_vlm", main)
    monkeypatch.setattr(ci, "gemini_provider", rapid)

    async def _no_archive(*a, **k):
        return None

    monkeypatch.setattr("src.services.clip_archival.archive_event", _no_archive)
    cache = MotionCache(ttl_seconds=60)
    cache.put(camera_id, in_motion=True)
    return CameraIngest(
        camera_id=camera_id, patient_id=patient_id,
        sessionmaker=get_sessionmaker(), motion_cache=cache,
        settings=settings, reid=_FakeReID(),
    )


@pytest.mark.asyncio
async def test_flag_on_tick_splits_calls_and_fires_seizure_event(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    det = await _seed_seizure(session, patient_id, camera_id)

    main = FakeVLM()
    rapid = FakeRapidProvider({
        "seizure_detected": True, "seizure_confidence": 0.9,
        "seizure_reasoning": "sustained rhythmic whole-body shaking",
        "rapid_motion_description": "high-amplitude oscillation",
        "people_present": True,   # must be filtered, never merged
    })
    ingest = await _make_ingest(camera_id, patient_id, monkeypatch,
                                main=main, rapid=rapid)
    _push_frames(ingest, flicker=True)

    await ingest._run_tick_inner(_RecorderDecision())

    # Main call ran WITHOUT seizure; RAPID call ran WITH it, on flash settings.
    assert len(main.calls) == 1
    main_idents = [d.identifier for d in main.calls[0]["detectors"]]
    assert "seizure" not in main_idents
    assert len(rapid.calls) >= 1
    rc = rapid.calls[0]
    assert rc["settings"].gemini_model == ingest._settings.rapid_model
    assert [d.identifier for d in rc["detectors"]] == ["rapid_motion", "seizure"]

    # The merged verdict promoted through the UNCHANGED path into an event.
    events = (await session.execute(
        select(Event).where(Event.patient_id == patient_id)
    )).scalars().all()
    assert any(e.detector_id == det.id for e in events), \
        "seizure event must fire via the normal promotion path"


@pytest.mark.asyncio
async def test_flag_on_still_scene_never_spends_the_rapid_call(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    det = await _seed_seizure(session, patient_id, camera_id)

    main = FakeVLM()
    rapid = FakeRapidProvider({"seizure_detected": True})
    ingest = await _make_ingest(camera_id, patient_id, monkeypatch,
                                main=main, rapid=rapid)
    _push_frames(ingest, flicker=False)   # person present, nothing shaking

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 1           # main call still runs
    assert rapid.calls == []              # churn gate stayed closed — $0
    events = (await session.execute(
        select(Event).where(Event.patient_id == patient_id)
    )).scalars().all()
    assert not any(e.detector_id == det.id for e in events)
