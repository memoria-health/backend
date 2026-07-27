"""Full-tick integration test for the RAPID path as a regular bucket
call (detection v2, Ledger row 5 — the P0 standalone lane and its flag
are gone; the scheduler flag owns the whole path).

Drives a REAL `CameraIngest._run_tick_inner` — real DB rows, real frame
buffer, real rapid-energy gate, real ffmpeg artifacts, real promotion —
with the VLM seam faked (`test/fakes/fake_vlm.py`): scene call →
FakeVLM, overlay flash call → FakeRapidProvider.
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
from src.services.identity.result import IdentityResult
from src.settings import get_settings
from test.fakes.fake_vlm import FakeRapidProvider, FakeVLM
from test.integration.identity_helpers import _setup_patient_with_camera

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
        category=entry.category, source="preset",
        alert_trigger=entry.alert_trigger, is_sustained=entry.is_sustained,
        where_rooms=(None if entry.where_scope == "ALL ROOMS" else [entry.where_scope]),
        when_scope=entry.when_scope,
        sensitivity="standard", is_on=True, custom_prompt=None,
        identifier=entry.key, prompt_fragment=entry.prompt_fragment,
        schema_json=entry.schema_json,
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
        "detection_v2_scheduler_enabled": True,
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
async def test_seizure_runs_as_overlay_bucket_call_and_fires_event(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    det = await _seed_seizure(session, patient_id, camera_id)

    main = FakeVLM(usage={"input": 13200, "output": 250})
    rapid = FakeRapidProvider({
        "seizure_detected": True, "seizure_confidence": 0.9,
        "seizure_reasoning": "sustained rhythmic whole-body shaking",
        "rapid_motion_description": "high-amplitude oscillation",
        "people_present": True,   # not a member key — must never merge
    }, usage={"input": 20900, "output": 300})
    ingest = await _make_ingest(camera_id, patient_id, monkeypatch,
                                main=main, rapid=rapid)
    # Steady state: a prior tick already retargeted the buffer to native
    # rate (the loadout carries a RAPID detector) — do it up front so
    # the 8 fps flicker survives retention un-aliased.
    ingest.frame_buffer.retarget(ingest._settings.rapid_fps)
    _push_frames(ingest, flicker=True)

    await ingest._run_tick_inner(_RecorderDecision())

    # Scene call ran WITHOUT seizure; the overlay call ran WITH it, on
    # the validated recipe (flash model, native fps, tint note, prompt
    # swap, describe-then-decide preamble).
    assert len(main.calls) == 1
    main_idents = [d.identifier for d in main.calls[0]["detectors"]]
    assert "seizure" not in main_idents
    assert len(rapid.calls) == 1
    rc = rapid.calls[0]
    assert rc["settings"].gemini_model == ingest._settings.rapid_model
    assert rc["settings"].gemini_video_fps == str(ingest._settings.rapid_fps)
    assert rc["settings"].motion_overlay_enabled is True
    assert [d.identifier for d in rc["detectors"]] == ["rapid_motion", "seizure"]
    assert "MOTION-OVERLAY" in rc["detectors"][1].prompt_fragment

    # The tick kept the buffer at native rate (loadout still has RAPID).
    assert ingest.frame_buffer.target_fps == ingest._settings.rapid_fps

    # The merged verdict promoted through the UNCHANGED path into an event.
    events = (await session.execute(
        select(Event).where(Event.patient_id == patient_id)
    )).scalars().all()
    assert any(e.detector_id == det.id for e in events), \
        "seizure event must fire via the normal promotion path"

    # Token ledger: one summed main-path total; the overlay call's spend
    # itemized in sched_calls (rapid_*_tokens row fields are gone).
    from src.db.analysis import Analysis

    rows = (await session.execute(
        select(Analysis).where(Analysis.patient_id == patient_id)
    )).scalars().all()
    assert rows
    fields = rows[-1].detector_fields
    assert fields["vlm_input_tokens"] == 13200 + 20900
    assert fields["vlm_output_tokens"] == 250 + 300
    overlay_entries = [
        c for c in fields["sched_calls"] if "overlay" in str(c.get("key", ""))
    ]
    assert overlay_entries and overlay_entries[0]["input_tokens"] == 20900
    assert fields["rapid_gate_active_fraction"] > 0


@pytest.mark.asyncio
async def test_still_scene_never_spends_the_overlay_call(
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
    ingest.frame_buffer.retarget(ingest._settings.rapid_fps)
    _push_frames(ingest, flicker=False)   # person present, nothing shaking

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 1           # scene call still runs
    assert rapid.calls == []              # rapid-energy gate closed — $0
    events = (await session.execute(
        select(Event).where(Event.patient_id == patient_id)
    )).scalars().all()
    assert not any(e.detector_id == det.id for e in events)


@pytest.mark.asyncio
async def test_first_tick_retargets_buffer_and_degrades_honestly(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """Transition tick: the buffer still holds 4 fps history when the
    loadout first shows a RAPID detector. The tick must retarget the
    buffer for the NEXT window and, if the gate opens this tick, encode
    at the measured (lower) rate with the degraded marker — never crash,
    never time-compress."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _seed_seizure(session, patient_id, camera_id)

    main = FakeVLM()
    rapid = FakeRapidProvider({"seizure_detected": False})
    ingest = await _make_ingest(camera_id, patient_id, monkeypatch,
                                main=main, rapid=rapid)
    assert ingest.frame_buffer.target_fps == \
        ingest._settings.livekit_worker_target_fps
    _push_frames(ingest, flicker=True)    # 8fps pushes, 4fps retention

    await ingest._run_tick_inner(_RecorderDecision())

    # The tick saw the RAPID loadout and retargeted for next window.
    assert ingest.frame_buffer.target_fps == ingest._settings.rapid_fps
    assert len(main.calls) == 1           # scene call unaffected
