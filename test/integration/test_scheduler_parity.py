"""Full-tick integration for the v2 scheduler path (detection v2 P3 T4).

Drives a REAL `CameraIngest._run_tick_inner` with
`detection_v2_scheduler_enabled=True` — real DB rows (born-classified
P1 seeding), real frame buffer + ffmpeg, VLM seam faked.

Pins the D2 parity contract at the tick level: with reid down (no
tracks), the entire seeded loadout rides ONE wide call exactly like
flag-off — same detectors, same event promotion — and the row carries
the sched_* observability block. With tracks present, person-bucket
detectors split into per-track crop calls (D3) and their answers merge
back into the same single analyses row.
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
from src.db.camera_detector import CameraDetector
from src.db.detector import Detector
from src.db.event import Event
from src.db.session import get_sessionmaker
from src.services.detection.camera_ingest import CameraIngest, _RecorderDecision
from src.services.detection.motion_cache import MotionCache
from src.services.reid.body_pipeline import TrackObservation
from src.services.reid.inference import IdentityResult
from src.settings import get_settings
from test.fakes.fake_vlm import FakeVLM
from test.integration.test_identity_suppression_gate import _FixedReID
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera

W, H = 160, 120

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)

# The three seeded-for-adults presets spanning buckets: scene
# (fall), person (medication), and scene+sampled+needs_audio
# (agitation — pins the sampled duty at k=1 parity too).
_IDENTS = ("fall", "medication", "agitation")


def _i420(y: np.ndarray) -> bytes:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _push_frames(ingest: CameraIngest, n: int = 24) -> None:
    t0 = time.monotonic() - n / 8.0
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        y[30:90, 40:120] = 200 if (i // 4) % 2 else 60   # survives 4fps decimation
        ingest.frame_buffer.push(
            width=W, height=H, data=_i420(y), captured_at=t0 + i / 8.0,
        )


async def _assign_detectors(session: AsyncSession, patient_id, camera_id) -> dict:
    """Assign three seeded presets (scene + person + env buckets) to
    the camera; return ident → Detector."""
    from datetime import datetime, timezone
    rows = (await session.execute(
        select(Detector).where(
            Detector.patient_id == patient_id,
            Detector.identifier.in_(_IDENTS),
        )
    )).scalars().all()
    assert len(rows) == 3, "onboarding must have seeded these presets"
    now = datetime.now(timezone.utc)
    for r in rows:
        # Neutralize seeded time windows (medication ships "8 AM, 8 PM
        # ±1H"): this suite pins bucket-splitting mechanics, and the
        # when_scope gate (2026-07-14) would otherwise skip windowed
        # rows depending on the wall clock the test runs at.
        r.when_scope = "ALL DAY"
        session.add(CameraDetector(
            camera_id=camera_id, detector_id=r.id, created_at=now,
        ))
    await session.commit()
    return {r.identifier: r for r in rows}


def _make_ingest(camera_id, patient_id, monkeypatch, *, identity, main):
    settings = get_settings().model_copy(update={
        "detection_v2_scheduler_enabled": True,
        "motion_overlay_enabled": False,
    })
    monkeypatch.setattr(ci, "run_vlm", main)

    async def _no_archive(*a, **k):
        return None

    monkeypatch.setattr("src.services.clip_archival.archive_event", _no_archive)
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
    assert rows
    return rows[-1]


@pytest.mark.asyncio
async def test_flag_on_no_tracks_is_one_wide_call_with_event_parity(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """Reid down ⇒ every bucket degrades to the wide call: exactly one
    provider call carrying the full loadout (person + snapshot members
    included), fall still promotes to an event — parity with today."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    dets = await _assign_detectors(session, patient_id, camera_id)

    main = FakeVLM(
        {"fall_detected": True, "fall_confidence": 0.9,
         "fall_reasoning": "collapsed"},
        usage={"input": 5000, "output": 111},
    )
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=main,
    )
    _push_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 1, "no tracks ⇒ everything rides one wide call"
    called = {d.identifier for d in main.calls[0]["detectors"]}
    assert called == set(_IDENTS)

    events = (await session.execute(
        select(Event).where(Event.patient_id == patient_id)
    )).scalars().all()
    assert any(e.detector_id == dets["fall"].id for e in events)

    row = await _last_analysis(session, patient_id)
    f = row.detector_fields
    assert len(f["sched_calls"]) == 1
    assert set(f["sched_calls"][0]["detectors"]) == set(_IDENTS)
    assert f["vlm_input_tokens"] == 5000
    assert f["vlm_output_tokens"] == 111
    assert "sig_env_motion" in f            # P2 signals still stashed


@pytest.mark.asyncio
async def test_flag_on_no_assigned_detectors_still_writes_presence_row(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """F1 regression pin: a camera with nothing assigned must behave
    like flag-off — one empty-detector wide call whose people_present
    answer drives the state machine and lands a row, never a dropped
    tick."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    main = FakeVLM()
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=main,
    )
    _push_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 1
    assert main.calls[0]["detectors"] == []
    row = await _last_analysis(session, patient_id)
    assert row.provider != "skipped"
    assert row.people_present is True
    assert len(row.detector_fields["sched_calls"]) == 1


@pytest.mark.asyncio
async def test_flag_on_with_track_splits_person_crop_call_and_merges(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _assign_detectors(session, patient_id, camera_id)

    track = TrackObservation(
        track_id=1, label="UNKNOWN", confidence=0.0,
        bbox_series=tuple((i, (40.0, 30.0, 120.0, 110.0)) for i in range(10)),
        face_locked=False, pre_labeled=False,
    )
    identity = IdentityResult(
        reid_ran=True, patient_in_clip=False, confidence=0.0,
        best_frame_index=None, face_count=0,
        people_present_reid=True, people_count_reid=1,
        attribution_ran=False, tracks=(track,), faces_enrolled=False,
    )
    main = FakeVLM({"medication_detected": True, "medication_confidence": 0.8,
                    "medication_reasoning": "took pills"})
    ingest = _make_ingest(camera_id, patient_id, monkeypatch,
                          identity=identity, main=main)
    _push_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 2, "person bucket must split into its own crop call"
    scene_idents = {d.identifier for d in main.calls[0]["detectors"]}
    crop_idents = [d.identifier for d in main.calls[1]["detectors"]]
    assert scene_idents == {"fall", "agitation"}
    assert crop_idents == ["medication"]
    assert main.calls[1]["mp4"] != main.calls[0]["mp4"], "crop artifact differs"

    row = await _last_analysis(session, patient_id)
    f = row.detector_fields
    assert len(f["sched_calls"]) == 2
    assert f["medication_detected"] is True        # crop answer merged
    assert "sched_degraded_audio" not in f
    assert row.people_present is True
