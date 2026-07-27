"""Tick-level tests for the P2 signals wiring (detection v2 P2 T4).

Three behavior changes land in T4 — everything else is stash-only:

  1. `motion_in_clip` = SAD hint OR backend motion (gateway cameras
     that never publish SAD packets get motion gating at last).
  2. Closed-state wake: with no SAD packet, pixels moving above the
     floor run the full tick; a still scene keeps skipping.
  3. Presence union: the tracker can ADD presence the LLM missed
     (union — it can never deny presence).

Same real-tick harness as the suppression tests: real DB rows, real
frame buffer + ffmpeg, VLM + reid seams faked.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

import src.services.detection.camera_ingest as ci
from src.db.analysis import Analysis
from src.db.camera import Camera
from src.db.session import get_sessionmaker
from src.services.detection.camera_ingest import CameraIngest, _RecorderDecision
from src.services.detection.motion_cache import MotionCache
from src.services.reid.body_pipeline import TrackObservation
from src.services.reid.result import IdentityResult
from src.settings import get_settings
from test.fakes.fake_vlm import FakeVLM
from test.integration.test_identity_suppression_gate import _FixedReID
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera

W, H = 160, 120

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


def _i420(y: np.ndarray) -> bytes:
    uv = np.full((H // 2) * (W // 2) * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _push_frames(ingest: CameraIngest, *, flicker: bool, n: int = 24) -> None:
    """Pushes at 8 fps into a 4 fps buffer (rapid flag off here), so the
    buffer's min-gap decimation keeps ~every other frame. The flicker
    period is therefore 4 pushed frames (1 Hz square wave) — a faster
    i%2 flicker would alias to a CONSTANT scene after decimation."""
    t0 = time.monotonic() - n / 8.0
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        if flicker:
            y[30:90, 40:120] = 200 if (i // 4) % 2 else 60
        ingest.frame_buffer.push(
            width=W, height=H, data=_i420(y), captured_at=t0 + i / 8.0,
        )


def _make_ingest(
    camera_id, patient_id, monkeypatch, *, identity, main, sad: bool,
):
    settings = get_settings().model_copy(update={
        "motion_overlay_enabled": False,
        # This suite pins the LEGACY branch; a developer .env with the
        # soak's v2 flags on must not leak into it.
        "detection_v2_scheduler_enabled": False,
        "detection_v2_decision_enabled": False,
    })
    monkeypatch.setattr(ci, "run_vlm", main)
    cache = MotionCache(ttl_seconds=60)
    if sad:
        cache.put(camera_id, in_motion=True)
    return CameraIngest(
        camera_id=camera_id, patient_id=patient_id,
        sessionmaker=get_sessionmaker(), motion_cache=cache,
        settings=settings, reid=_FixedReID(identity),
    )


async def _close_camera(session: AsyncSession, camera_id) -> None:
    await session.execute(
        update(Camera).where(Camera.id == camera_id).values(state="closed")
    )
    await session.commit()


async def _last_analysis(session: AsyncSession, patient_id) -> Analysis:
    rows = (await session.execute(
        select(Analysis).where(Analysis.patient_id == patient_id)
        .order_by(Analysis.created_at)
    )).scalars().all()
    assert rows, "tick must persist an analysis row"
    return rows[-1]


@pytest.mark.asyncio
async def test_open_tick_stashes_signals_and_motion_src(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """Every analyses row now carries the sig_* block + motion_src.
    SAD on, still pixels, reid down ⇒ motion_src is the hint alone and
    the unknowable presence signals are absent, not false."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    main = FakeVLM()
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=main, sad=True,
    )
    _push_frames(ingest, flicker=False)

    await ingest._run_tick_inner(_RecorderDecision())

    row = await _last_analysis(session, patient_id)
    f = row.detector_fields
    assert f["motion_src"] == "sad"
    assert "sig_env_motion" in f and "sig_frames" in f
    assert "sig_person_present" not in f          # tracking didn't run
    assert row.motion_in_clip is True             # the hint carries it
    # Flag-off rows must carry NO scheduler keys — pins that the tick
    # actually took the legacy branch (T3/T4 review test gap).
    assert not any(k.startswith("sched_") for k in f)


@pytest.mark.asyncio
async def test_backend_motion_sets_motion_in_clip_without_sad(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """The gateway-camera fix: no SAD packet ever arrives, pixels move
    ⇒ motion_in_clip=True with motion_src=backend."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    main = FakeVLM()
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=main, sad=False,
    )
    _push_frames(ingest, flicker=True)

    await ingest._run_tick_inner(_RecorderDecision())

    row = await _last_analysis(session, patient_id)
    assert row.motion_in_clip is True
    assert row.detector_fields["motion_src"] == "backend"


@pytest.mark.asyncio
async def test_presence_union_tracker_holds_room_open(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """Tracker sees a body the LLM missed (face away / occluded) ⇒
    presence is True via the union, attributed to the tracker."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    # Real TrackObservations so sig_tracks (the list shape) round-trips
    # Postgres JSONB in at least one test (T3/T4 review finding).
    tracks = tuple(
        TrackObservation(
            track_id=tid, label="UNKNOWN", confidence=0.0,
            bbox_series=((0, (10.0 * tid, 10.0, 10.0 * tid + 40.0, 90.0)),),
            face_locked=False, pre_labeled=False,
        )
        for tid in (1, 2)
    )
    identity = IdentityResult(
        reid_ran=True, patient_in_clip=False, confidence=0.0,
        best_frame_index=None, face_count=0,
        people_present_reid=True, people_count_reid=2,
        attribution_ran=False, tracks=tracks, faces_enrolled=False,
    )
    main = FakeVLM(people_present=False, people_count=0)
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=identity, main=main, sad=True,
    )
    _push_frames(ingest, flicker=False)

    await ingest._run_tick_inner(_RecorderDecision())

    row = await _last_analysis(session, patient_id)
    assert row.people_present is True
    assert row.detector_fields["people_count"] == 2
    assert row.detector_fields["people_present_src"] == "tracker_only"
    # The track list survived the JSONB round-trip.
    assert [t["id"] for t in row.detector_fields["sig_tracks"]] == [1, 2]


@pytest.mark.asyncio
async def test_closed_camera_wakes_on_backend_motion_without_sad(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """Closed + no SAD used to be an unconditional skip — a gateway
    camera could never wake itself. Backend motion now runs the tick."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _close_camera(session, camera_id)
    main = FakeVLM()
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=main, sad=False,
    )
    _push_frames(ingest, flicker=True)

    await ingest._run_tick_inner(_RecorderDecision())

    assert len(main.calls) == 1, "backend motion must wake the closed camera"
    row = await _last_analysis(session, patient_id)
    assert row.provider != "skipped"
    # The wake is stamped on the row (not just logs) and the tick can
    # never record itself as motionless after being woken by motion.
    assert row.detector_fields["woke_by_backend"] is True
    assert row.detector_fields["wake_env_motion"] > 0
    assert row.motion_in_clip is True


@pytest.mark.asyncio
async def test_closed_camera_still_skips_when_pixels_still(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _close_camera(session, camera_id)
    main = FakeVLM()
    ingest = _make_ingest(
        camera_id, patient_id, monkeypatch,
        identity=IdentityResult.empty(), main=main, sad=False,
    )
    _push_frames(ingest, flicker=False)

    await ingest._run_tick_inner(_RecorderDecision())

    assert main.calls == []
    row = await _last_analysis(session, patient_id)
    assert row.provider == "skipped"
    f = row.detector_fields
    assert f["skipped_reason"] == "camera_closed"
    assert f["motion_src"] == "none"
    # Motion consolidation (2026-07-17): closed-skip telemetry is the
    # block-motion verdict now (wake_block_max), not the old whole-
    # frame sig_env stash — block motion IS the wake authority.
    assert "wake_block_max" in f
