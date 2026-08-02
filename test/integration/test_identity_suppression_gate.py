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
from uuid import uuid4

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
from src.services.detection.patient_presence import TIER_FACE, PatientPresence
from src.services.identity.observations import LABEL_OTHER, TrackObservation
from src.services.identity.result import IdentityResult
from src.settings import get_settings
from test.fakes.fake_vlm import FakeVLM
from test.integration.identity_helpers import (
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


def _make_ingest(
    camera_id, patient_id, monkeypatch, *, identity, main,
    patient_presence=None,
):
    settings = get_settings().model_copy(update={
        # Don't inherit the live soak's VLM kill-switch — these tests
        # assert on whether the LLM was called.
        "reid_soak_only": False,
    })
    monkeypatch.setattr(ci, "run_vlm", main)
    cache = MotionCache(ttl_seconds=60)
    cache.put(camera_id, in_motion=True)
    return CameraIngest(
        camera_id=camera_id, patient_id=patient_id,
        sessionmaker=get_sessionmaker(), motion_cache=cache,
        settings=settings, reid=_FixedReID(identity),
        patient_presence=patient_presence,
    )


async def _last_analysis(session: AsyncSession, patient_id) -> Analysis:
    rows = (await session.execute(
        select(Analysis).where(Analysis.patient_id == patient_id)
        .order_by(Analysis.created_at)
    )).scalars().all()
    assert rows, "tick must persist an analysis row"
    return rows[-1]


# REMOVED 2026-07-22 (bind-once-hold purge): the old
# test_verified_other_with_attribution_suppresses_llm asserted that a
# face-verified OTHER track makes the ledger derive
# has_only_verified_other_tracks=True and SKIP the LLM. The bind-once-hold
# ledger has no OTHER label — everyone not-patient is UNKNOWN — so that
# flag is always False and the cross-camera "verified not-patient"
# suppression gate no longer fires: an occupied room always analyzes
# (the safe direction; it fixes the "living room went dark while the
# patient sat in the bedroom" bug). The three tests below still guard the
# invariant that we NEVER suppress on unknowns / an unlocated patient /
# no-attribution.


@pytest.mark.asyncio
async def test_detected_but_unverified_face_does_not_suppress(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """THE 2026-07-22 LIVE BUG: patient bound in ANOTHER room, and a
    face is DETECTED here (face_count>0) but never matched — the tracks
    are UNKNOWN, not verified-OTHER. This is the aunt/uncle at couch
    distance. The gate must NOT suppress: a detected-but-unmatchable
    face is not a confirmation that they aren't the patient, and
    silencing the room on it made the living room go dark all evening."""
    patient_id, camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    await _seed_ready_enrollment(session, patient_id)

    identity = IdentityResult(
        reid_ran=True,
        patient_in_clip=False,
        confidence=0.21,
        best_frame_index=None,
        face_count=3,                          # a face WAS detected...
        people_present_reid=True,
        people_count_reid=1,
        attribution_ran=True,
        has_only_verified_other_tracks=False,  # ...but nobody verified
        faces_enrolled=True,
    )
    # Patient positively held ELSEWHERE — the old gate would suppress.
    presence = PatientPresence()
    presence.claim(uuid4(), tier=TIER_FACE, confidence=0.9)
    main = FakeVLM()
    ingest = _make_ingest(camera_id, patient_id, monkeypatch,
                          identity=identity, main=main,
                          patient_presence=presence)
    _push_still_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert main.calls, (
        "unknown (detected but unverified) people must keep the room "
        "analyzing even when the patient is confirmed elsewhere"
    )
    row = await _last_analysis(session, patient_id)
    assert row.detector_fields.get("skipped_reason") != "identity_not_patient"


@pytest.mark.asyncio
async def test_suppression_refused_when_patient_unlocated(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """THE LIVING-ROOM SAFETY TEST (soak #2, 2026-07-19): faces seen,
    none matched, enrollment exists — but NOBODY holds the patient
    anywhere. The unmatched person might BE the patient (face failing
    at range is the norm at 640×360), so the whole-tick skip is
    forbidden and analysis must run."""
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
                          identity=identity, main=main,
                          patient_presence=PatientPresence())
    _push_still_frames(ingest)

    await ingest._run_tick_inner(_RecorderDecision())

    assert main.calls, (
        "patient unlocated → the room must keep analyzing"
    )
    row = await _last_analysis(session, patient_id)
    assert row.detector_fields.get("skipped_reason") != "identity_not_patient"


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
