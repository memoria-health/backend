"""Integration: P5 compiler v2 on the create path — hard rejects
surface as 422 with the plain-language reason, and a completed compile
writes the v2 dims onto the row (born-classified customs).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

import src.api.detectors as detectors_api
from src.db.detector import Detector
from src.services.detector_compiler import CompiledDetector
from src.settings import get_settings
from test.conftest import mint_test_jwt
from test.integration.identity_helpers import _setup_patient_with_camera


def _headers(auth_user) -> dict:
    uid, email = auth_user
    return {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}


@pytest.mark.asyncio
async def test_measured_zero_ask_is_rejected_at_create(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient_with_camera(client, auth_user, session)
    r = await client.post(
        f"/v1/patients/{patient_id}/detectors",
        headers=_headers(auth_user),
        json={
            "name": "Eye twitch watch",
            "description": "eye twitching",
            "category": "safety",
            "custom_prompt": "alert me when her eye twitches",
        },
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["code"] == "detector_rejected"
    assert "testing" in detail["reason"]
    # No row was created.
    rows = (await session.execute(
        select(Detector).where(Detector.patient_id == patient_id,
                               Detector.name == "Eye twitch watch")
    )).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_infer_endpoint_rejects_before_llm_spend(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    r = await client.post(
        "/v1/detectors/infer",
        headers=_headers(auth_user),
        json={"name": "Shiver", "prompt": "tell me when he is shivering"},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "detector_rejected"


@pytest.mark.asyncio
async def test_background_compile_writes_v2_dims(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    patient_id, _ = await _setup_patient_with_camera(client, auth_user, session)

    async def fake_compile(**kw) -> CompiledDetector:
        return CompiledDetector(
            identifier="custom_ab12",
            prompt_fragment="frag",
            schema_json={"custom_ab12_detected": {"type": "boolean"},
                         "custom_ab12_confidence": {"type": "number"},
                         "custom_ab12_reasoning": {"type": "string"}},
        )

    monkeypatch.setattr(detectors_api, "compile_custom_detector", fake_compile)

    # Insert the row the way create does, then run the background task
    # body directly (deterministic — no fire-and-forget race).
    r = await client.post(
        f"/v1/patients/{patient_id}/detectors",
        headers=_headers(auth_user),
        json={
            "name": "TV too long",
            "description": "tv",
            "category": "routine",
            "is_sustained": True,
            "custom_prompt": "watching tv for more than 15 minutes",
        },
    )
    assert r.status_code == 201
    det_id = r.json()["id"]

    await detectors_api._compile_and_update_detector(
        detector_id=det_id, name="TV too long", description="tv",
        user_text="watching tv for more than 15 minutes",
        absence=False, is_sustained=True, category="routine",
        settings=get_settings(),
    )

    row = (await session.execute(
        select(Detector).where(Detector.id == det_id)
    )).scalar_one()
    assert row.identifier == "custom_ab12"
    assert row.evidence_branch == "person_still"
    assert row.subject_scope == "patient"
    assert row.bucket == "person"
    assert row.duty_cycle == "thinned"
    assert row.eventing == {"entry": {"m": 5, "t_seconds": 900}}
    assert row.degradation_warning == "none"
