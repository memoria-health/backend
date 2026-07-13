"""Integration: seeded preset detectors carry their v2 dimension values
(detection v2 P1) — the row is born classified, matching catalog_v2 (and
therefore the prod backfill delta generated from it).
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.detector import Detector
from src.services.detection import catalog_v2
from test.integration.test_reid_safety_invariant import _setup_patient_with_camera


@pytest.mark.asyncio
async def test_seeded_presets_carry_v2_dims(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _camera_id = await _setup_patient_with_camera(
        client, auth_user, session,
    )
    rows = (await session.execute(
        select(Detector).where(Detector.patient_id == patient_id)
    )).scalars().all()
    assert rows, "onboarding must seed presets"
    for r in rows:
        dims = catalog_v2.DIMS[r.identifier]
        assert r.severity == dims.severity, r.identifier
        assert r.bucket == dims.bucket, r.identifier
        assert r.evidence_branch == dims.evidence_branch, r.identifier
        assert r.subject_scope == dims.subject_scope, r.identifier
        assert r.unknown_policy == dims.unknown_policy, r.identifier
        assert r.needs_audio == dims.needs_audio, r.identifier
        assert r.evidence_unit == dims.evidence_unit, r.identifier
        assert r.duty_cycle == dims.duty_cycle, r.identifier
        assert r.degradation_warning == dims.degradation_warning, r.identifier
