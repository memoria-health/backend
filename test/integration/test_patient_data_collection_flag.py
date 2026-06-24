"""Task 1 — per-patient data_collection flag defaults at the DB layer.

A freshly-created patient must default to data collection OFF with an
unset consent timestamp, so nothing records and the consent banner
shows until the caregiver explicitly decides. This exercises the ORM
mapping + DB column defaults directly; the API exposure of these
fields is covered separately in ``test_patient_data_collection_api``.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from src.db.patient import Patient
from src.db.session import get_sessionmaker
from test.integration.test_patients import _FULL_BODY, _provision


@pytest.mark.asyncio
async def test_new_patient_defaults_data_collection_off_and_undecided(
    client: AsyncClient, auth_user
) -> None:
    headers = await _provision(client, auth_user)
    r = await client.post("/v1/patients", headers=headers, json=_FULL_BODY)
    assert r.status_code == 201
    patient_id = UUID(r.json()["id"])

    sm = get_sessionmaker()
    async with sm() as s:
        row = (
            await s.execute(select(Patient).where(Patient.id == patient_id))
        ).scalar_one()

    # Default OFF: nothing records until explicit opt-in.
    assert row.data_collection_enabled is False
    # Never asked yet: NULL drives the first-open consent banner.
    assert row.data_collection_decided_at is None
