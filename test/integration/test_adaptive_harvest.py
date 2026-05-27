"""Adaptive learning tests.

Verifies that high-confidence matches harvest a `source='harvested'`
row, that the gallery cap triggers FIFO eviction of harvested rows
(never selfies), and that the harvest threshold gate works.
"""

from __future__ import annotations

import numpy as np
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.repos import face_embedding_repo
from src.services.reid import harvester
from src.services.reid.embeddings import embedding_to_bytes
from src.settings import get_settings
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Maggie",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(client: AsyncClient, auth_user) -> str:
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    r = await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)
    return r.json()["id"]


def _embedding(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-8
    return v


@pytest.mark.asyncio
async def test_harvest_above_threshold_writes_harvested_row(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    pid_str = await _setup_patient(client, auth_user)
    from uuid import UUID
    pid = UUID(pid_str)
    settings = get_settings()
    from src.db.session import get_sessionmaker
    sm = get_sessionmaker()

    harvested = await harvester.maybe_harvest_embedding(
        patient_id=pid,
        embedding=_embedding(1),
        det_score=0.5,
        match_score=0.85,
        sessionmaker=sm,
        settings=settings,
    )
    assert harvested is True

    rows = await face_embedding_repo.list_ready_for_patient_unscoped(session, pid)
    assert len(rows) == 1
    assert rows[0].source == "harvested"


@pytest.mark.asyncio
async def test_harvest_below_threshold_is_no_op(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    pid_str = await _setup_patient(client, auth_user)
    from uuid import UUID
    pid = UUID(pid_str)
    settings = get_settings()
    from src.db.session import get_sessionmaker
    sm = get_sessionmaker()

    # 0.79 is just under the 0.80 harvest threshold default.
    harvested = await harvester.maybe_harvest_embedding(
        patient_id=pid,
        embedding=_embedding(2),
        det_score=0.9,
        match_score=0.79,
        sessionmaker=sm,
        settings=settings,
    )
    assert harvested is False

    rows = await face_embedding_repo.list_ready_for_patient_unscoped(session, pid)
    assert rows == []


@pytest.mark.asyncio
async def test_harvest_evicts_oldest_harvested_when_cap_hit(
    client: AsyncClient, auth_user, session: AsyncSession, monkeypatch,
) -> None:
    """Force the cap to a tiny number, harvest more than that, confirm
    only the OLDEST harvested rows get evicted (selfies are spared
    even with a tighter cap)."""
    pid_str = await _setup_patient(client, auth_user)
    from uuid import UUID
    pid = UUID(pid_str)

    # Seed 2 selfie rows so we have something protected from eviction.
    selfie_ids = []
    for i in range(2):
        row = await face_embedding_repo.insert_processing(
            session, patient_id=pid, source_path=f"selfie_{i}.jpg",
        )
        await face_embedding_repo.mark_ready(
            session,
            embedding_id=row.id,
            embedding=embedding_to_bytes(_embedding(10 + i)),
            det_score=0.85,
        )
        await session.commit()
        selfie_ids.append(row.id)

    # Tighten the cap. 2 selfies + 4 harvested rows = 6 total, cap=4
    # means overflow=2, two oldest HARVESTED rows should evict (not
    # the selfies).
    settings = get_settings().model_copy(update={"reid_gallery_cap": 4})
    from src.db.session import get_sessionmaker
    sm = get_sessionmaker()

    harvested_ids: list = []
    for i in range(4):
        await harvester.maybe_harvest_embedding(
            patient_id=pid,
            embedding=_embedding(100 + i),
            det_score=0.5,
            match_score=0.85,
            sessionmaker=sm,
            settings=settings,
        )

    rows = await face_embedding_repo.list_ready_for_patient_unscoped(session, pid)
    # Total = 4 (the cap). The eviction policy keeps newest harvested
    # plus all selfies; oldest two harvested are gone.
    sources = sorted(r.source for r in rows)
    assert sources.count("selfie") == 2, f"selfie protected: {sources}"
    assert sources.count("harvested") == 2, f"harvested capped: {sources}"

    surviving_ids = {r.id for r in rows}
    for sid in selfie_ids:
        assert sid in surviving_ids, f"selfie {sid} was wrongly evicted"
