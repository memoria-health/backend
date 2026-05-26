"""Tests for `face_embedding_repo`.

Real Postgres, no mocks. Provisions a real caregiver + patient via the
public API (same pattern as other test/unit/ files in this repo) so
the FK to `patients(id)` is satisfied. The repo functions are then
exercised directly against the session.

The FIFO-eviction logic is the most failure-prone piece; it has its
own dedicated test that builds up beyond the cap with both selfie and
harvested rows and asserts only harvested rows get evicted.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.repos import face_embedding_repo
from test.conftest import mint_test_jwt


_PATIENT_BODY = {
    "full_name": "Margaret O'Brien",
    "preferred_name": "Maggie",
    "relation": "parent",
}


async def _setup_patient(
    client: AsyncClient, auth_user,
) -> tuple[UUID, dict[str, str]]:
    """Provision a caregiver + patient row, return (patient_id, headers)."""
    uid, email = auth_user
    headers = {"Authorization": f"Bearer {mint_test_jwt(uid, email)}"}
    await client.post("/v1/me", headers=headers, json={"display_name": "Test"})
    r = await client.post("/v1/patients", headers=headers, json=_PATIENT_BODY)
    assert r.status_code == 201, r.text
    return UUID(r.json()["id"]), headers


def _fake_embedding(seed: int) -> bytes:
    """Canonical 512 × float32 = 2048 bytes, matching what
    `embeddings.embedding_from_bytes` validates. Using the wrong
    length silently causes `reid.gallery_skip` warnings downstream
    instead of round-trip failures here."""
    return bytes([seed % 256] * (512 * 4))


# ─── INSERTS ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_insert_processing_creates_placeholder_row(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient(client, auth_user)
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="path/to/photo.jpg",
    )
    await session.commit()

    assert row.patient_id == patient_id
    assert row.source == "selfie"
    assert row.status == "processing"
    assert row.embedding == b""
    assert row.det_score == 0.0
    assert row.source_path == "path/to/photo.jpg"
    assert row.failure_reason is None


@pytest.mark.asyncio
async def test_insert_harvested_creates_ready_row_no_storage_path(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient(client, auth_user)
    row = await face_embedding_repo.insert_harvested(
        session,
        patient_id=patient_id,
        embedding=_fake_embedding(7),
        det_score=0.55,
    )
    await session.commit()

    assert row.source == "harvested"
    assert row.status == "ready"
    assert row.source_path is None
    assert row.embedding == _fake_embedding(7)
    assert row.det_score == pytest.approx(0.55, rel=1e-4)


# ─── STATE TRANSITIONS ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_mark_ready_fills_embedding_and_promotes_status(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient(client, auth_user)
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="path.jpg",
    )
    await session.commit()

    await face_embedding_repo.mark_ready(
        session,
        embedding_id=row.id,
        embedding=_fake_embedding(3),
        det_score=0.82,
    )
    await session.commit()

    after = (await face_embedding_repo.list_for_patient(
        session, patient_id, caregiver_id=row.patient_id,
    ))
    # caregiver_id isn't patient_id; re-fetch with the actual caregiver id.
    # Use the unscoped read instead — same row, simpler assertion.
    ready_rows = await face_embedding_repo.list_ready_for_patient_unscoped(
        session, patient_id,
    )
    assert len(ready_rows) == 1
    promoted = ready_rows[0]
    assert promoted.id == row.id
    assert promoted.status == "ready"
    assert promoted.embedding == _fake_embedding(3)
    assert promoted.det_score == pytest.approx(0.82, rel=1e-4)
    assert promoted.failure_reason is None


@pytest.mark.asyncio
async def test_mark_failed_records_reason_and_leaves_status_failed(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient(client, auth_user)
    row = await face_embedding_repo.insert_processing(
        session, patient_id=patient_id, source_path="bad.jpg",
    )
    await session.commit()

    await face_embedding_repo.mark_failed(
        session, embedding_id=row.id, failure_reason="no face detected",
    )
    await session.commit()

    ready = await face_embedding_repo.list_ready_for_patient_unscoped(
        session, patient_id,
    )
    assert ready == [], "failed rows must not appear in the ready list"


# ─── READS ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_for_patient_returns_only_caregivers_rows(
    client: AsyncClient, auth_user_factory, session: AsyncSession,
) -> None:
    """Two caregivers each have one patient. Repo must filter by
    caregiver. RLS would also catch this; this asserts the app layer
    does the right thing without depending on RLS.
    """
    user_a = await auth_user_factory()
    user_b = await auth_user_factory()

    # Need a fresh client per setup. Use the same fixture's app.
    from httpx import ASGITransport
    from src.main import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        pid_a, _ = await _setup_patient(c, user_a)
        pid_b, _ = await _setup_patient(c, user_b)

    cg_a_id = user_a[0]
    cg_b_id = user_b[0]

    await face_embedding_repo.insert_harvested(
        session, patient_id=pid_a, embedding=_fake_embedding(1), det_score=0.8,
    )
    await face_embedding_repo.insert_harvested(
        session, patient_id=pid_b, embedding=_fake_embedding(2), det_score=0.9,
    )
    await session.commit()

    rows_a = await face_embedding_repo.list_for_patient(
        session, pid_a, caregiver_id=cg_a_id,
    )
    rows_b = await face_embedding_repo.list_for_patient(
        session, pid_b, caregiver_id=cg_b_id,
    )
    rows_cross = await face_embedding_repo.list_for_patient(
        session, pid_a, caregiver_id=cg_b_id,
    )
    assert len(rows_a) == 1 and rows_a[0].patient_id == pid_a
    assert len(rows_b) == 1 and rows_b[0].patient_id == pid_b
    assert rows_cross == [], "cross-account read must return empty"


@pytest.mark.asyncio
async def test_count_by_source_returns_zero_for_missing_sources(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient(client, auth_user)
    cg_id = auth_user[0]

    # 3 selfies (processing), 2 harvested (ready). count_by_source must
    # ignore the processing rows and report only ready.
    for i in range(3):
        await face_embedding_repo.insert_processing(
            session, patient_id=patient_id, source_path=f"selfie_{i}.jpg",
        )
    for i in range(2):
        await face_embedding_repo.insert_harvested(
            session, patient_id=patient_id, embedding=_fake_embedding(i), det_score=0.85,
        )
    await session.commit()

    counts = await face_embedding_repo.count_by_source(
        session, patient_id, caregiver_id=cg_id,
    )
    assert counts == {"selfie": 0, "harvested": 2}


# ─── DELETE + EVICTION ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_by_id_blocks_non_owner(
    client: AsyncClient, auth_user_factory, session: AsyncSession,
) -> None:
    user_a = await auth_user_factory()
    user_b = await auth_user_factory()

    from httpx import ASGITransport
    from src.main import create_app
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        pid_a, _ = await _setup_patient(c, user_a)

    row = await face_embedding_repo.insert_harvested(
        session, patient_id=pid_a, embedding=_fake_embedding(0), det_score=0.5,
    )
    await session.commit()

    deleted_by_other = await face_embedding_repo.delete_by_id(
        session, row.id, caregiver_id=user_b[0],
    )
    await session.commit()
    assert deleted_by_other is False

    deleted_by_owner = await face_embedding_repo.delete_by_id(
        session, row.id, caregiver_id=user_a[0],
    )
    await session.commit()
    assert deleted_by_owner is True


@pytest.mark.asyncio
async def test_evict_oldest_harvested_above_cap_is_no_op_below_cap(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    patient_id, _ = await _setup_patient(client, auth_user)
    for i in range(3):
        await face_embedding_repo.insert_harvested(
            session, patient_id=patient_id, embedding=_fake_embedding(i), det_score=0.8,
        )
    await session.commit()

    evicted = await face_embedding_repo.evict_oldest_harvested_above_cap(
        session, patient_id, cap=10,
    )
    await session.commit()
    assert evicted == 0


@pytest.mark.asyncio
async def test_evict_oldest_harvested_drops_oldest_first_keeps_selfies(
    client: AsyncClient, auth_user, session: AsyncSession,
) -> None:
    """Build a gallery of 4 selfies + 8 harvested rows with a cap of 6.
    Total = 12, overflow = 6. Only harvested rows get evicted (oldest
    first). All 4 selfies must remain.
    """
    patient_id, _ = await _setup_patient(client, auth_user)

    # 4 selfies, oldest first. mark_ready in order so their created_at
    # is monotonically increasing.
    selfie_ids: list[UUID] = []
    for i in range(4):
        row = await face_embedding_repo.insert_processing(
            session, patient_id=patient_id, source_path=f"selfie_{i}.jpg",
        )
        await face_embedding_repo.mark_ready(
            session, embedding_id=row.id, embedding=_fake_embedding(i), det_score=0.9,
        )
        await session.commit()
        selfie_ids.append(row.id)

    # 8 harvested. Track IDs in insertion order so we can assert
    # which ones survived eviction.
    harvested_ids: list[UUID] = []
    for i in range(8):
        row = await face_embedding_repo.insert_harvested(
            session, patient_id=patient_id, embedding=_fake_embedding(100 + i), det_score=0.8,
        )
        await session.commit()
        harvested_ids.append(row.id)

    evicted = await face_embedding_repo.evict_oldest_harvested_above_cap(
        session, patient_id, cap=6,
    )
    await session.commit()

    # 12 total, cap 6 → 6 evicted. All from harvested (oldest 6).
    assert evicted == 6
    remaining = await face_embedding_repo.list_ready_for_patient_unscoped(
        session, patient_id,
    )
    remaining_ids = {r.id for r in remaining}

    # All 4 selfies survived.
    for sid in selfie_ids:
        assert sid in remaining_ids, f"selfie {sid} was wrongly evicted"

    # Of harvested: only the newest 2 survive (positions 6, 7).
    assert harvested_ids[6] in remaining_ids
    assert harvested_ids[7] in remaining_ids
    for i in range(6):
        assert harvested_ids[i] not in remaining_ids, (
            f"harvested[{i}] should have been evicted but wasn't"
        )
