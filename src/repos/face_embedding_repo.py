"""Repository for `patient_face_embeddings` rows.

Reads + writes are scoped through `patients.caregiver_id` at every
caregiver-facing call site. The pipeline-facing reads
(`list_ready_for_patient_unscoped`) bypass caregiver scoping because
the detector worker has no caregiver session — RLS is the load-bearing
defense there.

All FIFO-eviction logic lives in `evict_oldest_harvested_above_cap`;
the rest is plain selects + inserts + updates. No business logic.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.patient import Patient
from src.db.patient_face_embedding import PatientFaceEmbedding


# ── INSERTS ─────────────────────────────────────────────────────


async def insert_processing(
    session: AsyncSession,
    *,
    patient_id: UUID,
    source_path: str,
) -> PatientFaceEmbedding:
    """Create a `processing` row for an uploaded selfie. The
    enrollment processor fills in the embedding + det_score
    asynchronously via `mark_ready` once SCRFD + ArcFace complete.

    Caller-scoped: the API layer has already validated
    caregiver-ownership of `patient_id` before calling.
    """
    row = PatientFaceEmbedding(
        id=uuid4(),
        patient_id=patient_id,
        embedding=b"",                # placeholder until processed
        det_score=0.0,
        source="selfie",
        status="processing",
        created_at=datetime.now(timezone.utc),
        source_path=source_path,
        failure_reason=None,
    )
    session.add(row)
    await session.flush()
    return row


async def insert_harvested(
    session: AsyncSession,
    *,
    patient_id: UUID,
    embedding: bytes,
    det_score: float,
) -> PatientFaceEmbedding:
    """Direct-insert a harvested embedding (adaptive learning).
    Always status='ready' — harvest runs synchronously inside the tick
    so there's no "processing" state to wait through.
    """
    row = PatientFaceEmbedding(
        id=uuid4(),
        patient_id=patient_id,
        embedding=embedding,
        det_score=det_score,
        source="harvested",
        status="ready",
        created_at=datetime.now(timezone.utc),
        source_path=None,
        failure_reason=None,
    )
    session.add(row)
    await session.flush()
    return row


# ── PROCESSING STATE TRANSITIONS (selfies only) ─────────────────


async def mark_ready(
    session: AsyncSession,
    *,
    embedding_id: UUID,
    embedding: bytes,
    det_score: float,
) -> None:
    """Promote a `processing` selfie row to `ready` after SCRFD +
    ArcFace finish. No caregiver scoping — only the in-process
    enrollment processor calls this, with an ID it just minted.
    """
    await session.execute(
        update(PatientFaceEmbedding)
        .where(PatientFaceEmbedding.id == embedding_id)
        .values(
            embedding=embedding,
            det_score=det_score,
            status="ready",
            failure_reason=None,
        )
    )


async def mark_failed(
    session: AsyncSession,
    *,
    embedding_id: UUID,
    failure_reason: str,
) -> None:
    """Mark a selfie row as failed (e.g. SCRFD found no face)."""
    await session.execute(
        update(PatientFaceEmbedding)
        .where(PatientFaceEmbedding.id == embedding_id)
        .values(status="failed", failure_reason=failure_reason)
    )


# ── READS ───────────────────────────────────────────────────────


async def list_for_patient(
    session: AsyncSession,
    patient_id: UUID,
    *,
    caregiver_id: UUID,
) -> Sequence[PatientFaceEmbedding]:
    """All embeddings (any status, any source) for the caregiver's
    patient. Used by the status endpoint.

    Returns empty if `patient_id` isn't owned by `caregiver_id` — the
    join filters cross-account access.
    """
    stmt = (
        select(PatientFaceEmbedding)
        .join(Patient, Patient.id == PatientFaceEmbedding.patient_id)
        .where(
            PatientFaceEmbedding.patient_id == patient_id,
            Patient.caregiver_id == caregiver_id,
        )
        .order_by(PatientFaceEmbedding.created_at.desc())
    )
    return (await session.execute(stmt)).scalars().all()


async def list_ready_for_patient_unscoped(
    session: AsyncSession,
    patient_id: UUID,
) -> Sequence[PatientFaceEmbedding]:
    """Pipeline path: load the gallery for inference. NO caregiver
    scoping — the detector worker has no caregiver session. RLS is
    the load-bearing defense, and the only place that calls this is
    the Re-ID service which has already resolved `patient_id` from a
    cameras-row → patients-row chain that we trust.
    """
    stmt = (
        select(PatientFaceEmbedding)
        .where(
            PatientFaceEmbedding.patient_id == patient_id,
            PatientFaceEmbedding.status == "ready",
        )
    )
    return (await session.execute(stmt)).scalars().all()


async def count_by_source(
    session: AsyncSession,
    patient_id: UUID,
    *,
    caregiver_id: UUID,
) -> Mapping[str, int]:
    """Return `{'selfie': N, 'harvested': M}` for ready rows only —
    drives the status card on both web and iOS. Includes 0 entries so
    callers don't have to defaultdict.
    """
    stmt = (
        select(PatientFaceEmbedding.source, func.count(PatientFaceEmbedding.id))
        .join(Patient, Patient.id == PatientFaceEmbedding.patient_id)
        .where(
            PatientFaceEmbedding.patient_id == patient_id,
            PatientFaceEmbedding.status == "ready",
            Patient.caregiver_id == caregiver_id,
        )
        .group_by(PatientFaceEmbedding.source)
    )
    rows = (await session.execute(stmt)).all()
    counts = {"selfie": 0, "harvested": 0}
    for source, n in rows:
        counts[source] = int(n)
    return counts


# ── DELETES + EVICTION ──────────────────────────────────────────


async def delete_by_id(
    session: AsyncSession,
    embedding_id: UUID,
    *,
    caregiver_id: UUID,
) -> bool:
    """Delete one row owned by `caregiver_id`. Returns True iff a row
    was actually deleted (404 vs 200 in the API). The subquery makes
    a non-owned ID a no-op rather than an error."""
    owned = (
        select(PatientFaceEmbedding.id)
        .join(Patient, Patient.id == PatientFaceEmbedding.patient_id)
        .where(
            PatientFaceEmbedding.id == embedding_id,
            Patient.caregiver_id == caregiver_id,
        )
        .scalar_subquery()
    )
    result = await session.execute(
        delete(PatientFaceEmbedding).where(PatientFaceEmbedding.id == owned)
    )
    return (result.rowcount or 0) > 0


async def evict_oldest_harvested_above_cap(
    session: AsyncSession,
    patient_id: UUID,
    cap: int,
) -> int:
    """FIFO-evict the oldest 'harvested' rows for `patient_id` until
    the patient has at most `cap` total rows of any source. Selfies
    are never evicted — the eviction set is restricted to source =
    'harvested'.

    Called by the harvester after every successful harvest insert.
    No caregiver scoping — only the pipeline calls this, and the
    patient_id arrives via the trusted cameras → patients chain.

    Returns number of rows actually evicted.
    """
    # Two-step: count total, decide how many to evict, then delete the
    # oldest harvested rows by created_at. A single CTE would also work
    # but two clear statements are easier to follow + test.
    total_stmt = (
        select(func.count(PatientFaceEmbedding.id))
        .where(
            PatientFaceEmbedding.patient_id == patient_id,
            PatientFaceEmbedding.status == "ready",
        )
    )
    total = int((await session.execute(total_stmt)).scalar_one())
    if total <= cap:
        return 0
    overflow = total - cap

    victim_ids = (
        select(PatientFaceEmbedding.id)
        .where(
            PatientFaceEmbedding.patient_id == patient_id,
            PatientFaceEmbedding.source == "harvested",
            PatientFaceEmbedding.status == "ready",
        )
        .order_by(PatientFaceEmbedding.created_at.asc())
        .limit(overflow)
        .scalar_subquery()
    )
    result = await session.execute(
        delete(PatientFaceEmbedding).where(PatientFaceEmbedding.id.in_(victim_ids))
    )
    return int(result.rowcount or 0)
