"""Face-embedding API surface (Re-ID enrollment).

- POST   /v1/patients/{id}/face-embeddings  — multipart upload of N selfies
- GET    /v1/patients/{id}/face-embeddings  — list all rows
- GET    /v1/patients/{id}/face-embeddings/summary — status-card data
- DELETE /v1/face-embeddings/{id}           — remove one embedding

All endpoints require CaregiverAuth and 404 (not 403) for cross-
account attempts, matching the project's existing API conventions.

The POST endpoint returns 202 Accepted: it inserts placeholder rows
with `status='processing'`, dispatches background tasks to extract
embeddings, and lets the client poll the summary endpoint until
processing completes. This decouples the multipart upload latency
(N×storage uploads) from the embedding-extraction latency
(N×InsightFace passes, ~50ms each on CPU).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from sqlalchemy import func, select

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.patient import Patient
from src.db.patient_face_embedding import PatientFaceEmbedding
from src.db.session import get_sessionmaker
from src.errors import StorageError
from src.repos import face_embedding_repo, patient_repo
from src.schemas.face_embedding import (
    FaceEmbeddingResponse,
    FaceEmbeddingSource,
    FaceEmbeddingStatus,
    FaceEnrollmentSummary,
)
from src.services.identity import enrollment_processor
from src.services.identity import storage as reid_storage

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["face-embedding"])


# ─── Helpers ────────────────────────────────────────────────────


def _to_response(row: PatientFaceEmbedding) -> FaceEmbeddingResponse:
    return FaceEmbeddingResponse(
        id=row.id,
        patient_id=row.patient_id,
        source=FaceEmbeddingSource(row.source),
        status=FaceEmbeddingStatus(row.status),
        det_score=row.det_score,
        failure_reason=row.failure_reason,
        created_at=row.created_at,
    )


async def _read_capped_upload(file: UploadFile, *, max_bytes: int) -> bytes:
    """Drain `UploadFile` in 64 KiB chunks, raising 413 the moment we
    exceed `max_bytes`. Same pattern as `api/patients._read_capped_upload`."""
    buf = bytearray()
    chunk_size = 64 * 1024
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"photo too large: > {max_bytes} bytes",
            )
    return bytes(buf)


# ─── POST ───────────────────────────────────────────────────────


@router.post(
    "/patients/{patient_id}/face-embeddings",
    response_model=list[FaceEmbeddingResponse],
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_face_embeddings(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
    photos: list[UploadFile] = File(...),
) -> list[FaceEmbeddingResponse]:
    """Accept 1-N selfies, queue background embedding extraction,
    return placeholder rows. Client polls `/summary` for completion.

    Validates ownership, photo count, content-type, and per-photo
    size. Anything that gets past validation lands as a row in
    `patient_face_embeddings` with `status='processing'`. The
    background processor moves rows to `ready` or `failed`.
    """
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )

    if not photos:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="at least one photo is required",
        )
    if len(photos) > settings.reid_max_photos_per_enrollment:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"too many photos: {len(photos)} "
                f"(max {settings.reid_max_photos_per_enrollment})"
            ),
        )

    # Pre-allocate IDs so we can name the storage objects deterministically
    # by embedding_id before the INSERT lands. This keeps the storage key
    # 1:1 with the DB row even if a parallel transaction interleaves.
    pending: list[tuple[UUID, bytes, str]] = []
    for photo in photos:
        content_type = photo.content_type or ""
        body = await _read_capped_upload(
            photo, max_bytes=settings.storage_max_photo_bytes,
        )
        if not body:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="photo is empty",
            )
        if content_type not in {"image/jpeg", "image/png"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"unsupported content type {content_type!r}",
            )
        pending.append((uuid4(), body, content_type))

    # Release the pooled connection before the sequential Storage uploads
    # below — same fix as camera_ingest.py's 2026-08-15 tick fix (see its
    # comment): holding a connection idle across slow network I/O
    # multiplies pool pressure with concurrent request count. A fresh
    # session is acquired below just for the INSERTs.
    await session.close()

    # Upload all photos to Storage, then insert all rows, then dispatch
    # all background tasks. Upload before INSERT means a Storage failure
    # doesn't leave behind orphaned processing rows the user can't escape.
    uploaded: list[tuple[UUID, str]] = []
    for embedding_id, body, content_type in pending:
        try:
            key = await reid_storage.upload_face_photo(
                patient_id=patient_id,
                embedding_id=embedding_id,
                image_bytes=body,
                content_type=content_type,
                settings=settings,
                region=patient.region,
            )
        except StorageError as exc:
            log.warning(
                "face_embedding.upload_failed patient=%s embedding=%s err=%s",
                patient_id, embedding_id, exc,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"storage upload failed: {exc}",
            ) from exc
        uploaded.append((embedding_id, key))

    rows: list[PatientFaceEmbedding] = []
    sm = get_sessionmaker()
    async with sm() as insert_session:
        for embedding_id, key in uploaded:
            row = await face_embedding_repo.insert_processing(
                insert_session,
                patient_id=patient_id,
                source_path=key,
                embedding_id=embedding_id,  # match the storage key's UUID suffix
            )
            rows.append(row)
        await insert_session.commit()

    # Dispatch background tasks. Each task processes one photo:
    # download -> SCRFD -> ArcFace -> mark_ready / mark_failed.
    for row in rows:
        asyncio.create_task(
            enrollment_processor.process_enrollment_photo(
                embedding_id=row.id,
                storage_key=row.source_path,
                sessionmaker=sm,
                settings=settings,
            ),
            name=f"reid-enroll-{row.id}",
        )

    return [_to_response(r) for r in rows]


# ─── GET ────────────────────────────────────────────────────────


@router.get(
    "/patients/{patient_id}/face-embeddings",
    response_model=list[FaceEmbeddingResponse],
)
async def list_face_embeddings(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[FaceEmbeddingResponse]:
    """List all embeddings for the caregiver's patient (any status,
    any source). Empty list if patient doesn't exist or isn't owned."""
    rows = await face_embedding_repo.list_for_patient(
        session, patient_id, caregiver_id=auth.id,
    )
    return [_to_response(r) for r in rows]


@router.get(
    "/patients/{patient_id}/face-embeddings/summary",
    response_model=FaceEnrollmentSummary,
)
async def get_face_enrollment_summary(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> FaceEnrollmentSummary:
    """Status-card data: counts by source + status + most recent
    update. Drives the byte-identical status string the iOS card and
    web card both render.

    Returns the empty-gallery summary (all zeros, last_update=None) if
    the patient doesn't exist or isn't owned. We do NOT 404 on
    not-owned because the status card on iOS / web treats "no rows"
    and "wrong patient" the same way ("Not enrolled").
    """
    # All-in-one query: counts by (source, status) + max(created_at)
    # over ready rows. Aggregating in SQL keeps this one round-trip.
    stmt = (
        select(
            PatientFaceEmbedding.source,
            PatientFaceEmbedding.status,
            func.count(PatientFaceEmbedding.id),
            func.max(PatientFaceEmbedding.created_at),
        )
        .where(PatientFaceEmbedding.patient_id == patient_id)
        .group_by(PatientFaceEmbedding.source, PatientFaceEmbedding.status)
    )
    # Caregiver scoping: an unowned patient returns no rows from the
    # join, which collapses to "Not enrolled" on the client.
    is_owned = await patient_repo.find_by_id(
        session, patient_id, caregiver_id=auth.id,
    )
    if is_owned is None:
        return FaceEnrollmentSummary(
            selfie_count=0, harvested_count=0, processing_count=0,
            failed_count=0, last_update=None,
        )

    rows = (await session.execute(stmt)).all()
    selfie_count = 0
    harvested_count = 0
    processing_count = 0
    failed_count = 0
    last_update: datetime | None = None
    for src, st, n, max_ts in rows:
        n = int(n)
        if st == "ready":
            if src == "selfie":
                selfie_count += n
            elif src == "harvested":
                harvested_count += n
            if max_ts is not None and (last_update is None or max_ts > last_update):
                last_update = max_ts
        elif st == "processing":
            processing_count += n
        elif st == "failed":
            failed_count += n
    return FaceEnrollmentSummary(
        selfie_count=selfie_count,
        harvested_count=harvested_count,
        processing_count=processing_count,
        failed_count=failed_count,
        last_update=last_update,
    )


# ─── DELETE ─────────────────────────────────────────────────────


@router.delete(
    "/face-embeddings/{embedding_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_face_embedding(
    embedding_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> None:
    """Remove one embedding row + its storage object (selfie only).
    404 for not-found OR not-owned (no distinction by design)."""
    # Single ownership-scoped read so cross-account rows are never
    # materialized. The JOIN through patients.caregiver_id is the
    # same guard `face_embedding_repo.delete_by_id` uses; doing it
    # here too lets us grab `source_path` before the delete without
    # leaking unowned rows into local state.
    stmt = (
        select(PatientFaceEmbedding)
        .join(Patient, Patient.id == PatientFaceEmbedding.patient_id)
        .where(
            PatientFaceEmbedding.id == embedding_id,
            Patient.caregiver_id == auth.id,
        )
    )
    row = (await session.execute(stmt)).scalar_one_or_none()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="embedding_not_found",
        )

    deleted = await face_embedding_repo.delete_by_id(
        session, embedding_id, caregiver_id=auth.id,
    )
    if not deleted:
        # Shouldn't be reachable given the ownership-scoped read
        # above, but defend in depth: race against another caregiver
        # session deleting between read and delete.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="embedding_not_found",
        )
    await session.commit()

    # Best-effort storage cleanup. A Storage failure here doesn't
    # un-delete the row; the orphaned object is fine (we won't read it
    # again) and a future janitor can sweep.
    if row.source_path:
        try:
            await reid_storage.delete_face_photo(
                key=row.source_path, settings=settings,
            )
        except StorageError as exc:
            log.warning(
                "face_embedding.delete_storage_failed key=%s err=%s",
                row.source_path, exc,
            )
