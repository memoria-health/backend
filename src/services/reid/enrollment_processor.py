"""Background pipeline that turns uploaded selfies into ready
embeddings.

The API endpoint writes a row with `status='processing'` and dispatches
a `process_enrollment_photo` task. This module's job:

  1. Download the photo from Supabase Storage
  2. Run SCRFD + ArcFace via `embed_image_bytes`
  3. Either mark_ready (success) or mark_failed (no face / decode
     failure / storage transport error)

Concurrency: each photo is processed in its own task. The repo handles
its own DB session — we do NOT share a session across tasks. This is
the same pattern `detector_compiler.py` uses for the LLM-compile step.

Failures are caught and translated into mark_failed rows so the
caregiver sees "1 of 12 photos couldn't be processed: no face
detected" rather than a half-finished enrollment.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import async_sessionmaker

from src.errors import StorageError
from src.repos import face_embedding_repo
from src.services.reid import embeddings as reid_embeddings
from src.services.reid import storage as reid_storage
from src.settings import Settings

log = logging.getLogger(__name__)


async def process_enrollment_photo(
    *,
    embedding_id: UUID,
    storage_key: str,
    sessionmaker: async_sessionmaker,
    settings: Settings,
) -> None:
    """Process one uploaded selfie end-to-end.

    Caller-fire-and-forget: invoked from `asyncio.create_task` after
    the row is inserted. Failures are caught and persisted as
    `status='failed'` rows so the caregiver-facing status surface can
    report them.
    """
    log.info(
        "reid.enrollment.start embedding=%s key=%s",
        embedding_id, storage_key,
    )
    try:
        image_bytes = await reid_storage.download_face_photo(
            key=storage_key, settings=settings,
        )
    except StorageError as exc:
        await _mark_failed(embedding_id, f"download_failed: {exc}", sessionmaker)
        return

    try:
        hit = await reid_embeddings.embed_image_bytes(
            image_bytes,
            model_name=settings.reid_insightface_model,
        )
    except Exception as exc:
        # ArcFace itself shouldn't raise on bad input (it returns None
        # for "no face"), but bad image bytes can crash cv2's decode
        # or insightface's preprocessor. Translate any failure into a
        # `failed` row rather than crash the worker.
        log.exception(
            "reid.enrollment.embed_crashed embedding=%s err=%s",
            embedding_id, exc,
        )
        await _mark_failed(embedding_id, f"embed_failed: {exc}", sessionmaker)
        return

    if hit is None:
        log.info("reid.enrollment.no_face embedding=%s", embedding_id)
        await _mark_failed(embedding_id, "no_face_detected", sessionmaker)
        return

    # Quality gate at enrollment time. We bias against polluting the
    # selfie set with non-face crops; the dashboard surfaces det_score
    # so caregivers see why a photo was rejected.
    if hit.det_score < settings.reid_enrollment_min_det_score:
        log.info(
            "reid.enrollment.below_threshold embedding=%s det_score=%.3f min=%.3f",
            embedding_id, hit.det_score, settings.reid_enrollment_min_det_score,
        )
        await _mark_failed(
            embedding_id,
            f"low_quality_face: det_score={hit.det_score:.3f}",
            sessionmaker,
        )
        return

    embedding_bytes = reid_embeddings.embedding_to_bytes(hit.embedding)
    async with sessionmaker() as session:
        await face_embedding_repo.mark_ready(
            session,
            embedding_id=embedding_id,
            embedding=embedding_bytes,
            det_score=hit.det_score,
        )
        await session.commit()
    log.info(
        "reid.enrollment.ready embedding=%s det_score=%.3f",
        embedding_id, hit.det_score,
    )


async def _mark_failed(
    embedding_id: UUID,
    reason: str,
    sessionmaker: async_sessionmaker,
) -> None:
    """Single-use commit wrapper around mark_failed so the caller
    doesn't have to deal with session lifecycle in the error paths."""
    async with sessionmaker() as session:
        await face_embedding_repo.mark_failed(
            session, embedding_id=embedding_id, failure_reason=reason,
        )
        await session.commit()
