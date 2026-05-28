"""Adaptive learning: append confidently-matched face crops to the
patient gallery as `source='harvested'` rows.

Why: enrollment selfies are out-of-domain for wall-cam frames. The
match scores oscillate around 0.40-0.60 instead of the 0.70+ we get
on selfies. Harvesting in-domain face crops (wall-cam, real
conditions) closes the domain gap over a week of operation.

Why no det_score gate at harvest time: the whole point of adaptive
learning is to capture LOW-quality faces so we can match them in the
future. The 0.80 cosine threshold (vs the 0.40 lock threshold) is the
only gate against mis-attribution.

Called from `ReIDInference._label_clip_inner` AFTER a high-confidence
match. The new embedding becomes visible to the next tick once the
gallery cache TTL expires (or immediately via `invalidate_gallery`).

FIFO eviction keeps the gallery bounded. Selfies are never evicted.
"""

from __future__ import annotations

import logging
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.repos import face_embedding_repo
from src.services.reid import embeddings as reid_embeddings
from src.settings import Settings

log = logging.getLogger(__name__)


async def maybe_harvest_embedding(
    *,
    patient_id: UUID,
    embedding: np.ndarray,
    det_score: float,
    match_score: float,
    sessionmaker: async_sessionmaker,
    settings: Settings,
) -> bool:
    """Insert a harvested embedding if conditions are met.

    Returns True iff a row was actually written (caller can then
    invalidate the gallery cache to make the new row visible on the
    very next tick instead of waiting for TTL).

    Conditions:
      - `reid_harvest_enabled` setting is True
      - `match_score` >= `reid_harvest_threshold` (default 0.80)
      - `det_score > 0` as a sanity check (no zero-confidence faces)

    Side effect: after a successful insert, calls
    `evict_oldest_harvested_above_cap` so the gallery stays under
    `reid_gallery_cap`. The eviction is harvested-only (selfies
    untouched).
    """
    if not settings.reid_harvest_enabled:
        return False
    if match_score < settings.reid_harvest_threshold:
        return False
    if det_score <= 0:
        return False

    embedding_bytes = reid_embeddings.embedding_to_bytes(embedding)
    async with sessionmaker() as session:
        await face_embedding_repo.insert_harvested(
            session,
            patient_id=patient_id,
            embedding=embedding_bytes,
            det_score=det_score,
        )
        evicted = await face_embedding_repo.evict_oldest_harvested_above_cap(
            session, patient_id, cap=settings.reid_gallery_cap,
        )
        await session.commit()

    log.info(
        "reid.harvest patient=%s match_score=%.3f det_score=%.3f evicted=%d",
        patient_id, match_score, det_score, evicted,
    )
    return True
