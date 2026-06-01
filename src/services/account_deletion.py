"""Full caregiver-account deletion — the worker behind DELETE /v1/me.

The FK cascade (`caregivers.id → auth.users(id) ON DELETE CASCADE`, then
patients → cameras/detectors/events/analyses/face_embeddings and
caregiver_devices off the caregiver) erases every DB row the moment the
auth user is deleted. Two things are NOT cascade-covered and must be
handled here:

  1. Supabase Storage objects (patient avatars, event clips, Re-ID
     selfies) — no FK, so we collect their keys BEFORE deleting any rows
     and remove them best-effort.
  2. In-process detector workers holding the caregiver's cameras — we
     ask them to stop best-effort; the RoomWorker reconciler reaps any
     stragglers within ~30s regardless.

Order matters: collect keys → delete storage → stop workers → delete the
auth user (the cascade) → belt-and-braces caregiver-row delete. Required
for App Store Review Guideline 5.1.1(v).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver
from src.db.camera import Camera
from src.db.event import Event
from src.db.patient import Patient
from src.db.patient_face_embedding import PatientFaceEmbedding
from src.errors import StorageError
from src.services import auth_admin, storage
from src.settings import Settings

log = logging.getLogger(__name__)


async def delete_caregiver_account(
    caregiver_id: UUID,
    *,
    session: AsyncSession,
    settings: Settings,
    worker: Any | None = None,
) -> None:
    """Delete the caregiver and ALL associated data. Idempotent-ish:
    a re-run after a partial failure re-deletes whatever remains.

    `worker` is `app.state.detector_worker` (or None in tests); we only
    call its `stop_worker_for_camera`.
    """
    # 1. Collect storage keys + camera ids BEFORE anything is deleted —
    #    the cascade in step 4 erases the rows we'd otherwise read from.
    selfie_keys = list(
        (
            await session.execute(
                select(PatientFaceEmbedding.source_path)
                .join(Patient, Patient.id == PatientFaceEmbedding.patient_id)
                .where(Patient.caregiver_id == caregiver_id)
                .where(PatientFaceEmbedding.source_path.is_not(None))
            )
        ).scalars().all()
    )
    clip_keys = list(
        (
            await session.execute(
                select(Event.clip_storage_key)
                .join(Patient, Patient.id == Event.patient_id)
                .where(Patient.caregiver_id == caregiver_id)
                .where(Event.clip_storage_key.is_not(None))
            )
        ).scalars().all()
    )
    avatar_keys = list(
        (
            await session.execute(
                select(Patient.photo_storage_key)
                .where(Patient.caregiver_id == caregiver_id)
                .where(Patient.photo_storage_key.is_not(None))
            )
        ).scalars().all()
    )
    camera_ids = list(
        (
            await session.execute(
                select(Camera.id)
                .join(Patient, Patient.id == Camera.patient_id)
                .where(Patient.caregiver_id == caregiver_id)
            )
        ).scalars().all()
    )

    # 2. Delete storage objects. Best-effort: a StorageError is logged,
    #    not fatal — an orphaned object is harmless (never read again) and
    #    a future janitor can sweep. Selfies + avatars share the
    #    patient-photos bucket; clips live in incident-clips.
    for key in (*selfie_keys, *avatar_keys):
        await _safe_delete(settings.storage_bucket_patient_photos, key, settings)
    for key in clip_keys:
        await _safe_delete(settings.storage_bucket_incident_clips, key, settings)

    # 3. Stop in-process detector workers for the caregiver's cameras.
    #    Best-effort and bounded — the reconciler is the real safety net,
    #    so a stop that times out or errors must not block the delete.
    if worker is not None:
        for cid in camera_ids:
            try:
                await asyncio.wait_for(
                    worker.stop_worker_for_camera(cid, timeout=4.0),
                    timeout=5.0,
                )
            except Exception:
                log.warning(
                    "account_deletion.worker_stop_failed camera=%s — "
                    "reconciler will reap within ~30s",
                    cid,
                )

    # 4. Delete the Supabase auth user → FK cascade erases the whole DB
    #    graph. Raises AuthAdminError on real failure (mapped to 5xx by
    #    the route) so we never report success on a half-deleted account.
    await auth_admin.delete_auth_user(caregiver_id, settings=settings)

    # 5. Belt-and-braces: ensure the caregiver row is gone. Normally a
    #    no-op (step 4's cascade already removed it via a separate
    #    connection); kept so the endpoint stays correct even if the
    #    auth→caregivers FK is ever reconfigured.
    await session.execute(
        Caregiver.__table__.delete().where(Caregiver.id == caregiver_id)
    )
    await session.commit()


async def _safe_delete(bucket: str, key: str, settings: Settings) -> None:
    try:
        await storage.delete_object(bucket=bucket, key=key, settings=settings)
    except StorageError as exc:
        log.warning(
            "account_deletion.storage_delete_failed bucket=%s key=%s err=%s",
            bucket, key, exc,
        )
