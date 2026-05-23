"""Patient API surface — `/v1/patients/*` endpoints.

- GET    /v1/patients         — list the caregiver's patients.
- POST   /v1/patients         — create. Auto-seeds preset detectors.
- GET    /v1/patients/{id}    — fetch one.
- PATCH  /v1/patients/{id}    — partial update.
- DELETE /v1/patients/{id}    — remove (cascades to children).

All endpoints require CaregiverAuth. 404 (not 403) for cross-account
attempts so a probing client can't discover patient IDs.
"""

import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.patient import Patient
from src.errors import StorageError
from src.repos import patient_repo
from src.schemas.patient import (
    CreatePatientRequest,
    PatientResponse,
    PhotoURLResponse,
    UpdatePatientRequest,
)
from src.services import onboarding, storage

router = APIRouter(prefix="/v1/patients", tags=["patient"])


def _to_response(p: Patient) -> PatientResponse:
    return PatientResponse.model_validate(p, from_attributes=True)


_UPLOAD_CHUNK_BYTES = 64 * 1024


async def _read_capped_upload(file: UploadFile, *, max_bytes: int) -> bytes:
    """Drain an `UploadFile` in 64 kB chunks, raising 413 the moment
    the running total crosses `max_bytes`. Prevents `await
    file.read()` from materializing an arbitrarily large multipart
    body in RAM before the size check runs."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"image_bytes too large: > {max_bytes} bytes (max {max_bytes})",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.get("", response_model=list[PatientResponse])
async def list_patients(
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[PatientResponse]:
    rows = await patient_repo.list_for_caregiver(session, auth.id)
    return [_to_response(p) for p in rows]


@router.post(
    "",
    response_model=PatientResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_patient(
    auth: CaregiverAuth,
    body: CreatePatientRequest,
    session: SessionDep,
) -> PatientResponse:
    """Create a patient under the current caregiver.

    Auto-seeds the four preset detectors (Falls / Agitation / Wandering
    / Medication adherence) via `services.onboarding.seed_detectors_for_patient`
    in the same transaction. Lands in step 7 of the build sequence;
    until then, new patients have an empty detector list.
    """
    now = datetime.now(timezone.utc)
    p = Patient(
        id=uuid4(),
        caregiver_id=auth.id,
        full_name=body.full_name,
        preferred_name=body.preferred_name,
        relation=body.relation.value,
        state="calm",
        status_line="All quiet",
        cameras_paused=False,
        created_at=now,
        updated_at=now,
        date_of_birth=body.date_of_birth,
        gender=body.gender,
        home_address=body.home_address,
        last_event_at=None,
        battery_percent=None,
    )
    await patient_repo.insert(session, p)
    await onboarding.seed_detectors_for_patient(session, p.id)
    await session.commit()
    return _to_response(p)


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> PatientResponse:
    p = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )
    return _to_response(p)


@router.patch("/{patient_id}", response_model=PatientResponse)
async def patch_patient(
    patient_id: UUID,
    auth: CaregiverAuth,
    body: UpdatePatientRequest,
    session: SessionDep,
) -> PatientResponse:
    """Partial update.

    The Pydantic model only exposes caregiver-writable fields, so even
    a payload that tries to set `state` / `status_line` / `last_event_at`
    / `battery_percent` is rejected as 422 by `extra="forbid"`. Those
    columns transition via wall-phone telemetry (not in scope here).
    """
    fields = body.model_dump(exclude_unset=True)
    # Enum → string for the ORM string columns.
    if "relation" in fields and fields["relation"] is not None:
        fields["relation"] = fields["relation"].value
    p = await patient_repo.update_partial(
        session, patient_id, caregiver_id=auth.id, fields=fields
    )
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )
    await session.commit()
    return _to_response(p)


@router.delete(
    "/{patient_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None
)
async def delete_patient(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> None:
    """Remove a caregiver-owned patient. Cascades to cameras /
    detectors / events via DB-level `ON DELETE CASCADE`.
    """
    deleted = await patient_repo.delete_by_id(session, patient_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )
    await session.commit()


# ── Patient avatar (POST/GET/DELETE /v1/patients/{id}/photo) ────────


@router.post(
    "/{patient_id}/photo",
    response_model=PatientResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_patient_photo(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
    photo: UploadFile = File(...),
) -> PatientResponse:
    """Upload (or replace) the patient's avatar.

    Multipart with one part: `photo` (JPEG/PNG, ≤ storage_max_photo_bytes
    enforced server-side — iOS down-samples before upload). The same
    storage key is reused across re-uploads (one photo per patient),
    so this idempotently overwrites.
    """
    p = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )

    content_type = photo.content_type or ""
    # Stream the multipart body in chunks so a malicious 500 MB upload
    # can't OOM the backend before we get a chance to reject it. The
    # storage layer's size check is the second line of defense; this
    # one prevents the buffer from existing in the first place.
    body = await _read_capped_upload(photo, max_bytes=settings.storage_max_photo_bytes)
    try:
        key = await storage.upload_patient_photo(
            patient_id=patient_id,
            image_bytes=body,
            content_type=content_type,
            settings=settings,
        )
    except StorageError as exc:
        msg = str(exc)
        if "too large" in msg:
            raise HTTPException(
                # 413 — name changed in Starlette; both constants resolve
                # to 413 so the wire code is identical either way.
                status_code=413,
                detail=msg,
            ) from exc
        if "unsupported content type" in msg or "empty" in msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=msg,
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=msg,
        ) from exc

    p = await patient_repo.update_partial(
        session, patient_id,
        caregiver_id=auth.id,
        fields={"photo_storage_key": key},
    )
    if p is None:
        # Patient deleted mid-flight — the upload is now orphaned but
        # that's tolerable (key is keyed by patient_id; new patient
        # with same UUID can't happen).
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )
    await session.commit()
    return _to_response(p)


@router.get("/{patient_id}/photo-url", response_model=PhotoURLResponse)
async def get_patient_photo_url(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> PhotoURLResponse:
    """Mint a short-lived signed URL for the patient's avatar. 404 if
    the patient doesn't exist or isn't owned; 404 (not 409) if the
    avatar hasn't been uploaded yet — the caregiver app falls back to
    initials on any non-200."""
    p = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if p is None or p.photo_storage_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="photo_not_found",
        )
    url = await storage.signed_url(
        bucket=settings.storage_bucket_patient_photos,
        key=p.photo_storage_key,
        settings=settings,
    )
    return PhotoURLResponse(
        url=url,
        expires_in_seconds=settings.storage_signed_url_default_ttl_seconds,
    )


@router.delete(
    "/{patient_id}/photo",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_patient_photo(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> None:
    """Remove the avatar. Clears `photo_storage_key` and deletes the
    underlying Supabase Storage object. Idempotent — 204 even when
    there was no prior photo."""
    p = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if p is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )
    if p.photo_storage_key is None:
        return
    prior_key = p.photo_storage_key
    await patient_repo.update_partial(
        session, patient_id, caregiver_id=auth.id, fields={"photo_storage_key": None},
    )
    await session.commit()
    # Storage delete after DB commit — if it fails, the row already
    # forgot the key; the orphaned object can be cleaned up out-of-band.
    try:
        await storage.delete_object(
            bucket=settings.storage_bucket_patient_photos,
            key=prior_key,
            settings=settings,
        )
    except StorageError as exc:
        # Don't fail the API call; the caller's intent is satisfied
        # (DB row says no photo).
        logging.getLogger(__name__).warning(
            "patient_photo.delete_object_failed key=%s err=%s", prior_key, exc,
        )
