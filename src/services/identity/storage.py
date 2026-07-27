"""Supabase Storage helpers for Re-ID face photos.

Selfies live under the existing patient-photos bucket at
``patient-faces/{patient_id}/{embedding_id}.jpg``. Reuses the
primitives in `src/services/storage.py` so we don't duplicate the
auth + retry + size-check logic.

Harvested rows do NOT persist a photo to storage (only the embedding
goes into Postgres). The functions here are selfie-only.
"""

from __future__ import annotations

from uuid import UUID

from src.errors import StorageError
from src.services import region as region_routing
from src.services import storage as base_storage
from src.settings import Settings

# Sub-key under patient-photos. Distinct from {patient_id}.jpg (avatar)
# so the avatar and face gallery don't collide.
_FACE_PREFIX = "patient-faces"


def face_storage_key(patient_id: UUID, embedding_id: UUID, extension: str) -> str:
    """Canonical bucket-relative key for an enrollment photo."""
    return f"{_FACE_PREFIX}/{patient_id}/{embedding_id}.{extension}"


async def upload_face_photo(
    *,
    patient_id: UUID,
    embedding_id: UUID,
    image_bytes: bytes,
    content_type: str,
    settings: Settings,
    region: str | None = None,
) -> str:
    """Upload one enrollment photo, returning the bucket-relative key.

    Reuses `base_storage._validate_image_payload` indirectly via the
    upstream pattern: validation is done at the API layer (matching
    upload_patient_photo's behavior), so this just dispatches the
    upload. We don't expose `_validate_image_payload` publicly to keep
    the storage module's surface tight.

    India patients' face photos are region-prefixed (``IN/patient-faces/...``)
    so this biometric-adjacent media is isolated + liftable; US unchanged. The
    returned key is persisted on the embedding row (`source_path`) and reused
    for download/delete, so reads need no region knowledge.
    """
    extension = _ext_for_content_type(content_type)
    key = region_routing.storage_key(
        region, face_storage_key(patient_id, embedding_id, extension),
    )
    # Hand off to the shared primitive so the auth/retry/timeout logic
    # stays in one place. The face-photo cap matches the avatar cap;
    # iOS downsamples to <1MB before upload.
    if len(image_bytes) == 0:
        raise StorageError("image_bytes is empty")
    if len(image_bytes) > settings.storage_max_photo_bytes:
        raise StorageError(
            f"image_bytes too large: {len(image_bytes)} bytes "
            f"(max {settings.storage_max_photo_bytes})"
        )
    await base_storage.upload_object(
        bucket=settings.storage_bucket_patient_photos,
        key=key,
        body=image_bytes,
        content_type=content_type,
        settings=settings,
    )
    return key


async def download_face_photo(
    *,
    key: str,
    settings: Settings,
) -> bytes:
    """Pull a previously-uploaded face photo back for embedding
    extraction. The enrollment processor downloads each photo, runs
    SCRFD + ArcFace, then updates the row.
    """
    return await base_storage.download_object(
        bucket=settings.storage_bucket_patient_photos,
        key=key,
        settings=settings,
    )


async def delete_face_photo(
    *,
    key: str,
    settings: Settings,
) -> None:
    """Remove a face photo when its embedding row is deleted."""
    await base_storage.delete_object(
        bucket=settings.storage_bucket_patient_photos,
        key=key,
        settings=settings,
    )


_ALLOWED: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
}


def _ext_for_content_type(content_type: str) -> str:
    """Map the multipart content-type to a canonical file extension.

    Raises StorageError on unsupported types — matches the avatar
    upload path's error vocabulary so the API layer maps it the same
    way (400 Bad Request).
    """
    ext = _ALLOWED.get(content_type)
    if ext is None:
        raise StorageError(
            f"unsupported content type {content_type!r}; "
            f"expected one of {sorted(_ALLOWED)}"
        )
    return ext
