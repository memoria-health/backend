"""Supabase Storage uploads for patient photos and incident clips.

Lifted from wolf's `src/services/storage.py`, with two adaptations:

  - Both buckets here are **private**. Uploads return the bucket-
    relative storage key only; the caregiver app reaches the bytes by
    asking the backend for a short-lived signed URL via
    `signed_url(...)`. The plan flagged this as the
    "store-key-not-URL" pattern wolf converged on.
  - We use `logging` (project convention) instead of `logfire`.

Two buckets, one per concern:

  - ``patient-photos`` — one avatar per patient. Object key
    ``{patient_id}.{ext}``.
  - ``incident-clips`` — one MP4 per fired event. Object key
    ``events/{event_id}.mp4``.

Uploads use the service-role key in the Authorization header so they
bypass RLS; the caregiver JWT is checked at the API layer above. The
``x-upsert: true`` header lets a re-upload overwrite the same key
without a delete round-trip — important for the clip-archival concat
step that writes pre.mp4, post.mp4, then the final combined MP4.

Pure async functions; no module-level state, no clients held across
calls. Callers inject the frozen `Settings` instance.
"""

from __future__ import annotations

import logging
from uuid import UUID

import httpx

from src.errors import StorageError
from src.services import region as region_routing
from src.settings import Settings

log = logging.getLogger(__name__)


# Whitelist of accepted image MIME types and their canonical file
# extensions. We map content-type to extension so the object key
# reflects the actual encoding rather than guessing from filenames
# the client supplied.
_ALLOWED_IMAGE_CONTENT_TYPES: dict[str, str] = {
    "image/jpeg": "jpg",
    "image/png": "png",
}


def _validate_image_payload(
    image_bytes: bytes,
    content_type: str,
    max_bytes: int,
) -> str:
    """Validate an image upload and return the canonical extension.

    Raises:
        StorageError: payload empty, oversized, or wrong content type.
    """
    if content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
        raise StorageError(
            f"unsupported content type {content_type!r}; "
            f"expected one of {sorted(_ALLOWED_IMAGE_CONTENT_TYPES)}"
        )
    if len(image_bytes) == 0:
        raise StorageError("image_bytes is empty")
    if len(image_bytes) > max_bytes:
        raise StorageError(
            f"image_bytes too large: {len(image_bytes)} bytes (max {max_bytes})"
        )
    return _ALLOWED_IMAGE_CONTENT_TYPES[content_type]


async def upload_object(
    *,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
    settings: Settings,
    timeout_seconds: float | None = None,
) -> None:
    """Upload bytes to Supabase Storage. Returns nothing — callers
    that need a URL use `signed_url(...)` separately.

    `timeout_seconds` overrides the default photo-upload timeout for
    larger payloads (e.g. incident clips ~20 MB). Falls back to
    `settings.storage_upload_timeout_seconds` when None.

    Raises:
        StorageError: Supabase responded non-2xx or HTTP transport failed.
    """
    base_url = settings.supabase_url.rstrip("/")
    upload_url = f"{base_url}/storage/v1/object/{bucket}/{key}"

    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.storage_upload_timeout_seconds
    )

    log.info(
        "storage.upload.start bucket=%s key=%s content_type=%s size=%d",
        bucket, key, content_type, len(body),
    )

    try:
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            response = await client.post(upload_url, content=body, headers=headers)
    except httpx.TimeoutException as exc:
        log.warning("storage.upload.timeout bucket=%s key=%s err=%s", bucket, key, exc)
        raise StorageError(
            f"upload failed: timeout after {effective_timeout}s: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        log.warning("storage.upload.transport_error bucket=%s key=%s err=%s", bucket, key, exc)
        raise StorageError(f"upload failed: transport error: {exc}") from exc

    if response.status_code < 200 or response.status_code >= 300:
        log.warning(
            "storage.upload.provider_error bucket=%s key=%s status=%d",
            bucket, key, response.status_code,
        )
        raise StorageError(
            f"upload failed: {response.status_code} {response.reason_phrase}: "
            f"{response.text}"
        )

    log.info(
        "storage.upload.success bucket=%s key=%s status=%d",
        bucket, key, response.status_code,
    )


async def upload_patient_photo(
    *,
    patient_id: UUID,
    image_bytes: bytes,
    content_type: str,
    settings: Settings,
    region: str | None = None,
) -> str:
    """Upload a patient avatar and return its bucket-relative storage key.

    The key is ``{patient_id}.{ext}`` in the patient-photos bucket;
    re-uploading replaces the existing avatar (one photo per patient). For
    India patients the key is region-prefixed (``IN/{patient_id}.{ext}``) so
    the object is isolated + liftable; US is unchanged. The returned
    (prefixed) key is stored on the patient row and reused verbatim for
    signed-URL / delete, so reads need no region knowledge.

    Raises:
        StorageError: payload validation failed or storage backend rejected.
    """
    extension = _validate_image_payload(
        image_bytes, content_type, settings.storage_max_photo_bytes,
    )
    key = region_routing.storage_key(region, f"{patient_id}.{extension}")
    await upload_object(
        bucket=settings.storage_bucket_patient_photos,
        key=key,
        body=image_bytes,
        content_type=content_type,
        settings=settings,
    )
    return key


async def upload_clip(
    *,
    event_id: UUID,
    mp4_bytes: bytes,
    settings: Settings,
    sub_key: str | None = None,
    region: str | None = None,
) -> str:
    """Upload a clip MP4 and return its bucket-relative storage key.

    Default key is ``events/{event_id}.mp4`` — the final combined
    pre+post clip the caregiver app plays. `sub_key`, when provided,
    overrides the suffix for intermediate uploads
    (``events/{event_id}/pre.mp4`` / ``post.mp4``) during the concat
    pass; the concat result then re-uploads to the default key and
    the intermediates are deleted.

    Validates size via `settings.storage_max_clip_bytes`. Content-type
    is fixed at ``video/mp4`` — the only MIME the bucket accepts.

    Raises:
        StorageError: payload empty/oversized or storage backend rejected.
    """
    if len(mp4_bytes) == 0:
        raise StorageError("mp4_bytes is empty")
    if len(mp4_bytes) > settings.storage_max_clip_bytes:
        raise StorageError(
            f"mp4_bytes too large: {len(mp4_bytes)} bytes "
            f"(max {settings.storage_max_clip_bytes})"
        )
    base_key = f"events/{event_id}/{sub_key}" if sub_key else f"events/{event_id}.mp4"
    # India clips get an `IN/` prefix (isolated + liftable); US unchanged. The
    # returned key is stored on events.clip_storage_key and reused for
    # signed-URL / delete, so intermediates + final stay self-consistent.
    key = region_routing.storage_key(region, base_key)
    await upload_object(
        bucket=settings.storage_bucket_incident_clips,
        key=key,
        body=mp4_bytes,
        content_type="video/mp4",
        settings=settings,
        timeout_seconds=settings.storage_clip_upload_timeout_seconds,
    )
    return key


async def download_object(
    *,
    bucket: str,
    key: str,
    settings: Settings,
    timeout_seconds: float | None = None,
) -> bytes:
    """Fetch a stored object's bytes via the service-role key.

    Used by the clip-archival concat pass to pull pre.mp4 / post.mp4
    back into the worker for ffmpeg stitching. Not exposed to clients.

    Raises:
        StorageError: object missing or storage backend errored.
    """
    base_url = settings.supabase_url.rstrip("/")
    object_url = f"{base_url}/storage/v1/object/{bucket}/{key}"
    headers = {"Authorization": f"Bearer {settings.supabase_service_role_key}"}
    effective_timeout = (
        timeout_seconds
        if timeout_seconds is not None
        else settings.storage_clip_upload_timeout_seconds
    )
    try:
        async with httpx.AsyncClient(timeout=effective_timeout) as client:
            response = await client.get(object_url, headers=headers)
    except httpx.TimeoutException as exc:
        raise StorageError(f"download failed: timeout after {effective_timeout}s: {exc}") from exc
    except httpx.RequestError as exc:
        raise StorageError(f"download failed: transport error: {exc}") from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise StorageError(
            f"download failed: {response.status_code} {response.reason_phrase}: "
            f"{response.text}"
        )
    return response.content


async def delete_object(
    *,
    bucket: str,
    key: str,
    settings: Settings,
) -> None:
    """Remove an object from Supabase Storage.

    Used by intermediate-clip cleanup and patient-photo deletion. A
    404 is treated as success (the caller's intent — "make sure this
    object isn't there" — is satisfied).

    Raises:
        StorageError: storage backend errored on a non-404 response.
    """
    base_url = settings.supabase_url.rstrip("/")
    delete_url = f"{base_url}/storage/v1/object/{bucket}/{key}"
    headers = {"Authorization": f"Bearer {settings.supabase_service_role_key}"}
    try:
        async with httpx.AsyncClient(timeout=settings.storage_upload_timeout_seconds) as client:
            response = await client.delete(delete_url, headers=headers)
    except httpx.TimeoutException as exc:
        raise StorageError(f"delete failed: timeout: {exc}") from exc
    except httpx.RequestError as exc:
        raise StorageError(f"delete failed: transport error: {exc}") from exc
    if response.status_code == 404:
        log.info("storage.delete.absent bucket=%s key=%s", bucket, key)
        return
    if response.status_code < 200 or response.status_code >= 300:
        raise StorageError(
            f"delete failed: {response.status_code} {response.reason_phrase}: "
            f"{response.text}"
        )
    log.info("storage.delete.success bucket=%s key=%s", bucket, key)


async def signed_url(
    *,
    bucket: str,
    key: str,
    settings: Settings,
    ttl_seconds: int | None = None,
) -> str:
    """Mint a short-lived signed URL for the caregiver app to fetch.

    `ttl_seconds` defaults to `settings.storage_signed_url_default_ttl_seconds`.
    Supabase returns a path relative to `/storage/v1`; we prepend the
    project's base URL so callers get a directly-fetchable absolute URL.

    Raises:
        StorageError: object missing or storage backend errored.
    """
    base_url = settings.supabase_url.rstrip("/")
    sign_url = f"{base_url}/storage/v1/object/sign/{bucket}/{key}"
    effective_ttl = (
        ttl_seconds
        if ttl_seconds is not None
        else settings.storage_signed_url_default_ttl_seconds
    )
    headers = {
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
    }
    payload = {"expiresIn": effective_ttl}
    try:
        async with httpx.AsyncClient(timeout=settings.storage_upload_timeout_seconds) as client:
            response = await client.post(sign_url, json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise StorageError(f"sign failed: timeout: {exc}") from exc
    except httpx.RequestError as exc:
        raise StorageError(f"sign failed: transport error: {exc}") from exc
    if response.status_code < 200 or response.status_code >= 300:
        raise StorageError(
            f"sign failed: {response.status_code} {response.reason_phrase}: "
            f"{response.text}"
        )
    try:
        signed_path = response.json()["signedURL"]
    except (KeyError, ValueError) as exc:
        raise StorageError(f"sign response missing signedURL: {response.text}") from exc
    # Supabase returns a `/object/sign/{bucket}/{key}?token=...` path
    # starting with `/`. Strip the leading `/` so the join is unambiguous.
    rel = signed_path.lstrip("/")
    return f"{base_url}/storage/v1/{rel}"
