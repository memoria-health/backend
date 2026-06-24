"""Upload finished data-collection clips to Backblaze B2 (S3-compatible).

One function: stream a finished MP4 file to B2 with object metadata.
``aioboto3``'s ``upload_file`` switches to multipart automatically for
large files, so a ~280 MB clip uploads without being buffered in RAM.
Credentials + endpoint come from Settings (``NEMARA_B2_*``); callers
gate on :func:`b2_configured` so a box with no creds records locally
instead of crashing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CONTENT_TYPE = "video/mp4"


def b2_configured(settings: Any) -> bool:
    """True only when both B2 credentials are present. The recorder
    skips upload (and logs) when this is False, degrading to
    record-locally rather than failing on a credential-less box."""
    return bool(settings.b2_key_id) and bool(settings.b2_app_key)


async def upload_file_to_b2(
    path: Path | str,
    key: str,
    *,
    metadata: dict[str, Any],
    settings: Any,
    session: Any = None,
) -> str:
    """Upload the file at ``path`` to ``key`` in the B2 bucket.

    ``metadata`` is attached as S3 object metadata with values
    stringified (S3 requires string metadata values). Returns ``key``
    on success; propagates any client error so the caller owns
    retry/backoff.

    ``session`` is injectable for tests; in production it defaults to a
    fresh ``aioboto3.Session`` (imported lazily — botocore is heavy and
    only the real upload path needs it).
    """
    if session is None:
        import aioboto3

        session = aioboto3.Session()

    async with session.client(
        "s3",
        endpoint_url=settings.b2_endpoint_url,
        region_name=settings.b2_region,
        aws_access_key_id=settings.b2_key_id,
        aws_secret_access_key=settings.b2_app_key,
    ) as s3:
        await s3.upload_file(
            Filename=str(path),
            Bucket=settings.b2_bucket,
            Key=key,
            ExtraArgs={
                "ContentType": _CONTENT_TYPE,
                "Metadata": {k: str(v) for k, v in metadata.items()},
            },
        )
    return key
