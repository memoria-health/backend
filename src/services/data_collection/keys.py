"""Object-key construction for data-collection clips uploaded to B2.

Pure helpers — no IO. A clip's key is deterministic from
(caregiver, patient, session-start UTC, room):

    nemara-data/v1/<caregiver_id>/<patient_id>/<YYYY>/<MM>/<DD>/<HHMMSS>Z__<room-slug>.mp4

The date-partitioned layout sorts chronologically in a flat bucket
listing, and scopes every clip under its owning caregiver + patient so
an offline consumer can walk one patient's history with a prefix
listing. Times are UTC with no colons (colons are legal in S3 keys but
trip up downloaded-filename handling on some OSes). The room label is
slugified so an arbitrary name ("Mom's Bedroom") can't inject path
separators, spaces, or unicode into the key.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import UUID

_KEY_PREFIX = "nemara-data/v1"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(value: str, *, fallback: str = "room") -> str:
    """Lowercase, collapse any run of non-alphanumerics to a single
    ``-``, and trim leading/trailing ``-``. Returns ``fallback`` when
    nothing survives (empty string, all punctuation)."""
    slug = _NON_ALNUM.sub("-", value.lower()).strip("-")
    return slug or fallback


def build_clip_key(
    *,
    caregiver_id: UUID,
    patient_id: UUID,
    room: str,
    session_start: datetime,
) -> str:
    """Build the B2 object key for one presence-session clip.

    ``session_start`` is normalized to UTC; a naive datetime is assumed
    to already be UTC.
    """
    ts = (
        session_start.astimezone(timezone.utc)
        if session_start.tzinfo is not None
        else session_start.replace(tzinfo=timezone.utc)
    )
    room_slug = slugify(room)
    return (
        f"{_KEY_PREFIX}/{caregiver_id}/{patient_id}/"
        f"{ts:%Y}/{ts:%m}/{ts:%d}/{ts:%H%M%S}Z__{room_slug}.mp4"
    )
