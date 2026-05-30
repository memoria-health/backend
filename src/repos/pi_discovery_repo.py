"""Repository for the pi_discoveries aggregate.

The discovery flow has two write paths and one read path:

  - `upsert_announcement` is called by the Pi-side announce daemon
    via POST /v1/cameras/pi-announce. Inserts a new row or updates
    the existing one keyed by hardware_id, refreshing last_seen_at
    and current_url so a Pi that gets a new DHCP lease pushes its
    new address within ~60 s.

  - `claim` is called by POST /v1/cameras/pi-provision when a
    caregiver picks a discovered Pi from the "Add a camera" list.
    Marks the discovery as claimed and links it to the new
    cameras row.

  - `list_unclaimed_recent` is called by GET /v1/cameras/discovered
    to populate the picker. Filters to discoveries not yet claimed
    and seen within `seen_within_seconds` so stale devices that
    have fallen off the network do not pollute the list.
"""

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import exists, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.pi_discovery import PiDiscovery


async def upsert_announcement(
    session: AsyncSession,
    *,
    hardware_id: str,
    device_label: str,
    current_url: str,
    current_audio_url: str | None,
) -> PiDiscovery:
    """Insert a new discovery or refresh an existing one (keyed by
    hardware_id). Returns the live row.

    Uses PostgreSQL's INSERT ... ON CONFLICT to make the upsert
    atomic; without it, two concurrent announcements from the same
    Pi could race to insert.
    """
    now = datetime.now(UTC)
    stmt = (
        pg_insert(PiDiscovery)
        .values(
            id=uuid4(),
            hardware_id=hardware_id,
            device_label=device_label,
            current_url=current_url,
            current_audio_url=current_audio_url,
            first_seen_at=now,
            last_seen_at=now,
        )
        .on_conflict_do_update(
            index_elements=["hardware_id"],
            set_={
                "device_label": device_label,
                "current_url": current_url,
                "current_audio_url": current_audio_url,
                "last_seen_at": now,
            },
        )
        .returning(PiDiscovery)
    )
    result = await session.execute(stmt)
    await session.flush()
    return result.scalar_one()


async def list_unclaimed_recent(
    session: AsyncSession, *, seen_within_seconds: int,
) -> Sequence[PiDiscovery]:
    """All unclaimed discoveries seen within the recent window.

    Used by the caregiver-facing list endpoint to populate the
    "Add a camera" picker. The recency filter is the only thing
    preventing dead Pis (e.g. unplugged for days) from showing up.

    The picker also excludes any discovery whose stream URL already
    matches an existing camera row. This is defensive: the normal
    web pair flow always claims the discovery (which already removes
    it from the list), but cameras created via direct-URL mode
    (provision scripts, curl-based admin tooling) skip the claim
    step and leave the discovery orphaned. Without this URL filter
    the picker shows the same Pi as "available" forever even though
    it's already in the patient's cameras list.
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=seen_within_seconds)
    already_paired = exists().where(
        Camera.pi_stream_url == PiDiscovery.current_url,
    )
    result = await session.execute(
        select(PiDiscovery)
        .where(PiDiscovery.claimed_at.is_(None))
        .where(PiDiscovery.last_seen_at >= cutoff)
        .where(~already_paired)
        .order_by(PiDiscovery.last_seen_at.desc())
    )
    return result.scalars().all()


async def find_by_id(
    session: AsyncSession, discovery_id: UUID
) -> PiDiscovery | None:
    """Lookup by primary key. Used during /pi-provision to resolve
    a discovery_id picked from the list to the URLs to store.
    """
    result = await session.execute(
        select(PiDiscovery).where(PiDiscovery.id == discovery_id)
    )
    return result.scalar_one_or_none()


async def claim(
    session: AsyncSession,
    *,
    discovery_id: UUID,
    camera_id: UUID,
    pending_ingest_token: str,
) -> None:
    """Mark a discovery as claimed, link it to the new cameras row,
    and stash the plaintext ingest token for one-shot delivery on
    the Pi's next announce. Caller has already verified the
    discovery exists and is unclaimed.
    """
    now = datetime.now(UTC)
    await session.execute(
        update(PiDiscovery)
        .where(PiDiscovery.id == discovery_id)
        .values(
            claimed_at=now,
            claimed_camera_id=camera_id,
            pending_ingest_token=pending_ingest_token,
        )
    )
    await session.flush()


async def unclaim_for_camera(
    session: AsyncSession, *, camera_id: UUID,
) -> None:
    """Reset claim state on whichever pi_discovery currently points at
    `camera_id`. Used by the camera-delete handlers (caregiver-facing
    `delete_camera` and wall-phone `delete_self`) to keep the
    discovery row from rotting into a zombie state when the camera
    it claims is removed.

    Background: `pi_discoveries.claimed_camera_id` is `ON DELETE SET
    NULL`. Without this call, deleting a Pi-typed camera leaves the
    discovery with `claimed_camera_id=NULL` but `claimed_at` still
    populated. `list_unclaimed_recent` filters on `claimed_at IS NULL`,
    so the Pi can keep announcing forever without re-appearing in the
    caregiver-app picker. We saw this hit 2026-05-28 17:03 PDT when
    the user deleted the bedroom camera and the Pi vanished from
    "available cameras." Resetting the claim fields here keeps the
    discovery picker-ready on the very next announce.

    No-op when no discovery references `camera_id` (the iPhone path,
    where there is no discovery row at all).
    """
    await session.execute(
        update(PiDiscovery)
        .where(PiDiscovery.claimed_camera_id == camera_id)
        .values(
            claimed_at=None,
            claimed_camera_id=None,
            pending_ingest_token=None,
        )
    )
    await session.flush()


async def consume_pending_ingest_token(
    session: AsyncSession, *, hardware_id: str,
) -> tuple[UUID, str] | None:
    """Atomically read-and-null the pending_ingest_token for the
    discovery row matching `hardware_id`. Returns
    `(claimed_camera_id, plaintext_token)` if a token was waiting,
    or None if there was no waiting token. Used by the announce
    handler to deliver the token exactly once.

    Implementation: SELECT ... FOR UPDATE locks the row, then UPDATE
    nulls the token. Both run inside the caller's transaction. A
    concurrent announce from the same Pi blocks on the row lock,
    then sees the nulled token and returns None. So the token is
    delivered to exactly one HTTP response.
    """
    select_stmt = (
        select(PiDiscovery.claimed_camera_id, PiDiscovery.pending_ingest_token)
        .where(PiDiscovery.hardware_id == hardware_id)
        .where(PiDiscovery.pending_ingest_token.is_not(None))
        .with_for_update()
    )
    row = (await session.execute(select_stmt)).one_or_none()
    if row is None:
        return None
    claimed_camera_id, plaintext = row
    if claimed_camera_id is None or plaintext is None:
        return None
    await session.execute(
        update(PiDiscovery)
        .where(PiDiscovery.hardware_id == hardware_id)
        .values(pending_ingest_token=None)
    )
    await session.flush()
    return claimed_camera_id, plaintext
