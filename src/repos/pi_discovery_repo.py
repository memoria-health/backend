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

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

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
    """
    cutoff = datetime.now(UTC) - timedelta(seconds=seen_within_seconds)
    result = await session.execute(
        select(PiDiscovery)
        .where(PiDiscovery.claimed_at.is_(None))
        .where(PiDiscovery.last_seen_at >= cutoff)
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
    session: AsyncSession, *, discovery_id: UUID, camera_id: UUID,
) -> None:
    """Mark a discovery as claimed and link it to the new cameras
    row. Caller has already verified the discovery exists and is
    unclaimed.
    """
    now = datetime.now(UTC)
    await session.execute(
        update(PiDiscovery)
        .where(PiDiscovery.id == discovery_id)
        .values(claimed_at=now, claimed_camera_id=camera_id)
    )
    await session.flush()
