"""Which patients a camera watches.

A camera belongs to an ACCOUNT and covers 0..N patients. One of them is
the PRIMARY: the camera publishes into that patient's LiveKit room, and
everyone else covered receives the results rather than a second copy of
the video. Coverage with no rows means the camera is provisioned but
nobody has said who it watches — it is absent from every roster and so
never publishes.

`cameras.patient_id` is still written here, kept equal to the primary,
because ~35 readers have not migrated yet. It is dropped once they have.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def coverage(session: AsyncSession, camera_id: UUID) -> list[UUID]:
    """Every patient this camera watches, primary first."""
    rows = (await session.execute(
        text(
            "SELECT patient_id FROM public.camera_patients "
            " WHERE camera_id = :c ORDER BY is_primary DESC, added_at"
        ),
        {"c": camera_id},
    )).all()
    return [r[0] for r in rows]


async def primary_of(session: AsyncSession, camera_id: UUID) -> UUID | None:
    return (await session.execute(
        text(
            "SELECT patient_id FROM public.camera_patients "
            " WHERE camera_id = :c AND is_primary"
        ),
        {"c": camera_id},
    )).scalar_one_or_none()


async def set_coverage(
    session: AsyncSession,
    camera_id: UUID,
    *,
    primary_patient_id: UUID,
    also_patient_ids: list[UUID] | None = None,
) -> None:
    """Make these patients the camera's coverage, replacing what was there.

    The camera's LiveKit IDENTITY is never touched — the bridge pins it
    locally at provision time and would lose the camera if it changed. Only
    the ROOM moves, which the bridge re-reads from the roster on its next
    poll, so re-pointing a camera at a different patient needs nothing from
    the bridge.
    """
    also = [p for p in (also_patient_ids or []) if p != primary_patient_id]

    await session.execute(
        text("DELETE FROM public.camera_patients WHERE camera_id = :c"),
        {"c": camera_id},
    )
    await session.execute(
        text(
            "INSERT INTO public.camera_patients (camera_id, patient_id, is_primary) "
            "VALUES (:c, :p, true)"
        ),
        {"c": camera_id, "p": primary_patient_id},
    )
    for pid in also:
        await session.execute(
            text(
                "INSERT INTO public.camera_patients "
                "  (camera_id, patient_id, is_primary) "
                "VALUES (:c, :p, false) ON CONFLICT DO NOTHING"
            ),
            {"c": camera_id, "p": pid},
        )

    await session.execute(
        text(
            "UPDATE public.cameras "
            "   SET patient_id = :p, livekit_room = :room "
            " WHERE id = :c"
        ),
        {"c": camera_id, "p": primary_patient_id,
         "room": f"patient_{primary_patient_id}"},
    )


async def clear_coverage(session: AsyncSession, camera_id: UUID) -> None:
    """Unassign: the camera stays on the account and stops streaming.

    It leaves every roster on the next poll, so the bridge tears its
    publisher down without being told anything else.
    """
    await session.execute(
        text("DELETE FROM public.camera_patients WHERE camera_id = :c"),
        {"c": camera_id},
    )
    await session.execute(
        text(
            "UPDATE public.cameras "
            "   SET patient_id = NULL, livekit_room = NULL WHERE id = :c"
        ),
        {"c": camera_id},
    )
