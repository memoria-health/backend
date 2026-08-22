"""Which patients a sensor watches.

A sensor belongs to an ACCOUNT and covers 0..N patients, as equals. There
is no primary. There was one until 2026-08-22, and it existed for exactly
one reason: the room was named `patient_<uuid>`, a camera publishes into
exactly one room, so a sensor covering two people had to elect whose room
it landed in. Naming the room after the account retired the question, and
`apply_account_room_delta.sql` dropped the column.

Coverage is therefore an unordered set, displayed oldest-added first.
Nothing about the stream depends on it: a sensor with no coverage still
publishes, and adding or removing a patient never moves the video. That
is the point -- assignment is attribution, not plumbing.

`cameras.patient_id` is still written here, set to the earliest-added
covered patient, because 77 call sites have not migrated off it. It is
authoritative for NOTHING -- not the room, not the roster, not detector
links, all of which now resolve through `camera_patients`. It is a
compatibility shim with its own delta pending, and new code must not read
it.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def coverage(session: AsyncSession, camera_id: UUID) -> list[UUID]:
    """Every patient this sensor watches, oldest assignment first."""
    rows = (await session.execute(
        text(
            "SELECT patient_id FROM public.camera_patients "
            " WHERE camera_id = :c ORDER BY added_at, patient_id"
        ),
        {"c": camera_id},
    )).all()
    return [r[0] for r in rows]


async def camera_ids_for_patient(
    session: AsyncSession, patient_id: UUID,
) -> list[UUID]:
    """Every sensor watching this patient.

    The inverse direction, which used to be `cameras.patient_id == pid`
    and so could only ever return sensors this patient was PRIMARY on --
    a shared sensor was invisible to its second patient.
    """
    rows = (await session.execute(
        text(
            "SELECT camera_id FROM public.camera_patients "
            " WHERE patient_id = :p ORDER BY added_at, camera_id"
        ),
        {"p": patient_id},
    )).all()
    return [r[0] for r in rows]


async def set_coverage(
    session: AsyncSession, camera_id: UUID, patient_ids: list[UUID],
) -> None:
    """Make these patients the sensor's coverage, replacing what was there.

    An empty list is legal and means "watched by nobody yet" -- the state
    every sensor is in between the installer leaving and the caregiver
    opening the app. It keeps streaming throughout.

    The sensor's LiveKit identity and room are never touched here. The
    identity is pinned locally by the bridge at provision time and it
    would lose the sensor if it changed; the room belongs to the account,
    which coverage cannot alter. So re-pointing a sensor at a different
    patient needs nothing from the bridge and interrupts no video.
    """
    wanted = list(dict.fromkeys(patient_ids))  # de-dupe, keep order

    await session.execute(
        text("DELETE FROM public.camera_patients WHERE camera_id = :c"),
        {"c": camera_id},
    )
    for patient_id in wanted:
        await session.execute(
            text(
                "INSERT INTO public.camera_patients (camera_id, patient_id) "
                "VALUES (:c, :p) ON CONFLICT DO NOTHING"
            ),
            {"c": camera_id, "p": patient_id},
        )

    # Compatibility shim, not a primary. See the module docstring.
    await session.execute(
        text("UPDATE public.cameras SET patient_id = :p WHERE id = :c"),
        {"c": camera_id, "p": wanted[0] if wanted else None},
    )


async def clear_coverage(session: AsyncSession, camera_id: UUID) -> None:
    """Unassign a sensor. It stays on the account and keeps streaming."""
    await set_coverage(session, camera_id, [])
