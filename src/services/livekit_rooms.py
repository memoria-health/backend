"""What a LiveKit room is called, and who is allowed into it.

One place, because the room name used to be built by string interpolation
in eleven of them and READ by parsing that string back apart in five more.
Parsing was the actual problem: `patient_<uuid>` looked like it carried
the authorization answer, so callers took the patient out of the name and
trusted it, which quietly made the room name a security boundary.

It is an ADDRESS. `services/identity/identity_router` already says so of
itself -- "the room name embeds a patient uuid for historical reasons and
nothing downstream is allowed to read meaning into that" -- and this
module is that rule made enforceable rather than aspirational.

Two shapes exist on purpose:

  account_<caregiver_id>   every sensor on an account, sharing one room
  patient_<patient_id>     wall phones paired before 2026-08-22

A wall phone is handed its room ONCE, in the pairing response, and never
re-reads it, so its room cannot be renamed underneath it -- it would
simply publish into a room nobody joins. Gateway and bridge sensors
re-read theirs from the fleet poll every few seconds, which is why those
could be moved and these cannot. The old shape therefore has to keep
working until the last phone re-pairs, and `resolve_*` below answers
questions about a room by LOOKING IT UP rather than by believing its name.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

ACCOUNT_PREFIX = "account_"
# Wall phones paired before the account-room migration. New rows never
# get this shape; it is read-only compatibility.
LEGACY_PATIENT_PREFIX = "patient_"


def account_room(caregiver_id: UUID | str) -> str:
    """The room every sensor on this account publishes into.

    Known at provision time, because an installer types a caregiver's
    email and nothing else. That is the whole reason a sensor can stream
    before anyone has said which patient it watches.
    """
    return f"{ACCOUNT_PREFIX}{str(caregiver_id).lower()}"


def is_room_name(room: str) -> bool:
    """A well-formed room of either shape. Not an authorization check --
    see `caregiver_may_join`, which is."""
    return room.startswith((ACCOUNT_PREFIX, LEGACY_PATIENT_PREFIX))


async def caregiver_may_join(
    session: AsyncSession, *, caregiver_id: UUID, room: str,
) -> bool:
    """May this caregiver join this room? Resolved, never parsed.

    The old check took the uuid out of `patient_<uuid>` and asked whether
    the caregiver could reach that patient. That happened to be correct
    while every room was a patient's, and it stops being correct the
    moment a room is an account's -- but it would still PARSE, and a
    check that keeps parsing after its assumption dies is the shape of
    an access-control bug rather than an outage.

    So: an account room is joinable by the account it names, and a legacy
    patient room by a caregiver who can reach that patient. Anything that
    resolves to neither is refused, including a room that is merely
    well-formed. Unknown room, malformed room and unauthorized room all
    return the same False -- a caller probing for "does this home exist"
    learns nothing from the answer.
    """
    if room.startswith(ACCOUNT_PREFIX):
        return str(caregiver_id).lower() == room[len(ACCOUNT_PREFIX):].lower()

    if room.startswith(LEGACY_PATIENT_PREFIX):
        try:
            patient_id = UUID(room[len(LEGACY_PATIENT_PREFIX):])
        except ValueError:
            return False
        found = (await session.execute(
            text("SELECT 1 FROM public.patients "
                 " WHERE id = :p AND caregiver_id = :c"),
            {"p": str(patient_id), "c": str(caregiver_id)},
        )).first()
        return found is not None

    return False
