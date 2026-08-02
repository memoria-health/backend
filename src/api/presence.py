"""Last-presence facts for the Home briefing — claims only about what the
pipeline actually observed (people_present analysis ticks)."""

from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from src.api.deps import CaregiverAuth, SessionDep

router = APIRouter(prefix="/v1", tags=["presence"])


class PresenceResponse(BaseModel):
    last_person_seen_at: str | None   # last tick with a person in view
    last_analysis_at: str | None      # last tick at all (were we looking?)


@router.get("/patients/{patient_id}/last-presence", response_model=PresenceResponse)
async def last_presence(
    patient_id: UUID, auth: CaregiverAuth, session: SessionDep,
) -> PresenceResponse:
    row = (await session.execute(text(
        """select
             max(captured_at) filter (where people_present) as seen,
             max(captured_at) as looked
           from analyses a
           join patients p on p.id = a.patient_id
           where a.patient_id = :pid and p.caregiver_id = :cg"""),
        {"pid": str(patient_id), "cg": str(auth.id)})).first()
    return PresenceResponse(
        last_person_seen_at=row[0].isoformat() if row and row[0] else None,
        last_analysis_at=row[1].isoformat() if row and row[1] else None,
    )
