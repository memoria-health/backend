"""Reviewer endpoints for the human-in-the-loop alert gate."""

from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.session import get_sessionmaker
from src.services import review_gate, storage

router = APIRouter(prefix="/v1/review", tags=["review"])

# Reviewer/admin allow-list: these accounts see and act on EVERY patient's
# held events regardless of caregiver ownership. Separate from caregiver
# accounts on purpose. (Move to settings/DB when this leaves local dev.)
REVIEWER_EMAILS = {"admin@gmail.com"}


def _is_admin(auth) -> bool:
    return str(auth.email).lower() in REVIEWER_EMAILS


class QueueItem(BaseModel):
    event_id: str
    title: str
    narrative: str
    patient_name: str
    room: str
    occurred_at: str
    seconds_left: int
    clip_url: str | None = None  # signed URL once archival lands; null while processing


class ReviewAct(BaseModel):
    approve: bool


class DeviceBody(BaseModel):
    apns_token: str
    environment: str = "sandbox"


class ToggleBody(BaseModel):
    patient_id: str
    require_review: bool


@router.get("/queue", response_model=list[QueueItem])
async def queue(auth: CaregiverAuth, session: SessionDep, settings: SettingsDep) -> list[QueueItem]:
    rows = (await session.execute(text(
        """select e.id, e.title, coalesce(e.narrative,''),
                  coalesce(nullif(p.preferred_name,''), p.full_name),
                  coalesce(e.location,''), e.occurred_at,
                  greatest(0, :auto - extract(epoch from now() - e.created_at))::int,
                  coalesce(e.review_clip_key, case when e.has_clip then e.clip_storage_key end)
           from events e join patients p on p.id = e.patient_id
           where (:admin or p.caregiver_id = :cg) and e.review_gate = 'pending'
           order by e.created_at desc limit 20"""),
        {"cg": str(auth.id), "admin": _is_admin(auth),
         "auto": review_gate.AUTO_RELEASE_S})).all()
    out: list[QueueItem] = []
    for r in rows:
        clip_url = None
        if r[7]:
            try:
                clip_url = await storage.signed_url(
                    bucket=settings.storage_bucket_incident_clips,
                    key=r[7], settings=settings)
            except Exception:  # noqa: BLE001 — clip is best-effort in the queue
                clip_url = None
        out.append(QueueItem(
            event_id=str(r[0]), title=r[1], narrative=r[2], patient_name=r[3],
            room=r[4], occurred_at=r[5].isoformat(), seconds_left=r[6],
            clip_url=clip_url))
    return out


@router.post("/device", status_code=status.HTTP_204_NO_CONTENT)
async def register_device(body: DeviceBody, auth: CaregiverAuth, session: SessionDep) -> None:
    """Register this reviewer phone for review-needed pushes (idempotent)."""
    await session.execute(text(
        """insert into review_devices (apns_token, user_id, environment)
           values (:t, :u, :e)
           on conflict (apns_token) do update set user_id = :u, environment = :e"""),
        {"t": body.apns_token, "u": str(auth.id), "e": body.environment})
    await session.commit()


@router.post("/toggle", status_code=status.HTTP_204_NO_CONTENT)
async def toggle(body: ToggleBody, auth: CaregiverAuth, session: SessionDep) -> None:
    row = (await session.execute(text(
        """update patients set require_review = :rr
           where id = :pid and (:admin or caregiver_id = :cg) returning id"""),
        {"rr": body.require_review, "pid": body.patient_id,
         "cg": str(auth.id), "admin": _is_admin(auth)})).first()
    if row is None:
        raise HTTPException(status_code=404, detail="patient_not_found")
    await session.commit()


@router.post("/{event_id}")
async def act(event_id: UUID, body: ReviewAct, auth: CaregiverAuth,
              session: SessionDep, settings: SettingsDep) -> dict:
    owned = (await session.execute(text(
        """select 1 from events e join patients p on p.id = e.patient_id
           where e.id = :id and (:admin or p.caregiver_id = :cg)"""),
        {"id": str(event_id), "cg": str(auth.id), "admin": _is_admin(auth)})).first()
    if owned is None:
        raise HTTPException(status_code=404, detail="event_not_found")
    result = await review_gate.act(
        sessionmaker=get_sessionmaker(),
        event_id=event_id, approve=body.approve, settings=settings)
    return {"result": result}
