"""Patient sharing endpoints — manage which caregivers can access a patient.

MVP rules:
  - Any MEMBER may LIST members.
  - Only an OWNER may share (add a member by email) or remove a member.
  - The last owner can't be removed.
Roles beyond 'owner' are stored but unenforced (every member has full access);
they're the seam for the future per-patient permission levels.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select

from src.api.deps import CaregiverAuth, SessionDep
from src.db.caregiver import Caregiver
from src.db.patient_member import PatientMember
from src.repos import access

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1/patients", tags=["patient_members"])

_SHAREABLE_ROLES = {"caregiver", "viewer"}


class ShareRequest(BaseModel):
    email: str
    role: str = "caregiver"


class MemberResponse(BaseModel):
    caregiver_id: UUID
    email: str
    role: str


async def _require_member(session, patient_id: UUID, caregiver_id: UUID) -> str:
    """Return the caller's role, or 404 (not 403 — don't reveal existence)."""
    role = await access.role_of(
        session, patient_id=patient_id, caregiver_id=caregiver_id
    )
    if role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Patient not found")
    return role


@router.get("/{patient_id}/members", response_model=list[MemberResponse])
async def list_patient_members(
    patient_id: UUID, auth: CaregiverAuth, session: SessionDep
) -> list[MemberResponse]:
    await _require_member(session, patient_id, auth.id)
    rows = (
        await session.execute(
            select(PatientMember.caregiver_id, Caregiver.email, PatientMember.role)
            .join(Caregiver, Caregiver.id == PatientMember.caregiver_id)
            .where(PatientMember.patient_id == patient_id)
            .order_by(PatientMember.created_at.asc())
        )
    ).all()
    return [MemberResponse(caregiver_id=r[0], email=r[1], role=r[2]) for r in rows]


@router.post(
    "/{patient_id}/members", response_model=MemberResponse, status_code=201
)
async def share_patient(
    patient_id: UUID, body: ShareRequest, auth: CaregiverAuth, session: SessionDep
) -> MemberResponse:
    if (await _require_member(session, patient_id, auth.id)) != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an owner can share")
    if body.role not in _SHAREABLE_ROLES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"role must be one of {sorted(_SHAREABLE_ROLES)}",
        )
    # MVP: existing accounts only. Case-insensitive email match.
    invitee = (
        await session.execute(
            select(Caregiver).where(func.lower(Caregiver.email) == body.email.strip().lower())
        )
    ).scalar_one_or_none()
    if invitee is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, "No Memoria account with that email"
        )
    if invitee.id == auth.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You already have access")
    await access.add_member(
        session,
        patient_id=patient_id,
        caregiver_id=invitee.id,
        role=body.role,
        invited_by=auth.id,
    )
    await session.commit()
    log.info(
        "patient_members.share patient=%s invitee=%s by=%s role=%s",
        patient_id, invitee.id, auth.id, body.role,
    )
    return MemberResponse(caregiver_id=invitee.id, email=invitee.email, role=body.role)


@router.delete("/{patient_id}/members/{member_id}", status_code=204)
async def remove_patient_member(
    patient_id: UUID, member_id: UUID, auth: CaregiverAuth, session: SessionDep
) -> None:
    if (await _require_member(session, patient_id, auth.id)) != "owner":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an owner can remove members")
    target_role = await access.role_of(
        session, patient_id=patient_id, caregiver_id=member_id
    )
    if target_role is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not a member")
    if target_role == "owner" and await access.owner_count(session, patient_id=patient_id) <= 1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Can't remove the last owner")
    await access.remove_member(
        session, patient_id=patient_id, caregiver_id=member_id
    )
    await session.commit()
    log.info("patient_members.remove patient=%s member=%s by=%s", patient_id, member_id, auth.id)
