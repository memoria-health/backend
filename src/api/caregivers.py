"""Caregiver API surface — `/v1/me` endpoints.

- POST  /v1/me     — provision a caregiver row (idempotent).
- GET   /v1/me     — fetch the current caregiver. 404 if no row.
- PATCH /v1/me     — partial update.

Auth required for all three (CaregiverAuth dep).
"""

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CaregiverAuth, SessionDep
from src.repos import caregiver_repo
from src.schemas.caregiver import (
    CaregiverResponse,
    CreateCaregiverRequest,
    UpdateCaregiverRequest,
)
from src.services import onboarding

router = APIRouter(prefix="/v1", tags=["caregiver"])


@router.post("/me", response_model=CaregiverResponse, status_code=status.HTTP_200_OK)
async def post_me(
    auth: CaregiverAuth,
    body: CreateCaregiverRequest,
    session: SessionDep,
) -> CaregiverResponse:
    """Provision a caregiver row for the JWT principal.

    Idempotent: if a row already exists for `auth.id`, returns it
    ignoring the request body.
    """
    cg = await onboarding.provision_caregiver(session, auth, body)
    await session.commit()
    return onboarding.to_response(cg)


@router.get("/me", response_model=CaregiverResponse)
async def get_me(auth: CaregiverAuth, session: SessionDep) -> CaregiverResponse:
    """Fetch the current caregiver row. 404 if not provisioned yet."""
    cg = await caregiver_repo.find_by_id(session, auth.id)
    if cg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="caregiver_not_found",
        )
    return onboarding.to_response(cg)


@router.patch("/me", response_model=CaregiverResponse)
async def patch_me(
    auth: CaregiverAuth,
    body: UpdateCaregiverRequest,
    session: SessionDep,
) -> CaregiverResponse:
    """Partial update of the current caregiver. Writable fields:
    `display_name`, `phone`. Email goes through Supabase auth — see
    the `UpdateCaregiverRequest` docstring.
    """
    cg = await onboarding.update_caregiver(session, auth, body)
    if cg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="caregiver_not_found",
        )
    await session.commit()
    return onboarding.to_response(cg)
