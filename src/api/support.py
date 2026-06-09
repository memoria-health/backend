"""Contact Support API surface: POST /v1/support.

Caregivers submit messages from the iOS app's Settings → Contact
support sheet. The handler forwards the message via Resend to the
addresses in `settings.support_email_recipients`, with `replyTo` set
to the caregiver's account email so the team can hit reply.
"""

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.repos import caregiver_repo
from src.schemas.support import SupportRequest, SupportResponse
from src.services import email

router = APIRouter(prefix="/v1/support", tags=["support"])


@router.post("", response_model=SupportResponse)
async def post_support(
    body: SupportRequest,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> SupportResponse:
    cg = await caregiver_repo.find_by_id(session, auth.id)
    if cg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="caregiver_not_found",
        )
    result = await email.send_support_email(
        caregiver=cg,
        subject=body.subject,
        message=body.message,
        settings=settings,
    )
    return SupportResponse(status=result)
