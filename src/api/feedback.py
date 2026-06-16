"""Product Feedback API surface: POST /v1/feedback.

Caregivers submit product feedback from the iOS app's Settings → Send
feedback sheet. Unlike /v1/support (which forwards purely via email and
lets email errors propagate), feedback is PERSISTED first: the row in
the `feedback` table is the source of truth. After the row commits we
fire a best-effort notification email to the founders via Resend —
that email failing must NOT fail the request or lose the stored row.
"""

import logging

from fastapi import APIRouter, HTTPException, status

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.repos import caregiver_repo, feedback_repo
from src.schemas.feedback import FeedbackRequest, FeedbackResponse
from src.services import email

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def post_feedback(
    body: FeedbackRequest,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> FeedbackResponse:
    cg = await caregiver_repo.find_by_id(session, auth.id)
    if cg is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="caregiver_not_found",
        )

    # Source of truth: persist the row first and commit before we touch
    # email. If the email step later blows up, the feedback is safe.
    row = await feedback_repo.create(
        session,
        caregiver_id=cg.id,
        rating=body.rating,
        whats_working=body.whats_working,
        whats_frustrating=body.whats_frustrating,
        category=body.category,
    )
    await session.commit()

    # Best-effort founder notification. Differs from /v1/support: an
    # email failure here is logged and swallowed, never raised, so a
    # flaky Resend can't fail a request whose row is already committed.
    try:
        await email.send_feedback_email(
            caregiver=cg,
            rating=body.rating,
            whats_working=body.whats_working,
            whats_frustrating=body.whats_frustrating,
            category=body.category,
            attachments=body.attachments,
            settings=settings,
        )
    except Exception:  # noqa: BLE001
        log.exception("feedback.email_failed feedback_id=%s", row.id)

    return FeedbackResponse(status="received", id=row.id)
