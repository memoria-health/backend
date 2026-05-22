"""Event API surface.

- GET   /v1/patients/{id}/events  — list, with filter + cursor pagination.
- GET   /v1/events/{id}           — fetch one (for Event Detail).
- PATCH /v1/events/{id}           — update opened / reviewed / review_choice / review_notes.

Events are INSERTED by wall phones in a later plan. The caregiver app
never POSTs an event in v0.
"""

from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status

from src.api.deps import CaregiverAuth, SessionDep
from src.db.event import Event
from src.repos import event_repo
from src.schemas.event import (
    EventFilter,
    EventListResponse,
    EventResponse,
    UpdateEventRequest,
)

router = APIRouter(prefix="/v1", tags=["event"])


def _to_response(e: Event) -> EventResponse:
    return EventResponse.model_validate(e, from_attributes=True)


@router.get(
    "/patients/{patient_id}/events",
    response_model=EventListResponse,
)
async def list_events(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    filter: EventFilter = Query(EventFilter.all),
    cursor: str | None = Query(None, description="Opaque cursor from the previous page."),
    limit: int = Query(30, ge=1, le=100),
) -> EventListResponse:
    only_unreviewed: bool | None
    if filter is EventFilter.unreviewed:
        only_unreviewed = True
    elif filter is EventFilter.reviewed:
        only_unreviewed = False
    else:
        only_unreviewed = None

    rows, next_cursor = await event_repo.list_for_patient(
        session,
        patient_id,
        caregiver_id=auth.id,
        only_unreviewed=only_unreviewed,
        cursor=cursor,
        limit=limit,
    )
    return EventListResponse(
        items=[_to_response(e) for e in rows],
        next_cursor=next_cursor,
    )


@router.get("/events/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> EventResponse:
    e = await event_repo.find_by_id(session, event_id, caregiver_id=auth.id)
    if e is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="event_not_found",
        )
    return _to_response(e)


@router.patch("/events/{event_id}", response_model=EventResponse)
async def patch_event(
    event_id: UUID,
    auth: CaregiverAuth,
    body: UpdateEventRequest,
    session: SessionDep,
) -> EventResponse:
    """Mark opened / reviewed.

    Server stamps `reviewed_at = now()` when `reviewed` flips from
    `false` to `true`. Setting `reviewed = false` does NOT clear
    `reviewed_at` (keeps an audit trail of the original review).
    """
    fields = body.model_dump(exclude_unset=True)
    if "review_choice" in fields and fields["review_choice"] is not None:
        fields["review_choice"] = fields["review_choice"].value

    # `reviewed_at` is server-stamped when the caregiver first marks
    # the event reviewed. We need the prior row to decide whether to
    # stamp; do a pre-read.
    if fields.get("reviewed") is True:
        prior = await event_repo.find_by_id(session, event_id, caregiver_id=auth.id)
        if prior is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="event_not_found",
            )
        if not prior.reviewed:
            fields["reviewed_at"] = datetime.now(timezone.utc)

    e = await event_repo.update_partial(
        session, event_id, caregiver_id=auth.id, fields=fields
    )
    if e is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="event_not_found",
        )
    await session.commit()
    return _to_response(e)
