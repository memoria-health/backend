"""Event API surface.

- GET   /v1/patients/{id}/events  — list, with filter + cursor pagination.
- GET   /v1/events/{id}           — fetch one (for Event Detail).
- PATCH /v1/events/{id}           — update opened / reviewed / review_choice / review_notes.

Events are INSERTED by wall phones in a later plan. The caregiver app
never POSTs an event in v0.
"""

from datetime import datetime, timezone
from uuid import UUID

import logging

from fastapi import APIRouter, HTTPException, Query, Response, status

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.event import Event
from src.repos import event_repo
from src.schemas.event import (
    ClipURLResponse,
    EventFilter,
    EventListResponse,
    EventResponse,
    UpdateEventRequest,
)
from src.services import storage

router = APIRouter(prefix="/v1", tags=["event"])


log = logging.getLogger(__name__)


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


@router.get("/events/{event_id}/clip-url", response_model=ClipURLResponse)
async def get_event_clip_url(
    event_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> ClipURLResponse:
    """Mint a short-lived signed URL for the event's clip.

    Returns:
        - 200 + signed URL when `has_clip=true` and the caregiver owns the event.
        - 404 when the event doesn't exist (or isn't owned by the caregiver).
        - 409 when the clip is still processing (`has_clip=false`).

    Mints a fresh URL every call rather than caching — iOS only asks
    when the player opens, and the TTL is short enough that re-mints
    are cheap.
    """
    e = await event_repo.find_by_id(session, event_id, caregiver_id=auth.id)
    if e is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event_not_found",
        )
    if not e.has_clip or e.clip_storage_key is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="clip_not_ready",
        )
    # OUR origin, not storage's. `storage.signed_url` builds its absolute
    # URL from `settings.supabase_url` — the address the BACKEND uses to
    # reach Supabase. In Docker that is host.docker.internal, natively
    # loopback or a LAN IP, and none of them resolve on a phone. The app
    # got a URL it could never fetch and showed a player that silently
    # refused to start, indistinguishable from a corrupt clip.
    #
    # The operator verdict page hit this first and already proxies
    # (`/act/{token}/clip`). Same reasoning applies here, plus the same
    # bonus: no separately-shareable storage URL for a household's
    # footage exists at any point.
    return ClipURLResponse(
        url=f"/v1/events/{event_id}/clip",
        expires_in_seconds=settings.storage_signed_url_default_ttl_seconds,
    )


@router.get("/events/{event_id}/clip")
async def get_event_clip(
    event_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> Response:
    """Stream the event's clip through the backend, under caregiver auth.

    `find_by_id` is scoped by `caregiver_id`, so another household's
    event is a 404 here exactly as it is everywhere else — proxying moves
    the bytes behind OUR auth, which only helps if that auth is real.
    """
    e = await event_repo.find_by_id(session, event_id, caregiver_id=auth.id)
    if e is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="event_not_found",
        )
    if not e.has_clip or e.clip_storage_key is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="clip_not_ready",
        )
    try:
        data = await storage.download_object(
            bucket=settings.storage_bucket_incident_clips,
            key=e.clip_storage_key,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("events.clip_proxy_failed event=%s", event_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="clip_unavailable",
        ) from exc
    return Response(
        content=data,
        media_type="video/mp4",
        headers={
            # Short and private: a household's footage must not sit in a
            # shared cache, and the player re-asks on each open anyway.
            "Cache-Control": "private, max-age=60",
            "Content-Disposition": "inline",
        },
    )
