"""LiveKit Cloud webhook receiver.

LiveKit posts JSON-encoded `WebhookEvent`s to this endpoint with a
signed `Authorization` header. We verify the signature against
`NEMARA_LIVEKIT_WEBHOOK_API_KEY` (or `LIVEKIT_API_KEY` as a fallback),
then dispatch the event to the in-process `DetectorWorker`:

  - `participant_joined` in a `patient_<id>` room with a `cam_<...>`
    identity → `DetectorWorker.ensure_room_joined(room)`
  - `participant_left` for the only `cam_<...>` participant → the
    worker tears down its `RoomWorker` for that room. (Phase 2 v0
    treats every `participant_left` as a candidate to leave;
    re-joining is free.)

We acknowledge with 204 on success, 401 on signature failure. Other
event types (`room_started`, `egress_started`, etc.) are accepted
with 204 but not acted on — Phase 4 may grow the dispatcher.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status
from livekit import api as lkapi

from src.api.deps import SettingsDep

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["livekit"])


@router.post("/livekit/webhooks", status_code=status.HTTP_204_NO_CONTENT)
async def receive_livekit_webhook(
    request: Request,
    settings: SettingsDep,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> None:
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        log.error("livekit_webhook: LIVEKIT_API_KEY/SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="livekit_not_configured",
        )
    if not authorization:
        log.info("livekit_webhook reject reason=missing_auth_header")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_auth")

    # Body is verified verbatim — the signed JWT in `Authorization`
    # commits to the body's SHA-256, so any tamper invalidates it.
    body = (await request.body()).decode("utf-8")

    verifier_key = settings.livekit_webhook_api_key or settings.livekit_api_key
    receiver = lkapi.WebhookReceiver(
        lkapi.TokenVerifier(verifier_key, settings.livekit_api_secret),
    )
    try:
        event = receiver.receive(body, authorization)
    except Exception as exc:
        log.info("livekit_webhook reject reason=signature_failed err=%s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_signature",
        ) from exc

    worker = getattr(request.app.state, "detector_worker", None)
    if worker is None:
        # Lifespan never started the worker (disabled or
        # mis-configured). Accept the webhook silently so LiveKit
        # doesn't retry forever, but log loudly so operators notice.
        log.warning(
            "livekit_webhook event=%s but no detector_worker in app.state",
            event.event,
        )
        return

    room_name = event.room.name if event.room else ""
    participant_identity = event.participant.identity if event.participant else ""

    # Only patient rooms are interesting to the detector. Other rooms
    # (e.g. future surface area) are accepted-but-ignored.
    if not room_name.startswith("patient_"):
        return

    if event.event == "participant_joined" and participant_identity.startswith("cam_"):
        await worker.ensure_room_joined(room_name)
        # Webhook is a fast-path HINT, not authoritative for
        # `cameras.online`. Ask the reconciler to do an immediate poll
        # so the column updates in <1s instead of waiting up to 10s
        # for the next periodic tick. The reconciler's REST poll +
        # frame-arrival check is the actual source of truth.
        await worker.signal_reconcile(room_name)
        log.info(
            "livekit_webhook participant_joined room=%s identity=%s -> joined",
            room_name, participant_identity,
        )
    elif event.event == "participant_left" and participant_identity.startswith("cam_"):
        await worker.ensure_room_left_if_empty(room_name)
        await worker.signal_reconcile(room_name)
        log.info(
            "livekit_webhook participant_left room=%s identity=%s -> left-check",
            room_name, participant_identity,
        )
    # Other event types are no-ops in v0.
