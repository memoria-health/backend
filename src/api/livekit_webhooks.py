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

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, status
from livekit import api as lkapi
from starlette.requests import ClientDisconnect

from src.api.deps import SettingsDep
from src.db.session import get_sessionmaker
from src.services import livekit_privacy, livekit_rooms
from src.services.detection import camera_presence

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["livekit"])

# Fire-and-forget privacy re-sever tasks, held so the loop can't GC them
# (same pattern as room_classifier's attempt tasks).
_privacy_tasks: set[asyncio.Task] = set()


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
    try:
        body = (await request.body()).decode("utf-8")
    except ClientDisconnect:
        # The webhook client (LiveKit Cloud, via ngrok in dev) hung up
        # before the body was fully read. Nothing to verify or dispatch;
        # ack quietly instead of surfacing an ASGI 500 traceback.
        log.info("livekit_webhook reject reason=client_disconnect")
        return

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

    # The detector worker is the per-room subscriber/reconciler. It may
    # be absent (LiveKit disabled/mis-configured) — in that case we
    # still write the authoritative `cameras.online` transitions below
    # so the column never gets stuck, and just skip the worker calls.
    worker = getattr(request.app.state, "detector_worker", None)
    if worker is None:
        log.warning(
            "livekit_webhook event=%s with no detector_worker in app.state "
            "— applying online-state transition only",
            event.event,
        )

    room_name = event.room.name if event.room else ""
    participant_identity = event.participant.identity if event.participant else ""

    # Only OUR rooms are interesting to the detector; anything else is
    # accepted-but-ignored. Both shapes count: `account_<caregiver_id>`
    # for sensors, and `patient_<patient_id>` for wall phones paired
    # before the account-room change. Checking only the old prefix meant
    # every bridge sensor's join and leave was dropped on the floor —
    # no RoomWorker spawned, and `online` never written, so a streaming
    # camera read offline forever.
    if not livekit_rooms.is_room_name(room_name):
        return

    if event.event == "participant_joined" and participant_identity.startswith("cam_"):
        # A camera publisher arrived. Spawn/keep the RoomWorker; the
        # reconciler writes `cameras.online=true` ONCE frames actually
        # flow (two-signal liveness), so "joined but not streaming yet"
        # reads as connecting, not online. `signal_reconcile` collapses
        # the next poll to ~1s. No direct online write here — see
        # `camera_presence.py` trigger-point #1.
        if worker is not None:
            await worker.ensure_room_joined(room_name)
            await worker.signal_reconcile(room_name)
        # A rejoining publisher is what a camera restart looks like here, so
        # re-arm its setup self-test. Best-effort: never blocks the ack.
        await camera_presence.rearm_setup_test_by_identity(
            get_sessionmaker(), participant_identity,
        )
        log.info(
            "livekit_webhook participant_joined room=%s identity=%s -> joined",
            room_name, participant_identity,
        )
    elif event.event == "participant_joined" and participant_identity.startswith(
        "caregiver_"
    ):
        # Per-camera privacy self-heal (2026-07-27): the sever in
        # `livekit_privacy` fires only when privacy FLIPS, so a
        # caregiver who joins LATER could subscribe to a private
        # camera until publisher-side track permissions (camera-app /
        # gateway deploy trains) are live. Re-sever every already-
        # private camera in the room on EVERY caregiver join —
        # fire-and-forget + best-effort (the helper swallows all
        # errors), so the webhook ack is never blocked.
        task = asyncio.create_task(
            livekit_privacy.enforce_privacy_for_room(
                settings, get_sessionmaker(), room=room_name,
            ),
            name=f"privacy-resever-{room_name}",
        )
        _privacy_tasks.add(task)
        task.add_done_callback(_privacy_tasks.discard)
        log.info(
            "livekit_webhook participant_joined room=%s identity=%s "
            "-> privacy re-sever",
            room_name, participant_identity,
        )
    elif event.event == "participant_left" and participant_identity.startswith("cam_"):
        # AUTHORITATIVE offline signal. LiveKit Cloud confirmed the
        # publisher is gone, so write `online=false` NOW — this is the
        # one trigger that always fires (the reconciler is torn down
        # with the RoomWorker just below, and on a clean leave the SDK
        # pops the ingest before the reconciler's next poll). See
        # `camera_presence.py` trigger-point #2.
        await camera_presence.mark_offline_by_identity(
            get_sessionmaker(), participant_identity,
        )
        if worker is not None:
            await worker.ensure_room_left_if_empty(room_name)
        log.info(
            "livekit_webhook participant_left room=%s identity=%s -> offline",
            room_name, participant_identity,
        )
    elif event.event == "room_finished":
        # The whole room went away; every camera in it is offline.
        # See `camera_presence.py` trigger-point #3.
        flipped = await camera_presence.mark_room_offline(
            get_sessionmaker(), room_name,
        )
        if worker is not None:
            await worker.ensure_room_left_if_empty(room_name)
        log.info(
            "livekit_webhook room_finished room=%s -> %d offline",
            room_name, flipped,
        )
    # Other event types are no-ops.
