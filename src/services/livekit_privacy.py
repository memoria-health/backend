"""Live SFU severance for per-camera privacy (room-derived, 2026-07-27).

When a camera's ``privacy_enabled`` flips to true (room classifier
finalization, internal reclassify override), any caregiver ALREADY
connected to the patient's LiveKit room is still subscribed to that
camera's tracks. The durable cut is PUBLISHER-side track subscription
permissions (the camera app / gateway restrict their tracks to
``detector_worker_*`` identities when the desired-state / roster poll
reports ``privacy_enabled=true`` — SFU-enforced). This module is the
instant-effect defense-in-depth layer to go with it: it walks the room
and force-unsubscribes every ``caregiver_*`` participant from THAT
camera's tracks via the server API's ``UpdateSubscriptions``.

``UpdateSubscriptions`` alone is BEST-EFFORT — a tampered client can
re-subscribe until the publisher-side permissions land on its next poll
(~seconds). That ordering is why this call never gates anything: it
shrinks the window, the publisher permission closes it.

The detector bot (``detector_worker_*``) and cameras (``cam_*``) are
never touched, so backend analysis — and therefore text insights —
keep flowing. Other cameras' tracks are also untouched: privacy is
per-camera now, not per-patient.

Best-effort by design: ``enforce_privacy_for_camera`` never raises into
its caller — a transient LiveKit error must not fail the classifier
finalization / reclassify request that triggered it.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livekit import api as lkapi
from sqlalchemy import select

from src.db.camera import Camera

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker

log = logging.getLogger(__name__)

_CAREGIVER_PREFIX = "caregiver_"


async def sever_caregiver_media(
    lk: lkapi.LiveKitAPI, room: str, camera_identity: str,
) -> int:
    """Force-unsubscribe every ``caregiver_*`` participant in ``room``
    from the tracks published by ``camera_identity`` (a ``cam_*``
    participant). Returns how many caregivers were unsubscribed.

    Leaves ``cam_*`` and ``detector_worker_*`` participants alone, and
    other cameras' tracks alone — this is a per-camera cut, not the old
    patient-wide permission revoke.

    ``lk`` is an already-constructed LiveKit server client (injected so
    this is unit-testable with a fake).
    """
    resp = await lk.room.list_participants(lkapi.ListParticipantsRequest(room=room))
    publisher = next(
        (p for p in resp.participants if p.identity == camera_identity), None,
    )
    if publisher is None:
        # Camera not currently publishing — nothing to unsubscribe from.
        # New subscriptions are prevented publisher-side once it joins.
        return 0
    track_sids = [t.sid for t in publisher.tracks]
    if not track_sids:
        return 0
    severed = 0
    for participant in resp.participants:
        if not participant.identity.startswith(_CAREGIVER_PREFIX):
            continue
        await lk.room.update_subscriptions(
            lkapi.UpdateSubscriptionsRequest(
                room=room,
                identity=participant.identity,
                track_sids=track_sids,
                subscribe=False,
            )
        )
        severed += 1
    return severed


async def enforce_privacy_for_camera(
    settings, *, room: str | None, camera_identity: str | None,
) -> None:
    """Open a LiveKit client and sever caregiver subscriptions to one
    camera's tracks. No-op when LiveKit isn't configured or the camera
    has no LiveKit wiring yet (unpaired rows). Best-effort: logs and
    swallows any error so the privacy flip that triggered it always
    succeeds — publisher-side track permissions are the backstop.
    """
    if not (
        settings.livekit_url
        and settings.livekit_api_key
        and settings.livekit_api_secret
    ):
        return
    if not room or not camera_identity:
        return
    # Fully guarded: the LiveKit client construction AND aclose() are
    # inside the try, so NOTHING here can raise into the caller. A
    # transient LiveKit error must not fail a classifier finalization /
    # reclassify request that already committed privacy_enabled=true.
    lk = None
    try:
        lk = lkapi.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        n = await sever_caregiver_media(lk, room, camera_identity)
        log.info(
            "privacy.sever room=%s camera=%s severed_caregivers=%d",
            room, camera_identity, n,
        )
    except Exception:
        log.exception(
            "privacy.sever failed room=%s camera=%s — publisher-side track "
            "permissions still hold",
            room, camera_identity,
        )
    finally:
        if lk is not None:
            try:
                await lk.aclose()
            except Exception:
                log.exception("privacy.sever aclose failed room=%s", room)


async def enforce_privacy_for_room(
    settings, sessionmaker: async_sessionmaker, *, room: str,
) -> None:
    """Re-sever every ALREADY-private camera in ``room`` — the
    caregiver-JOIN self-heal (2026-07-27 review fix).

    `enforce_privacy_for_camera` fires only when privacy FLIPS. Until
    publisher-side track permissions ship in the camera-app/gateway
    deploy trains, a caregiver who joins AFTER the flip can still
    subscribe to a private camera's tracks. The livekit_webhooks
    ``participant_joined`` handler calls this for every ``caregiver_*``
    join, so the backend cut is re-applied on every join regardless of
    publisher version.

    Best-effort like everything in this module: the camera lookup and
    each per-camera sever swallow errors — a webhook ack must never
    fail on a transient DB/LiveKit error.
    """
    try:
        async with sessionmaker() as session:
            result = await session.execute(
                select(Camera.livekit_identity).where(
                    Camera.livekit_room == room,
                    Camera.privacy_enabled.is_(True),
                    Camera.livekit_identity.is_not(None),
                )
            )
            identities = [row[0] for row in result]
    except Exception:
        log.exception(
            "privacy.room_resever camera lookup failed room=%s — skipping "
            "(publisher-side track permissions still hold)",
            room,
        )
        return
    for identity in identities:
        # Never raises (fully guarded internally).
        await enforce_privacy_for_camera(
            settings, room=room, camera_identity=identity,
        )
