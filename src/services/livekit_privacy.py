"""Live SFU enforcement for privacy mode.

When a patient flips into privacy mode, any caregiver ALREADY connected
to their LiveKit room is still holding a token minted with
``can_subscribe=true``. The token-mint gate (``api/livekit.py``) cuts
them off on their next reconnect/re-mint, but until then their current
session keeps receiving media. This module closes that window: it walks
the room's participants and revokes subscribe + publish on every
``caregiver_*`` participant *now*, via the LiveKit server API.

The detector bot (``detector_worker_*``) and the camera (``cam_*``) are
left untouched, so backend analysis — and therefore the text insights —
keep flowing.

Best-effort by design: ``enforce_privacy_for_room`` never raises into
its caller, because the durable guarantee is the token-mint gate; this
live cut is just the instant-effect belt to go with those braces.
"""

from __future__ import annotations

import logging

from livekit import api as lkapi

log = logging.getLogger(__name__)

_CAREGIVER_PREFIX = "caregiver_"


async def sever_caregiver_media(lk: "lkapi.LiveKitAPI", room: str) -> int:
    """Revoke subscribe + publish for every ``caregiver_*`` participant
    currently in ``room``. Returns how many were severed. Leaves
    ``cam_*`` and ``detector_worker_*`` participants alone.

    ``lk`` is an already-constructed LiveKit server client (injected so
    this is unit-testable with a fake).
    """
    resp = await lk.room.list_participants(lkapi.ListParticipantsRequest(room=room))
    severed = 0
    for participant in resp.participants:
        if not participant.identity.startswith(_CAREGIVER_PREFIX):
            continue
        await lk.room.update_participant(
            lkapi.UpdateParticipantRequest(
                room=room,
                identity=participant.identity,
                permission=lkapi.ParticipantPermission(
                    can_subscribe=False,
                    can_publish=False,
                    can_publish_data=False,
                ),
            )
        )
        severed += 1
    return severed


async def enforce_privacy_for_room(settings, patient_id) -> None:
    """Open a LiveKit client and sever caregiver media in the patient's
    room (``patient_<id>``). No-op when LiveKit isn't configured.
    Best-effort: logs and swallows any error so the privacy toggle that
    triggered it always succeeds — the token-mint gate is the backstop.
    """
    if not (
        settings.livekit_url
        and settings.livekit_api_key
        and settings.livekit_api_secret
    ):
        return
    room = f"patient_{patient_id}"
    # Fully guarded: the LiveKit client construction AND aclose() are
    # inside the try, so NOTHING here can raise into patch_patient. A
    # transient LiveKit error must not 500 a PATCH that already committed
    # privacy_mode_enabled=True (that would desync the client's optimistic
    # toggle from the server). The token-mint gate is the durable backstop.
    lk = None
    try:
        lk = lkapi.LiveKitAPI(
            settings.livekit_url,
            settings.livekit_api_key,
            settings.livekit_api_secret,
        )
        n = await sever_caregiver_media(lk, room)
        log.info("privacy.sever room=%s severed_caregivers=%d", room, n)
    except Exception:
        log.exception(
            "privacy.sever failed room=%s — token-mint gate still holds", room
        )
    finally:
        if lk is not None:
            try:
                await lk.aclose()
            except Exception:
                log.exception("privacy.sever aclose failed room=%s", room)
