"""Camera API surface.

- GET    /v1/patients/{id}/cameras                 — list cameras for a patient.
- POST   /v1/patients/{id}/cameras/pairing-code    — mint a 6-digit pairing code.
- DELETE /v1/cameras/{id}                          — caregiver-initiated removal.

Cameras are CREATED in two steps by the pairing flow:
  1. Caregiver calls /pairing-code → cameras row appears with
     pairing_code + pairing_expires_at + online=false.
  2. Wall phone calls /v1/cameras/pair (in `src/api/pairing.py`)
     → cameras row gets api_key_hash, pairing_code cleared.
"""

import asyncio
import logging
import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import delete, func, select

log = logging.getLogger(__name__)

from src.api import cameras_ingest as pi_ingest_module
from src.api.deps import CameraAuth, CaregiverAuth, SessionDep, SettingsDep
from src.services import room_classifier
from src.services.ingest.pi.auth import mint_ingest_token
from src.db.camera import Camera
from src.repos import (
    camera_detector_repo,
    camera_repo,
    patient_repo,
    pi_discovery_repo,
)
from src.schemas.camera import (
    CameraDetectorsResponse,
    CameraResponse,
    DesiredStateResponse,
    PiProvisionRequest,
    PiProvisionResponse,
    SetCameraDetectorsRequest,
    UpdateCameraRequest,
    UpdateMyDesiredStateRequest,
)
from src.schemas.pairing import (
    CreatePairingCodeRequest,
    CreatePairingCodeResponse,
)

router = APIRouter(prefix="/v1", tags=["camera"])


def _to_response(c: Camera) -> CameraResponse:
    # `paired` is derived from whichever auth-credential column is
    # populated for this camera's source_type. iPhone cameras use
    # api_key_hash (issued at pairing-code redemption); Pi cameras
    # skip that flow and instead carry ingest_token_hash, minted at
    # provision time. Both columns are server-private — clients only
    # see the boolean. The ORM row has no `paired` attribute so a
    # plain `model_validate(c, from_attributes=True)` would raise;
    # build the dict explicitly instead.
    paired = (
        c.ingest_token_hash is not None
        if c.source_type in ("pi", "gateway")
        else c.api_key_hash is not None
    )
    return CameraResponse(
        id=c.id,
        patient_id=c.patient_id,
        display_id=c.display_id,
        room=c.room,
        icon_name=c.icon_name,
        online=c.online,
        disabled=c.disabled,
        ai_paused=c.ai_paused,
        desired_state=c.desired_state,  # type: ignore[arg-type]
        source_type=c.source_type,  # type: ignore[arg-type]
        livekit_identity=c.livekit_identity,
        created_at=c.created_at,
        paired=paired,
        app_present=c.app_present,
        room_class=c.room_class,  # type: ignore[arg-type]
        room_class_confidence=c.room_class_confidence,
        room_classified_at=c.room_classified_at,
        privacy_enabled=c.privacy_enabled,
    )


@router.get(
    "/patients/{patient_id}/cameras",
    response_model=list[CameraResponse],
)
async def list_cameras(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[CameraResponse]:
    rows = await camera_repo.list_for_patient(session, patient_id, caregiver_id=auth.id)
    return [_to_response(c) for c in rows]


@router.post(
    "/patients/{patient_id}/cameras/pairing-code",
    response_model=CreatePairingCodeResponse,
    status_code=status.HTTP_200_OK,
)
async def create_pairing_code(
    patient_id: UUID,
    body: CreatePairingCodeRequest,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> CreatePairingCodeResponse:
    """Mint a 6-digit pairing code + insert a placeholder camera row.

    The wall phone exchanges the code via POST /v1/cameras/pair
    (`src/api/pairing.py`). Until that happens the row has
    `online=false` and `api_key_hash IS NULL`.

    Verifies caregiver ownership of the patient and rejects pairing
    code collisions (extremely unlikely with 1e6 codes × ~10 active
    windows, but defended against).
    """
    # Ownership check — RLS would also catch this, but explicit 404 is
    # the right wire shape for "this patient isn't yours".
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )

    requested = body.room.strip()

    # Sweep orphan pairing rows for this patient + room name before
    # checking duplicates. An "orphan" is a row with no api_key_hash:
    # `create_pairing_code` was called but `/v1/cameras/pair` never
    # completed it. Two ways orphans get created:
    #
    #   1. User taps "Generate code" in iOS, sees the code, then
    #      dismisses the sheet without entering it on NemaraCam.
    #   2. iOS double-fires `createPairingCode` on a single user
    #      action (SwiftUI `.task` re-runs when the parent view
    #      re-renders mid-flight — typically after the realtime
    #      INSERT for the row just created lands and refreshes the
    #      camera list). First call creates the row, second collides.
    #
    # Either way the right behavior is: a fresh pairing attempt
    # supersedes any incomplete previous attempt for the same name.
    # The old pairing_code is invalidated, but no one entered it
    # anyway. Real (api_key_hash IS NOT NULL) cameras still block.
    await session.execute(
        delete(Camera)
        .where(Camera.patient_id == patient_id)
        .where(func.lower(Camera.room) == requested.lower())
        .where(Camera.api_key_hash.is_(None))
    )

    # Reject ACTUAL duplicates — fully paired cameras with the same
    # room name. App-level check; no DB unique constraint on
    # (patient_id, room) yet, since adding one would need a migration
    # + cleanup of any historical duplicates. Case-insensitive.
    duplicate_stmt = (
        select(Camera.id)
        .where(Camera.patient_id == patient_id)
        .where(func.lower(Camera.room) == requested.lower())
        .limit(1)
    )
    existing_dup = (await session.execute(duplicate_stmt)).scalar_one_or_none()
    if existing_dup is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="camera_room_duplicate",
        )

    # Next display_id for this patient (1-based, dense).
    max_display_stmt = select(func.coalesce(func.max(Camera.display_id), 0)).where(
        Camera.patient_id == patient_id
    )
    next_display_id = (await session.execute(max_display_stmt)).scalar_one() + 1

    # Generate a 6-digit code that doesn't collide with any active
    # pairing window. Restrict to 100000–999999 so we never emit a
    # leading-zero code (UX: confusing to type / mis-paste). Retry up
    # to 5 times before giving up — collisions are vanishingly rare at
    # our scale (≤ a few active windows).
    code: str | None = None
    for _ in range(5):
        candidate = str(secrets.randbelow(900_000) + 100_000)
        existing = (
            await session.execute(
                select(Camera.id).where(Camera.pairing_code == candidate).limit(1)
            )
        ).scalar_one_or_none()
        if existing is None:
            code = candidate
            break
    if code is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="could_not_generate_pairing_code",
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=settings.pairing_code_ttl_seconds)
    camera_id = uuid4()
    short = camera_id.hex[:8]

    cam = Camera(
        id=camera_id,
        patient_id=patient_id,
        display_id=next_display_id,
        room=body.room,
        icon_name=body.icon_name,
        online=False,
        created_at=now,
        pairing_code=code,
        pairing_expires_at=expires_at,
        api_key_hash=None,
        livekit_identity=f"cam_{short}",
        livekit_room=f"patient_{patient_id}",
        state="open",
        no_people_count=0,
        state_changed_at=now,
        close_threshold=6,
        battery_percent=None,
        # Default to DISCONNECTED on creation. A freshly-paired wall
        # phone hasn't actually joined LiveKit yet, the human still
        # has to tap Start on NemaraCam (or Connect from the caregiver
        # UI). Defaulting to `connected` made the caregiver-side tile
        # show a "Disconnect" button immediately, which lied: there
        # was nothing to disconnect from. Once Start is tapped on the
        # wall phone, its own `patchDesiredState("connected")` flips
        # this to "connected" and the caregiver UIs flip their button
        # label to "Disconnect", matching reality.
        desired_state="disconnected",
        # The pairing flow only creates iPhone cameras. Pi cameras
        # come in through /v1/cameras/pi-provision (Phase 4) which
        # constructs its own Camera row. Setting source_type and
        # pi_stream_url explicitly here keeps the row's full state
        # grep-able and surfaces source_type drift if a future change
        # alters the Python-side default.
        source_type="iphone",
        pi_stream_url=None,
    )
    session.add(cam)
    await session.flush()  # Ensure the camera row is visible to the assignment query below.
    # Reconcile detector links from each detector's where_rooms: presets
    # attach per their scope (all-rooms presets "just work" everywhere),
    # room-scoped detectors attach only to matching rooms, and null-scoped
    # custom detectors stay opt-in per camera — the reconciler never
    # touches those (see sync_links_for_patient's docstring).
    await camera_detector_repo.sync_links_for_patient(session, patient_id)
    await session.commit()

    return CreatePairingCodeResponse(
        pairing_code=code,
        expires_at=expires_at,
        camera_id=camera_id,
    )


@router.post(
    "/patients/{patient_id}/cameras/pi-provision",
    response_model=PiProvisionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def provision_pi_camera(
    patient_id: UUID,
    body: PiProvisionRequest,
    auth: CaregiverAuth,
    session: SessionDep,
) -> PiProvisionResponse:
    """Create a Pi camera row.

    Two ways to specify which Pi this row points at:

      - discovery_id: pick from the announce list (normal customer
        flow). Backend resolves the URL from the discovery record
        and marks it claimed.
      - pi_stream_url (+ optional pi_audio_url): direct URL for
        scripted provisioning or admin use.

    The Pydantic validator on PiProvisionRequest enforces that
    exactly one of the two modes is set.

    Sets source_type='pi'. Same LiveKit identity / room convention
    as iPhone cameras so the detection pipeline (RoomWorker,
    CameraIngest) treats Pi cameras identically once they appear
    in the room. The row appears with `online=false` and
    `state='open'`. Once the Pi starts pushing frames and the relay
    publishes into LiveKit, the room reconciler flips `online=true`
    (two-signal rule: REST participant present + fresh frames).
    """
    # Caregiver must own the patient. RLS would also catch this but
    # an explicit 404 is the right wire shape for "this patient isn't
    # yours" (matching the iPhone pairing-code endpoint above).
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )

    # Resolve the URLs from whichever mode the caller used. The
    # Pydantic validator guarantees exactly one is set.
    discovery_id_to_claim: UUID | None = None
    if body.discovery_id is not None:
        discovery = await pi_discovery_repo.find_by_id(session, body.discovery_id)
        if discovery is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="discovery_not_found",
            )
        if discovery.claimed_at is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="discovery_already_claimed",
            )
        pi_stream_url = discovery.current_url
        pi_audio_url = discovery.current_audio_url
        discovery_id_to_claim = discovery.id
    else:
        # Direct-URL mode. The validator guarantees pi_stream_url is set.
        assert body.pi_stream_url is not None
        pi_stream_url = body.pi_stream_url
        pi_audio_url = body.pi_audio_url

    # Reject duplicate provisioning: a discovery is uniquely tied to
    # one physical Pi (via hardware MAC); if it has been claimed
    # before, the caregiver should delete the existing camera row
    # before re-pairing. For direct-URL mode (vestigial) we fall back
    # to the URL check for backwards compat with test fixtures.
    if discovery_id_to_claim is None and pi_stream_url is not None:
        existing_url_stmt = (
            select(Camera.id).where(Camera.pi_stream_url == pi_stream_url).limit(1)
        )
        existing_url = (await session.execute(existing_url_stmt)).scalar_one_or_none()
        if existing_url is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="pi_stream_url_already_registered",
            )

    # Next display_id for this patient, same as the iPhone path.
    max_display_stmt = select(func.coalesce(func.max(Camera.display_id), 0)).where(
        Camera.patient_id == patient_id
    )
    next_display_id = (await session.execute(max_display_stmt)).scalar_one() + 1

    now = datetime.now(timezone.utc)
    camera_id = uuid4()
    short = camera_id.hex[:8]
    livekit_identity = f"cam_{short}"
    livekit_room = f"patient_{patient_id}"

    # Mint the Pi's ingest bearer token. The hash lands on the camera
    # row (used by the push endpoint to authenticate); the plaintext
    # gets stashed on the discovery row (read once by the Pi on its
    # next announce, then nulled). For Mode B (no discovery), there
    # is no per-Pi channel to deliver the plaintext, so we surface it
    # in the provision response as the only place it can land.
    ingest_token_plaintext, ingest_token_hash = mint_ingest_token()

    cam = Camera(
        id=camera_id,
        patient_id=patient_id,
        display_id=next_display_id,
        room=body.room,
        icon_name=body.icon_name,
        online=False,
        created_at=now,
        # Pi rows skip the pairing-code / api_key dance used by the
        # iPhone path. Instead, the Pi authenticates with the
        # ingest_token_hash bearer token below when pushing frames.
        pairing_code=None,
        pairing_expires_at=None,
        api_key_hash=None,
        livekit_identity=livekit_identity,
        livekit_room=livekit_room,
        state="open",
        no_people_count=0,
        state_changed_at=now,
        close_threshold=6,
        battery_percent=None,
        desired_state="connected",
        source_type="pi",
        pi_stream_url=pi_stream_url,
        pi_audio_url=pi_audio_url,
        ingest_token_hash=ingest_token_hash,
        ingest_token_created_at=now,
    )
    session.add(cam)
    await session.flush()
    # Reconcile detector links from where_rooms, same convention as the
    # iPhone path so a freshly-provisioned Pi camera is
    # detector-equivalent to an iPhone camera in the same room.
    await camera_detector_repo.sync_links_for_patient(session, patient_id)
    # Mark the discovery claimed (if we got here via discovery mode)
    # so it disappears from the next /discovered listing, AND stash
    # the plaintext ingest token on the discovery row so the Pi can
    # pick it up on its next announce.
    if discovery_id_to_claim is not None:
        await pi_discovery_repo.claim(
            session,
            discovery_id=discovery_id_to_claim,
            camera_id=camera_id,
            pending_ingest_token=ingest_token_plaintext,
        )
    await session.commit()

    return PiProvisionResponse(
        camera_id=camera_id,
        patient_id=patient_id,
        livekit_identity=livekit_identity,
        livekit_room=livekit_room,
    )


@router.post(
    "/patients/{patient_id}/cameras/gateway-provision",
    status_code=status.HTTP_201_CREATED,
    response_model=None,
)
async def provision_gateway_camera(
    patient_id: UUID,
    body: CreatePairingCodeRequest,
    auth: CaregiverAuth,
    session: SessionDep,
) -> dict:
    """Register a gateway camera (a Tapo behind a mini-PC).

    Like the iPhone pairing-code flow and pi-provision, but for our direct-
    publish gateway: NO pairing code (the camera is dumb — the mini-PC pairs on
    its behalf) and NO pi_stream_url (the gateway publishes straight to LiveKit,
    so the backend relay never pulls anything). Creates a `source_type='pi'` row
    with an ingest token and RETURNS the plaintext token — the only place it
    lands — so the mini-PC's setup agent can write it into its own config.
    """
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )
    now = datetime.now(timezone.utc)
    camera_id = uuid4()
    short = camera_id.hex[:8]
    livekit_identity = f"cam_{short}"
    livekit_room = f"patient_{patient_id}"
    next_display_id = (await session.execute(
        select(func.coalesce(func.max(Camera.display_id), 0)).where(
            Camera.patient_id == patient_id)
    )).scalar_one() + 1
    ingest_token_plaintext, ingest_token_hash = mint_ingest_token()
    cam = Camera(
        id=camera_id,
        patient_id=patient_id,
        display_id=next_display_id,
        room=body.room,
        icon_name=body.icon_name,
        online=False,
        created_at=now,
        pairing_code=None,
        pairing_expires_at=None,
        api_key_hash=None,
        livekit_identity=livekit_identity,
        livekit_room=livekit_room,
        state="open",
        no_people_count=0,
        state_changed_at=now,
        close_threshold=6,
        battery_percent=None,
        desired_state="disconnected",
        source_type="gateway",
        pi_stream_url=None,
        pi_audio_url=None,
        ingest_token_hash=ingest_token_hash,
        ingest_token_created_at=now,
        app_present=True,
    )
    session.add(cam)
    await session.flush()
    # Reconcile detector links from where_rooms — same as the other
    # camera-create paths, and also repairs any pre-existing drift.
    await camera_detector_repo.sync_links_for_patient(session, patient_id)
    await session.commit()
    return {
        "camera_id": str(camera_id),
        "livekit_identity": livekit_identity,
        "livekit_room": livekit_room,
        "ingest_token": ingest_token_plaintext,
    }


@router.delete(
    "/cameras/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_camera(
    camera_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    request: Request,
) -> None:
    # Reset the matching pi_discovery's claim state BEFORE the camera
    # row goes away. The pi_discoveries.claimed_camera_id FK is
    # `ON DELETE SET NULL`, so without this the discovery survives the
    # cascade in a zombie state — claimed_camera_id=NULL but
    # claimed_at still set, which makes the row a paired-but-pointing-
    # at-nothing record. The /discovered endpoint filters on
    # `claimed_at IS NULL`, so the Pi can't be re-paired without
    # operator-level SQL surgery (which we just had to do at 17:03
    # PDT 2026-05-28 when the bedroom camera was deleted). Clearing
    # claimed_at and pending_ingest_token here lets the Pi appear in
    # /discovered immediately on the next caregiver-app refresh.
    # No-op when the camera was never a Pi (no matching discovery row).
    await pi_discovery_repo.unclaim_for_camera(session, camera_id=camera_id)
    # Capture patient_id BEFORE the row goes away — needed for the link
    # reconcile below.
    cam = await camera_repo.find_by_id(session, camera_id, caregiver_id=auth.id)
    deleted = await camera_repo.delete_by_id(session, camera_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera_not_found",
        )
    # The FK cascade already dropped this camera's cameras_detectors rows;
    # this reconcile is the drift-repair safety net for the patient's
    # remaining cameras (cheap, same transaction).
    if cam is not None:
        await camera_detector_repo.sync_links_for_patient(
            session, cam.patient_id,
        )
    await session.commit()
    # Best-effort in-process worker shutdown. If the camera was actively
    # streaming, this stops the detector tick loop within ~1s rather
    # than waiting for the CameraIngest self-destruct (P1) to fire on
    # its own ~30s later. Either path produces the same end state — no
    # zombie worker — but doing it here gives the caregiver instant
    # feedback ("deleted" = no more analyses for this camera).
    #
    # Failure handling per the user's choice ("force the DELETE, log a
    # warning"): we've ALREADY committed the DB delete, so even if the
    # worker stop fails or times out we return 204. The P1 self-destruct
    # acts as the safety net.
    worker = getattr(request.app.state, "detector_worker", None)
    if worker is not None:
        try:
            await asyncio.wait_for(
                worker.stop_worker_for_camera(camera_id, timeout=4.0),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            log.warning(
                "delete_camera worker_stop_timeout camera=%s — "
                "P1 self-destruct will reap the zombie within ~30s",
                camera_id,
            )
        except Exception:
            log.exception(
                "delete_camera worker_stop_failed camera=%s — "
                "P1 self-destruct will reap the zombie within ~30s",
                camera_id,
            )


@router.post(
    "/cameras/{camera_id}/pairing-code",
    response_model=CreatePairingCodeResponse,
    status_code=status.HTTP_200_OK,
)
async def regenerate_pairing_code(
    camera_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    settings: SettingsDep,
) -> CreatePairingCodeResponse:
    """Re-mint a 6-digit pairing code for an UNPAIRED camera row.

    Companion to `create_pairing_code`. Used by the caregiver UI's
    "Show pairing code" button on the camera detail page so a user
    who dismissed the pairing sheet (or whose code expired) can
    resume pairing without deleting + recreating the camera row.

    Behavior:
      - 404 if the camera doesn't exist or isn't owned by this
        caregiver's patient (via `find_by_id` caregiver-scope).
      - 409 if the camera is already paired (api_key_hash set) —
        there's nothing to re-mint a code for. The detail-page button
        is hidden in that state too; this is the belt-and-braces.
      - Otherwise: generate a fresh 6-digit code, refresh
        `pairing_expires_at` to now + TTL, return both. Any previously
        emitted code becomes invalid (it was, in practice, abandoned).
    """
    cam = await camera_repo.find_by_id(session, camera_id, caregiver_id=auth.id)
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="camera_not_found",
        )
    if cam.api_key_hash is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="camera_already_paired",
        )

    # Mint a fresh code (same retry-on-collision loop as
    # create_pairing_code). Exclude ourselves from the collision check
    # so re-minting on an already-active row isn't a false-positive
    # collision against its own currently-held code.
    code: str | None = None
    for _ in range(5):
        candidate = str(secrets.randbelow(900_000) + 100_000)
        existing = (
            await session.execute(
                select(Camera.id)
                .where(Camera.pairing_code == candidate)
                .where(Camera.id != camera_id)
                .limit(1)
            )
        ).scalar_one_or_none()
        if existing is None:
            code = candidate
            break
    if code is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="could_not_generate_pairing_code",
        )

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(seconds=settings.pairing_code_ttl_seconds)
    cam.pairing_code = code
    cam.pairing_expires_at = expires_at
    await session.commit()

    return CreatePairingCodeResponse(
        pairing_code=code,
        expires_at=expires_at,
        camera_id=camera_id,
    )


def _reset_camera_to_unpaired(cam: Camera) -> None:
    """Transition a camera from paired → UNPAIRED in place, keeping the
    row (and its room / detectors / event history).

    Clearing `api_key_hash` is the operative move: the wire `paired` flag
    is derived from it, so this flips `paired` to False, which (a) re-opens
    the "Show pairing code" / re-pair path in the apps (gated on
    `!paired`), and (b) invalidates the old sensor's key so its next poll
    401s and it bounces back to its own pairing screen. The connection /
    identity / state-machine fields are reset so a fresh pairing starts
    clean; caregiver preferences (disabled / ai_paused) and the room/icon
    are intentionally preserved.
    """
    cam.api_key_hash = None
    cam.pairing_code = None
    cam.pairing_expires_at = None
    # Keep the LiveKit identity/room — they're deterministic per camera
    # (cam_<id> / patient_<patient_id>) and the re-pair claim does NOT
    # rewrite them in the DB. Nulling them here left livekit_room=None, so
    # the next Start's token request was rejected (camera_room_mismatch,
    # cam_room=None). Restore the canonical values so re-pair → Start works.
    cam.livekit_identity = f"cam_{cam.id.hex[:8]}"
    cam.livekit_room = f"patient_{cam.patient_id}"
    cam.online = False
    # Start the re-paired camera DISCONNECTED, not connected. A fresh re-pair
    # isn't streaming yet, so the apps must show "Reconnect" (an action) — not
    # "Disconnect" / "connecting…", which implies it's already live and forces
    # the caregiver to disconnect something that was never connected. The
    # sensor honours this on its desired-state poll and stays idle until the
    # caregiver explicitly reconnects.
    cam.desired_state = "disconnected"
    cam.state = "open"
    cam.no_people_count = 0
    cam.battery_percent = None
    cam.app_present = True  # reset heartbeat flag; moot while unpaired
    # Re-pair → re-classify (room-privacy plan, escape hatch ⚑3): the
    # phone may be re-mounted in a different room, so the old verdict is
    # void. Clearing room_classified_at makes the classifier re-run on
    # the next frames; room_class + privacy_enabled are KEPT until the
    # new verdict lands (fail-closed — a bathroom stays private through
    # the re-pair window rather than briefly unlocking).
    cam.room_class_confidence = None
    cam.room_classified_at = None
    room_classifier.reset_attempts(cam.id)


async def _stop_worker_best_effort(request: Request, camera_id: UUID) -> None:
    """Stop the detector worker for a camera, best-effort (same 5s budget
    and P1-self-destruct safety net as the delete paths)."""
    worker = getattr(request.app.state, "detector_worker", None)
    if worker is None:
        return
    try:
        await asyncio.wait_for(
            worker.stop_worker_for_camera(camera_id, timeout=4.0), timeout=5.0,
        )
    except asyncio.TimeoutError:
        log.warning(
            "unpair worker_stop_timeout camera=%s — P1 self-destruct reaps within ~30s",
            camera_id,
        )
    except Exception:  # noqa: BLE001
        log.exception("unpair worker_stop_failed camera=%s", camera_id)


# NOTE: `/cameras/me/unpair` MUST be declared before `/cameras/{camera_id}/unpair`
# or Starlette would match the literal "me" as a camera_id path param (→ 422).
@router.post(
    "/cameras/me/unpair",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def unpair_self(
    auth: CameraAuth,
    session: SessionDep,
    request: Request,
) -> None:
    """Wall-phone self-unpair (soft). The sensor clears its OWN pairing so
    the camera row STAYS but goes UNPAIRED + re-pairable, then wipes its
    local creds. Soft counterpart to DELETE /v1/cameras/me: same auth, but
    keeps the row + its history/detectors so a caregiver (or this phone)
    can re-pair to it later from the app.
    """
    cam = await session.get(Camera, auth.id)
    if cam is None:
        return  # already gone — nothing to unpair
    _reset_camera_to_unpaired(cam)
    await session.commit()
    await _stop_worker_best_effort(request, auth.id)


@router.post(
    "/cameras/{camera_id}/unpair",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def unpair_camera(
    camera_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
    request: Request,
) -> None:
    """Caregiver-initiated soft unpair: return the camera to the UNPAIRED
    state without deleting it. Room, detectors, and event history are
    preserved, and the camera becomes re-pairable via the existing "Show
    pairing code" flow. Any currently-paired sensor loses its key and
    returns to its pairing screen on the next poll.

    Distinct from DELETE /v1/cameras/{id} (removes the row) and from
    "APP CLOSED"/offline (still paired, just not streaming). 404 if the
    camera isn't owned by this caregiver's patient.
    """
    cam = await camera_repo.find_by_id(session, camera_id, caregiver_id=auth.id)
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="camera_not_found",
        )
    _reset_camera_to_unpaired(cam)
    await session.commit()
    await _stop_worker_best_effort(request, camera_id)


@router.get(
    "/cameras/me/desired-state",
    response_model=DesiredStateResponse,
)
async def get_my_desired_state(
    auth: CameraAuth,
    request: Request,
) -> DesiredStateResponse:
    """Polled by the wall phone every 1s. `CameraAuth` already
    resolves the bearer api_key against `cameras.api_key_hash` and
    returns the Camera row — we just echo `desired_state` back. No
    DB write here.

    Path is literal `/me/` not `/{camera_id}/` so the camera doesn't
    have to know its own UUID; the api_key uniquely identifies it.
    Path collision with `/cameras/{camera_id}/*` is impossible
    because every existing `{camera_id}/...` route ends in a different
    literal suffix (detectors), and `/cameras/{camera_id}` (2-segment)
    can't match this 3-segment path.

    Type-cast at the boundary: `auth.desired_state` is a `str` from
    the ORM. Pydantic Literal coerces — if the DB row somehow holds
    anything other than the two valid values (impossible because of
    the CHECK constraint, but if a migration ever drifts), Pydantic
    raises 500 here. Fail-loud aligned.
    """
    # App-present heartbeat. This 1s poll IS the "Memoria Sensor app is
    # open" signal — record it in the in-memory PresenceCache so the
    # PresenceSweeper can tell a closed app from a reconnectable
    # disconnect. Pure in-memory: NO DB write on this hot path (a
    # per-poll write is exactly what exhausted the pool before).
    # getattr-guarded so a missing cache (presence disabled, or a
    # startup race before lifespan wired it) can never 500 the poll.
    cache = getattr(request.app.state, "presence_cache", None)
    if cache is not None:
        cache.touch(auth.id)
    return DesiredStateResponse(
        desired_state=auth.desired_state,  # type: ignore[arg-type]
        polled_at=datetime.now(timezone.utc),
        room=auth.room,
        # Room-derived privacy: the publisher restricts its tracks'
        # subscription permissions when this flips true (picked up
        # within ~1s via this poll).
        privacy_enabled=auth.privacy_enabled,
    )


@router.delete(
    "/cameras/me",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_self(
    auth: CameraAuth,
    session: SessionDep,
    request: Request,
) -> None:
    """Wall-phone self-delete on user-initiated unpair.

    The wall phone calls this immediately before wiping its local
    Keychain so the cameras row doesn't orphan with `api_key_hash`
    still set. Without this, the caregiver dashboard accumulates one
    "paired but silent" row per unpair cycle (which has been the
    actual situation we kept hitting locally).

    Mirror of the caregiver-facing `delete_camera` (route id-based,
    `CaregiverAuth`) — same DB-then-worker order, same best-effort
    worker shutdown on a 5s budget, same 204 even if the worker
    stop misses (the P1 self-destruct cleans up zombie ticks).

    No-op behavior: returns 204 if the row is already gone (e.g.
    caregiver deleted it concurrently). The api_key bearer would
    have already raised 404 inside `camera_auth` in that case, so
    by the time we land here the row definitely existed at request
    time.
    """
    camera_id = auth.id
    # Symmetric with delete_camera: reset the matching pi_discovery
    # claim state before the FK cascade strands it. No-op for wall
    # phone (no discovery row), but kept here so the contract is
    # uniform across both delete paths and we don't recreate the
    # zombie-discovery bug if the wall-phone DELETE ever runs against
    # a Pi-typed row by accident.
    await pi_discovery_repo.unclaim_for_camera(session, camera_id=camera_id)
    await session.execute(delete(Camera).where(Camera.id == camera_id))
    await session.commit()
    worker = getattr(request.app.state, "detector_worker", None)
    if worker is not None:
        try:
            await asyncio.wait_for(
                worker.stop_worker_for_camera(camera_id, timeout=4.0),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            log.warning(
                "delete_self worker_stop_timeout camera=%s, "
                "P1 self-destruct will reap the zombie within ~30s",
                camera_id,
            )
        except Exception:
            log.exception(
                "delete_self worker_stop_failed camera=%s, "
                "P1 self-destruct will reap the zombie within ~30s",
                camera_id,
            )


@router.patch(
    "/cameras/me/desired-state",
    response_model=DesiredStateResponse,
)
async def update_my_desired_state(
    body: UpdateMyDesiredStateRequest,
    auth: CameraAuth,
    session: SessionDep,
) -> DesiredStateResponse:
    """Wall phone updates its own `desired_state`. Mirrors what
    PATCH /v1/cameras/{id} does for the caregiver, but scoped to the
    caller's api_key so the phone doesn't need its own UUID. Called
    on Start/Stop taps so the web dashboard's tile menu stays in
    sync with reality — otherwise the menu shows stale buttons and
    the next caregiver click acts on out-of-date intent.

    `auth` is the Camera ORM row loaded by `camera_auth`; it's already
    attached to this request's session, so mutating + commit persists.
    """
    auth.desired_state = body.desired_state
    await session.commit()
    return DesiredStateResponse(
        desired_state=body.desired_state,
        polled_at=datetime.now(timezone.utc),
        room=auth.room,
        privacy_enabled=auth.privacy_enabled,
    )


@router.get(
    "/cameras/{camera_id}/detectors",
    response_model=CameraDetectorsResponse,
)
async def list_camera_detectors(
    camera_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> CameraDetectorsResponse:
    """Return the detector IDs assigned to this camera. Empty if the
    camera doesn't belong to the caller — same 404-via-empty pattern
    used elsewhere to avoid leaking camera existence."""
    cam = await camera_repo.find_by_id(session, camera_id, caregiver_id=auth.id)
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="camera_not_found",
        )
    ids = await camera_detector_repo.list_detector_ids_for_camera(
        session, camera_id, caregiver_id=auth.id,
    )
    return CameraDetectorsResponse(detector_ids=list(ids))


@router.put(
    "/cameras/{camera_id}/detectors",
    response_model=CameraDetectorsResponse,
)
async def set_camera_detectors(
    camera_id: UUID,
    body: SetCameraDetectorsRequest,
    auth: CaregiverAuth,
    session: SessionDep,
) -> CameraDetectorsResponse:
    """Replace the camera's detector assignment set.

    Validates ownership + that every detector belongs to the camera's
    patient. Takes effect on the next tick — the registry reads the
    join table per call, no in-process cache to invalidate.

    Reconciliation with where_rooms (2026-07-28): links are DERIVED
    from each detector's where_rooms by sync_links_for_patient, so a
    picker edit must write back through where_rooms or the next sync
    would revert it. After the replace, every detector whose
    membership changed here gets where_rooms rewritten from its
    resulting link set (NULL when it spans every camera). Null-scoped
    CUSTOM detectors are exempt — their links are opt-in-authoritative
    and sync never touches them. Consequence of the room-derived
    model: two cameras sharing one room name cannot diverge for a
    room-scoped detector; the closing sync aligns them.
    """
    prior = set(
        await camera_detector_repo.list_detector_ids_for_camera(
            session, camera_id, caregiver_id=auth.id,
        )
    )
    try:
        final = await camera_detector_repo.replace_assignments_for_camera(
            session,
            camera_id,
            body.detector_ids,
            caregiver_id=auth.id,
        )
    except ValueError as e:
        msg = str(e)
        # The two errors raise_value_errors can throw map to different statuses.
        if "not owned" in msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="camera_not_found",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid_detector_ids",
        ) from e
    changed = prior.symmetric_difference(final)
    if changed:
        cam = await camera_repo.find_by_id(
            session, camera_id, caregiver_id=auth.id,
        )
        if cam is not None:
            await camera_detector_repo.rewrite_where_rooms_from_links(
                session, cam.patient_id, detector_ids=changed,
            )
            await camera_detector_repo.sync_links_for_patient(
                session, cam.patient_id,
            )
    await session.commit()
    return CameraDetectorsResponse(detector_ids=list(final))


@router.patch(
    "/cameras/{camera_id}",
    response_model=CameraResponse,
)
async def update_camera(
    camera_id: UUID,
    body: UpdateCameraRequest,
    auth: CaregiverAuth,
    session: SessionDep,
) -> CameraResponse:
    """Partial-update a caregiver-owned camera.

    Four mutable fields, all user-facing:
      - `disabled`: pauses LLM ticks AND hides the tile from the live
        grid. LiveKit stays connected. Instant on/off.
      - `ai_paused`: pauses LLM ticks only; tile + video stay visible.
        Lighter than `disabled` — for watching without paying for
        Gemini calls.
      - `desired_state`: 'disconnected' makes the wall phone leave
        the LiveKit room entirely. Phone polls
        /v1/cameras/me/desired-state every 1s; reconnect latency
        ~1-2s. Full battery + privacy savings.
      - `room`: cosmetic rename. Side effect: clears the camera's room
        classification so the room classifier re-runs (privacy stays as
        it was until the new verdict lands — fail-closed).

    The privacy fields (room_class / privacy_enabled / …) are NEVER
    settable here — `extra="forbid"` on the body model 422s them.
    """
    renamed = body.room is not None
    if renamed:
        # Same case-insensitive per-patient uniqueness rule the pairing
        # flow enforces at creation — two cameras named "Kitchen" would
        # break the room-name-keyed flows (orphan sweep, detector
        # where_rooms). Excludes self so a case-only rename is allowed.
        cam_for_dup = await camera_repo.find_by_id(
            session, camera_id, caregiver_id=auth.id,
        )
        if cam_for_dup is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="camera_not_found",
            )
        requested = body.room.strip()
        dup_stmt = (
            select(Camera.id)
            .where(Camera.patient_id == cam_for_dup.patient_id)
            .where(func.lower(Camera.room) == requested.lower())
            .where(Camera.id != camera_id)
            .limit(1)
        )
        if (await session.execute(dup_stmt)).scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="camera_room_duplicate",
            )
    cam = await camera_repo.update_partial(
        session,
        camera_id,
        caregiver_id=auth.id,
        disabled=body.disabled,
        ai_paused=body.ai_paused,
        desired_state=body.desired_state,
        room=body.room,
    )
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera_not_found",
        )
    if renamed:
        # A room rename changes which detectors' where_rooms this camera
        # falls under — re-derive the cameras_detectors links in the same
        # transaction (gains AND losses).
        await camera_detector_repo.sync_links_for_patient(
            session, cam.patient_id,
        )
    await session.commit()
    if renamed:
        # Restart the classifier's in-process attempt ledger so the
        # re-run begins at attempt 1 (the repo cleared the DB timestamp).
        room_classifier.reset_attempts(camera_id)
    # If the caregiver just disconnected a Pi camera, cancel the in-flight
    # HTTP push so the Pi sees EOF, reconnects, hits the desired_state
    # gate in /v1/cameras/me/frames, and switches into paused mode within
    # ~1 frame. Without this the Pi would keep streaming for the rest of
    # the existing push connection's lifetime (long-lived). No-op for
    # wall-phone rows (they have no active relay) and no-op when nothing
    # is currently pushing — both safe by construction in
    # cancel_active_relay.
    if (
        body.desired_state == "disconnected"
        and cam.source_type == "pi"
    ):
        await pi_ingest_module.cancel_active_relay(camera_id)
    # Use the helper that derives `paired` from api_key_hash — bare
    # model_validate would fail since the ORM row has no `paired`
    # attribute (it's a Pydantic-only computed field).
    return _to_response(cam)
