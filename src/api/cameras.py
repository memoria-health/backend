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

import secrets
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import func, select

from src.api.deps import CameraAuth, CaregiverAuth, SessionDep, SettingsDep
from src.db.camera import Camera
from src.repos import camera_detector_repo, camera_repo, patient_repo
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
    return CameraResponse.model_validate(c, from_attributes=True)


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
        last_seen_at=None,
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
        # Default to connected so a freshly-paired phone starts
        # streaming immediately. Server default would cover this too,
        # set it explicitly so the row's full state is grep-able from
        # this construction site.
        desired_state="connected",
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
    # Auto-assign only the PRESET detectors (Falls, Wandering, etc.) to
    # the new camera — caregivers expect those to "just work" on every
    # camera. Custom detectors stay opt-in: a caregiver who built a
    # `stove_on` detector for the kitchen doesn't expect it to start
    # running on the bedroom camera the moment a bedroom phone is paired.
    await camera_detector_repo.assign_preset_detectors_of_patient_to_camera(
        session, camera_id, patient_id,
    )
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
    """Create a Raspberry Pi camera row in one shot.

    Unlike the iPhone path, the Pi camera has no UI to display a
    pairing code and no device-side bearer token. The backend's
    pi_ingest worker (Phase 5) connects to LiveKit using the
    server's own LiveKit credentials, so the Pi device itself does
    not need authentication; the pi_stream_url is the entire
    contract between worker and Pi.

    Sets source_type='pi' and pi_stream_url=body.pi_stream_url. Same
    LiveKit identity / room convention as iPhone cameras so the
    detection pipeline (RoomWorker, CameraIngest) treats Pi cameras
    identically once they appear in the room.

    The Pi camera row appears with `online=false` and `state='open'`.
    The pi_ingest worker flips `online=true` after its first
    successful frame pull.
    """
    # Caregiver must own the patient. RLS would also catch this but
    # an explicit 404 is the right wire shape for "this patient isn't
    # yours" (matching the iPhone pairing-code endpoint above).
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )

    # Reject duplicate provisioning: a Pi URL is uniquely tied to one
    # physical device; if it is already registered, the caregiver
    # should delete the existing row rather than create a duplicate.
    existing_url_stmt = (
        select(Camera.id).where(Camera.pi_stream_url == body.pi_stream_url).limit(1)
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

    cam = Camera(
        id=camera_id,
        patient_id=patient_id,
        display_id=next_display_id,
        room=body.room,
        icon_name=body.icon_name,
        online=False,
        last_seen_at=None,
        created_at=now,
        # Pi rows skip the pairing-code / api_key dance: there is no
        # device-side bearer token. The pi_ingest worker authenticates
        # to LiveKit using server credentials.
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
        pi_stream_url=body.pi_stream_url,
        pi_audio_url=body.pi_audio_url,
    )
    session.add(cam)
    await session.flush()
    # Auto-assign preset detectors, same convention as the iPhone path
    # so a freshly-provisioned Pi camera is detector-equivalent to an
    # iPhone camera in the same room.
    await camera_detector_repo.assign_preset_detectors_of_patient_to_camera(
        session, camera_id, patient_id,
    )
    await session.commit()

    return PiProvisionResponse(
        camera_id=camera_id,
        patient_id=patient_id,
        pi_stream_url=body.pi_stream_url,
        pi_audio_url=body.pi_audio_url,
        livekit_identity=livekit_identity,
        livekit_room=livekit_room,
    )


@router.delete(
    "/cameras/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_camera(
    camera_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> None:
    deleted = await camera_repo.delete_by_id(session, camera_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera_not_found",
        )
    await session.commit()


@router.get(
    "/cameras/me/desired-state",
    response_model=DesiredStateResponse,
)
async def get_my_desired_state(
    auth: CameraAuth,
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
    return DesiredStateResponse(
        desired_state=auth.desired_state,  # type: ignore[arg-type]
        polled_at=datetime.now(timezone.utc),
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
    """
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

    Two mutable fields, both user-facing toggles:
      - `disabled`: pauses LLM ticks. LiveKit stays connected.
        Instant on/off, no battery savings.
      - `desired_state`: 'disconnected' makes the wall phone leave
        the LiveKit room entirely. Phone polls
        /v1/cameras/me/desired-state every 1s; reconnect latency
        ~1-2s. Full battery + privacy savings.
    """
    cam = await camera_repo.update_partial(
        session,
        camera_id,
        caregiver_id=auth.id,
        disabled=body.disabled,
        desired_state=body.desired_state,
    )
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera_not_found",
        )
    await session.commit()
    return CameraResponse.model_validate(cam, from_attributes=True)
