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
from sqlalchemy import delete, func, select

from src.api.deps import CameraAuth, CaregiverAuth, SessionDep, SettingsDep
from src.db.camera import Camera
from src.repos import camera_detector_repo, camera_repo, patient_repo
from src.schemas.camera import (
    CameraDetectorsResponse,
    CameraResponse,
    DesiredStateResponse,
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
    # `paired` is derived from `api_key_hash` (which we never expose
    # to clients). The ORM row has no `paired` attribute, so we can't
    # use plain `model_validate(c, from_attributes=True)` — that would
    # raise for the missing field. Build the dict explicitly instead.
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
        livekit_identity=c.livekit_identity,
        last_seen_at=c.last_seen_at,
        created_at=c.created_at,
        paired=c.api_key_hash is not None,
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
        # Default to DISCONNECTED on creation. A freshly-paired wall
        # phone hasn't actually joined LiveKit yet — the human still
        # has to tap Start on NemaraCam (or Connect from the caregiver
        # UI). Defaulting to `connected` made the caregiver-side tile
        # show a "Disconnect" button immediately, which lied: there
        # was nothing to disconnect from. Once Start is tapped on the
        # wall phone, its own `patchDesiredState("connected")` flips
        # this to "connected" and the caregiver UIs flip their button
        # label to "Disconnect" — matching reality.
        desired_state="disconnected",
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

    Three mutable fields, all user-facing toggles:
      - `disabled`: pauses LLM ticks AND hides the tile from the live
        grid. LiveKit stays connected. Instant on/off.
      - `ai_paused`: pauses LLM ticks only; tile + video stay visible.
        Lighter than `disabled` — for watching without paying for
        Gemini calls.
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
        ai_paused=body.ai_paused,
        desired_state=body.desired_state,
    )
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="camera_not_found",
        )
    await session.commit()
    # Use the helper that derives `paired` from api_key_hash — bare
    # model_validate would fail since the ORM row has no `paired`
    # attribute (it's a Pydantic-only computed field).
    return _to_response(cam)
