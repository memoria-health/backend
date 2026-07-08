"""Detector API surface.

- GET    /v1/patients/{id}/detectors  — list.
- POST   /v1/patients/{id}/detectors  — create a custom detector.
- PATCH  /v1/detectors/{id}           — update (is_on, sensitivity, …).
- DELETE /v1/detectors/{id}           — remove.

All endpoints require CaregiverAuth and 404 (not 403) for cross-account
attempts.
"""

import asyncio
import base64
import logging
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import update

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.repos import camera_detector_repo, detector_repo, patient_repo
from src.schemas.detector import (
    AddFromLibraryRequest,
    CreateDetectorRequest,
    DetectorLibraryEntry,
    DetectorResponse,
    DetectorShape,
    InferDetectorRequest,
    InferDetectorResponse,
    ScheduleConfig,
    UpdateDetectorRequest,
)
from src.services.detection import catalog
from src.services.detection.door_detection import detect_doors
from src.services.detector_compiler import (
    compile_custom_detector,
    infer_detector_config,
)
from src.settings import Settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["detector"])


# ── Wandering exit-door picker ──────────────────────────────────────
class DetectDoorsRequest(BaseModel):
    image_base64: str


class DoorBox(BaseModel):
    box: list[float]   # [x, y, w, h] as fractions of the frame (0-1)
    label: str


class DetectDoorsResponse(BaseModel):
    doors: list[DoorBox]


class SetWanderingDoorRequest(BaseModel):
    # The chosen exit door, normalized [x, y, w, h] (0-1). None only when
    # no_exit_door is true.
    box: list[float] | None = None
    label: str | None = None
    # "vlm" = tapped a highlighted box; "manual" = tap-override elsewhere.
    source: str = "vlm"
    # True => no door leads outside; the wandering detector is turned OFF.
    no_exit_door: bool = False


@router.post(
    "/detect-doors",
    response_model=DetectDoorsResponse,
    status_code=status.HTTP_200_OK,
)
async def detect_doors_endpoint(
    body: DetectDoorsRequest,
    auth: CaregiverAuth,
    settings: SettingsDep,
) -> DetectDoorsResponse:
    """One-shot: given a camera frame (base64 JPEG), return the doorways the app
    highlights for the Wandering exit-door picker. The caregiver taps one (or
    taps elsewhere to override); the choice is saved via `set-wandering-door`."""
    try:
        image = base64.b64decode(body.image_base64, validate=True)
    except Exception:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid image_base64",
        )
    doors = await detect_doors(image, settings=settings)
    return DetectDoorsResponse(doors=[DoorBox(**d) for d in doors])


@router.post(
    "/patients/{patient_id}/wandering-door",
    response_model=DetectorResponse,
)
async def set_wandering_door(
    patient_id: UUID,
    body: SetWanderingDoorRequest,
    auth: CaregiverAuth,
    session: SessionDep,
) -> DetectorResponse:
    """Save the caregiver's exit-door choice for a patient's wandering detector.

    Regenerates the wandering `prompt_fragment` so the per-tick VLM only counts
    leaving THROUGH the chosen door as wandering. `no_exit_door` turns the
    detector off (nowhere to wander to). 404 if the patient has no wandering
    detector configured.
    """
    det = await detector_repo.find_by_patient_identifier(
        session, patient_id, "wandering", caregiver_id=auth.id
    )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="wandering_detector_not_found",
        )

    if body.no_exit_door:
        # No door leads outside — record the choice and disable the detector.
        fields = {
            "door_config": {"no_exit_door": True},
            "is_on": False,
        }
    else:
        box = body.box or []
        if len(box) != 4 or not all(
            isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in box
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid door box — expected [x, y, w, h] in 0-1",
            )
        source = body.source if body.source in ("vlm", "manual") else "vlm"
        door_config = {
            "box": [float(v) for v in box],
            "label": (body.label or "").strip(),
            "source": source,
            "no_exit_door": False,
        }
        fields = {
            "door_config": door_config,
            "prompt_fragment": catalog.build_wandering_fragment(door_config),
            "is_on": True,
        }

    updated = await detector_repo.update_partial(
        session, det.id, caregiver_id=auth.id, fields=fields
    )
    await session.commit()
    return _to_response(updated)


def _to_response(d: Detector) -> DetectorResponse:
    return DetectorResponse.model_validate(d, from_attributes=True)


@router.get(
    "/patients/{patient_id}/detectors",
    response_model=list[DetectorResponse],
)
async def list_detectors(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[DetectorResponse]:
    rows = await detector_repo.list_for_patient(session, patient_id, caregiver_id=auth.id)
    return [_to_response(d) for d in rows]


@router.get(
    "/patients/{patient_id}/detectors/library",
    response_model=list[DetectorLibraryEntry],
)
async def list_detector_library(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[DetectorLibraryEntry]:
    """The detection library for THIS patient's cohort (child if under 18, else
    adult — plus the `both` entries), in catalog order (recommended first).

    `already_added` is true for entries the patient already has by identifier,
    so the picker can pre-check / disable them. Pure catalog read + one detector
    list; no LLM, no writes.
    """
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )
    existing = {
        d.identifier
        for d in await detector_repo.list_for_patient(
            session, patient_id, caregiver_id=auth.id,
        )
        if d.identifier
    }
    minor = catalog.is_minor(patient.date_of_birth)
    entries = catalog.for_cohort(minor)
    log.info(
        "[detectors] library patient=%s minor=%s -> %d entries (%d already added)",
        patient_id, minor, len(entries),
        sum(1 for e in entries if e.key in existing),
    )
    return [
        DetectorLibraryEntry(
            key=e.key,
            name=e.name,
            icon_name=e.icon_name,
            description=e.description,
            category=e.category,  # str -> DetectorCategory (pydantic coerces)
            cohort=e.cohort,
            recommended=e.recommended,
            already_added=e.key in existing,
        )
        for e in entries
    ]


@router.post(
    "/patients/{patient_id}/detectors/from-library",
    response_model=list[DetectorResponse],
    status_code=status.HTTP_201_CREATED,
)
async def add_detectors_from_library(
    patient_id: UUID,
    auth: CaregiverAuth,
    body: AddFromLibraryRequest,
    session: SessionDep,
) -> list[DetectorResponse]:
    """Instantiate one or more catalog detectors for the patient.

    Unlike custom detectors there is **NO compile step** — catalog entries
    already carry the runtime contract (identifier / prompt_fragment /
    schema_json), so the rows are immediately usable by the pipeline. They are
    `source="preset"`, so they auto-attach to newly-paired cameras exactly like
    the auto-seeded presets.

    - Unknown keys → 422 (fail loud; the client should only send catalog keys).
    - Keys the patient already has (by identifier) → skipped (idempotent).
    - Each new detector is auto-assigned to the patient's cameras by its
      `where_scope`. No-op if the patient has no cameras yet (onboarding); the
      detector attaches when the first camera is paired.
    """
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )

    requested = list(dict.fromkeys(body.keys))  # dedupe, preserve order
    unknown = [k for k in requested if catalog.by_key(k) is None]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"unknown_detector_keys:{','.join(unknown)}",
        )

    existing = {
        d.identifier
        for d in await detector_repo.list_for_patient(
            session, patient_id, caregiver_id=auth.id,
        )
        if d.identifier
    }

    now = datetime.now(timezone.utc)
    created: list[Detector] = []
    skipped: list[str] = []
    for i, key in enumerate(requested):
        if key in existing:
            skipped.append(key)
            continue
        e = catalog.by_key(key)
        if e is None:  # unreachable — guarded above; defensive, fail loud
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"unknown_detector_key:{key}",
            )
        ts = now + timedelta(microseconds=i)
        det = Detector(
            id=uuid4(),
            patient_id=patient_id,
            name=e.name,
            icon_name=e.icon_name,
            description=e.description,
            category=e.category,
            source="preset",
            shape=e.shape,
            where_scope=e.where_scope,
            when_scope=e.when_scope,
            sensitivity="standard",
            is_on=True,
            created_at=ts,
            updated_at=ts,
            custom_prompt=None,
            identifier=e.key,
            prompt_fragment=e.prompt_fragment,
            schema_json=e.schema_json,
            always_on=False,
            is_builtin=True,
            schedule_config=None,
        )
        await detector_repo.insert(session, det)
        room = None if e.where_scope == "ALL ROOMS" else e.where_scope
        assigned = await camera_detector_repo.assign_detector_to_cameras(
            session, det.id, patient_id, room=room,
        )
        log.info(
            "[detectors] from-library add key=%s detector=%s patient=%s assigned_cameras=%d",
            e.key, det.id, patient_id, assigned,
        )
        created.append(det)

    await session.commit()
    log.info(
        "[detectors] from-library patient=%s created=%d skipped_already_present=%s",
        patient_id, len(created), skipped,
    )
    return [_to_response(d) for d in created]


@router.post("/detectors/infer", response_model=InferDetectorResponse)
async def infer_detector(
    auth: CaregiverAuth,
    body: InferDetectorRequest,
    settings: SettingsDep,
) -> InferDetectorResponse:
    """Classify + structure a plain-English detector request for the
    create review screen. Synchronous (the user is waiting on "Generate"),
    ~2-4s. Returns shape / category / icon / sensitivity and, for
    absence detectors, the schedule window — with nulls where the model
    couldn't infer, so the UI highlights them for the user to fill.

    Auth-only (no patient scope) — inference touches no patient data,
    just the caregiver's text.
    """
    try:
        inferred = await infer_detector_config(
            name=body.name, prompt=body.prompt, settings=settings,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="detector_inference_unavailable",
        ) from e
    return InferDetectorResponse(
        shape=inferred.shape,            # type: ignore[arg-type]
        category=inferred.category,      # type: ignore[arg-type]
        icon_name=inferred.icon_name,
        sensitivity=inferred.sensitivity,  # type: ignore[arg-type]
        schedule_config=(
            ScheduleConfig(**inferred.schedule) if inferred.schedule else None
        ),
        episode_gap_seconds=inferred.episode_gap_seconds,
    )


@router.post(
    "/patients/{patient_id}/detectors",
    response_model=DetectorResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_detector(
    patient_id: UUID,
    auth: CaregiverAuth,
    body: CreateDetectorRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> DetectorResponse:
    """Create a custom detector. **Returns immediately**; compile runs
    in the background.

    Why async compile: the LLM call that turns the caregiver's
    natural-language prompt into the runtime contract
    (`identifier`, `prompt_fragment`, `schema_json`) takes 3–5s. Holding
    the HTTP request open that long makes the dashboard's "Create"
    modal feel stuck. Instead we INSERT the row with NULL identifier,
    return 201 right away (~50ms), and fire-and-forget the compile in a
    background task. The dashboard sees the new row via Realtime, shows
    a "Compiling…" badge while `identifier IS NULL`, and the badge
    drops when the background UPDATE lands.

    `registry.get_enabled_detectors_for_camera` already filters
    `identifier IS NOT NULL`, so a compiling detector is invisible to
    the tick pipeline until compile completes — no half-baked LLM
    calls.

    Custom detectors are NOT auto-assigned to cameras. They're opt-in
    per camera via `/cameras/[id]`. Preset detectors (auto-seeded on
    patient create) still run on all cameras by default.
    """
    # 404 if the parent patient isn't owned by the caller — never leak
    # patient existence.
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )

    now = datetime.now(timezone.utc)
    det = Detector(
        id=uuid4(),
        patient_id=patient_id,
        name=body.name,
        icon_name=body.icon_name,
        description=body.description,
        category=body.category.value,
        source="custom",
        shape=body.shape.value,
        where_scope=body.where_scope,
        when_scope=body.when_scope,
        sensitivity=body.sensitivity.value,
        is_on=True,
        created_at=now,
        updated_at=now,
        custom_prompt=body.custom_prompt,
        # Runtime-contract fields stay NULL until the background
        # compile finishes — registry filter keeps it out of ticks
        # in the interim.
        identifier=None,
        prompt_fragment=None,
        schema_json=None,
        always_on=False,
        is_builtin=False,
        # Null for reactive; the evaluation window for `scheduled`.
        schedule_config=(
            body.schedule_config.model_dump() if body.schedule_config else None
        ),
        # Null unless shape == 'state'; the rising-edge episode gap.
        episode_gap_seconds=body.episode_gap_seconds,
    )
    await detector_repo.insert(session, det)

    # Auto-assign based on the "Where" hint. Saves the caregiver from
    # opening /cameras/[id] after creating each detector — they
    # already told us which room(s) they want it on.
    #   - "ALL ROOMS" → every camera under the patient
    #   - "Kitchen"   → every camera whose room == "Kitchen"
    # If `where_scope` is a legacy string that doesn't match any
    # current camera room, no rows get inserted and the user can still
    # assign manually from the cameras page.
    if body.where_scope == "ALL ROOMS":
        await camera_detector_repo.assign_detector_to_cameras(
            session, det.id, patient_id, room=None,
        )
    else:
        await camera_detector_repo.assign_detector_to_cameras(
            session, det.id, patient_id, room=body.where_scope,
        )
    await session.commit()

    # Kick off the compile. asyncio.create_task lets the request return
    # before the LLM call completes — we don't await it. The background
    # task uses its own session (session-per-task pattern) since the
    # request's SessionDep will be closed by the time we'd write back.
    asyncio.create_task(
        _compile_and_update_detector(
            detector_id=det.id,
            name=body.name,
            description=body.description,
            user_text=body.custom_prompt,
            scheduled=body.shape == DetectorShape.scheduled,
            category=det.category,
            settings=settings,
        )
    )

    return _to_response(det)


async def _compile_and_update_detector(
    *,
    detector_id: UUID,
    name: str,
    description: str,
    user_text: str,
    scheduled: bool,
    category: str,
    settings: Settings,
) -> None:
    """Background task: run Gemini compile + UPDATE the detector row.

    Owns its own AsyncSession (the request's session is gone by now).
    If compile fails, the row stays with NULL `identifier` — the
    dashboard's "Compiling…" badge then becomes "Compile failed" so
    the user knows to retry (vs. silently never firing). Errors are
    logged but not re-raised — there's no caller to surface them to.
    """
    try:
        compiled = await compile_custom_detector(
            name=name, description=description,
            user_text=user_text, scheduled=scheduled, category=category,
            settings=settings,
        )
    except Exception as e:
        log.error(
            "detector_compile_background.failed detector_id=%s name=%r err=%s",
            detector_id, name, e,
        )
        return

    # Write back. Use a fresh session — the request's session is closed.
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        await session.execute(
            update(Detector)
            .where(Detector.id == detector_id)
            .values(
                identifier=compiled.identifier,
                prompt_fragment=compiled.prompt_fragment,
                schema_json=compiled.schema_json,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()
    log.info(
        "detector_compile_background.ok detector_id=%s identifier=%s",
        detector_id, compiled.identifier,
    )


@router.patch(
    "/detectors/{detector_id}",
    response_model=DetectorResponse,
)
async def patch_detector(
    detector_id: UUID,
    auth: CaregiverAuth,
    body: UpdateDetectorRequest,
    session: SessionDep,
) -> DetectorResponse:
    fields = body.model_dump(exclude_unset=True)
    # Enum → string for the ORM string columns.
    for k in ("category", "shape", "sensitivity"):
        if k in fields and fields[k] is not None:
            fields[k] = fields[k].value
    det = await detector_repo.update_partial(
        session, detector_id, caregiver_id=auth.id, fields=fields
    )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="detector_not_found",
        )
    await session.commit()
    return _to_response(det)


@router.delete(
    "/detectors/{detector_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_model=None,
)
async def delete_detector(
    detector_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> None:
    deleted = await detector_repo.delete_by_id(session, detector_id, caregiver_id=auth.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="detector_not_found",
        )
    await session.commit()
