"""Detector API surface.

- GET    /v1/patients/{id}/detectors  — list.
- POST   /v1/patients/{id}/detectors  — create a custom detector.
- PATCH  /v1/detectors/{id}           — update (is_on, sensitivity, …).
- DELETE /v1/detectors/{id}           — remove.

All endpoints require CaregiverAuth and 404 (not 403) for cross-account
attempts.
"""

import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import update

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.db.detector import Detector
from src.db.session import get_sessionmaker
from src.repos import camera_detector_repo, detector_repo, patient_repo
from src.schemas.detector import (
    CreateDetectorRequest,
    DetectorResponse,
    DetectorShape,
    InferDetectorRequest,
    InferDetectorResponse,
    ScheduleConfig,
    UpdateDetectorRequest,
)
from src.services.detector_compiler import (
    compile_custom_detector,
    infer_detector_config,
)
from src.settings import Settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["detector"])


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
