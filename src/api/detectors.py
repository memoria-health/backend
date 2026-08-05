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
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select, update

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.api.rate_limit import RateLimit, RateLimitPolicy
from src.db.camera import Camera
from src.db.camera_detector import CameraDetector
from src.db.detector import Detector
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.repos import camera_detector_repo, detector_repo, patient_repo
from src.schemas.detector import (
    AddFromLibraryRequest,
    AlertTrigger,
    CreateDetectorRequest,
    DetectorLibraryEntry,
    DetectorResponse,
    InferDetectorRequest,
    InferDetectorResponse,
    RefineDetectorRequest,
    ScheduleConfig,
    UpdateDetectorRequest,
)
from src.services.detection import catalog, catalog_v2
from src.services.detection.door_detection import detect_doors
from src.services.detection.scheduling import compiler_v2
from src.services.detector_compiler import (
    compile_custom_detector,
    infer_detector_config,
    summarize_detector_prompt,
)
from src.settings import Settings

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["detector"])

# Creating a detector compiles a plan through the VLM, and every tick it
# then schedules is a paid call. Detectors are user-creatable with no
# per-account cap, so a compromised account could drive spend without
# bound (audit M1). 30/minute is far above real use — a caregiver adds a
# handful, once — while capping automated creation.
CREATE_DETECTOR_RATE_LIMIT = RateLimitPolicy(
    name="detectors_create", limit=30, window_seconds=60,
)


# ── Wandering exit-door picker ──────────────────────────────────────
class DetectDoorsRequest(BaseModel):
    image_base64: str


class DoorBox(BaseModel):
    box: list[float]   # [x, y, w, h] as fractions of the frame (0-1)
    label: str


class DetectDoorsResponse(BaseModel):
    doors: list[DoorBox]


class WanderingDoorInput(BaseModel):
    # One chosen exit door, normalized [x, y, w, h] (0-1).
    box: list[float]
    label: str | None = None
    # "vlm" = tapped a highlighted box; "manual" = tap-override elsewhere.
    source: str = "vlm"


class SetWanderingDoorRequest(BaseModel):
    # The chosen exit door(s). Empty only when no_exit_door is true.
    doors: list[WanderingDoorInput] = []
    # True => no door leads outside on THIS camera; wandering is skipped here.
    no_exit_door: bool = False


class CameraWanderingDoor(BaseModel):
    camera_id: UUID
    # {doors:[...], no_exit_door} or null if this camera has no wandering config.
    door_config: dict | None = None


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
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="invalid image_base64",
        )
    doors = await detect_doors(image, settings=settings)
    return DetectDoorsResponse(doors=[DoorBox(**d) for d in doors])


async def _owned_camera(
    session: SessionDep, camera_id: UUID, *, caregiver_id: UUID,
) -> Camera | None:
    """Return the camera if the caregiver owns its patient, else None."""
    result = await session.execute(
        select(Camera)
        .join(Patient, Patient.id == Camera.patient_id)
        .where(Camera.id == camera_id, Patient.caregiver_id == caregiver_id)
    )
    return result.scalar_one_or_none()


def _normalize_doors(body: SetWanderingDoorRequest) -> dict:
    """Validate + shape the request into a `door_config` payload."""
    if body.no_exit_door:
        return {"doors": [], "no_exit_door": True}
    if not body.doors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="select at least one exit door, or set no_exit_door",
        )
    doors: list[dict] = []
    for d in body.doors:
        if len(d.box) != 4 or not all(
            isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in d.box
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="invalid door box — expected [x, y, w, h] in 0-1",
            )
        doors.append({
            "box": [float(v) for v in d.box],
            "label": (d.label or "").strip(),
            "source": d.source if d.source in ("vlm", "manual") else "vlm",
        })
    return {"doors": doors, "no_exit_door": False}


@router.post(
    "/cameras/{camera_id}/wandering-door",
    response_model=CameraWanderingDoor,
)
async def set_camera_wandering_door(
    camera_id: UUID,
    body: SetWanderingDoorRequest,
    auth: CaregiverAuth,
    session: SessionDep,
) -> CameraWanderingDoor:
    """Save the exit-door choice for a SPECIFIC camera (sensor). The exit door
    is per-sensor — each camera sees a different scene — so it's stored on the
    camera↔wandering-detector assignment and the runtime builds THIS camera's
    wandering prompt from it. Setting doors also assigns wandering to the camera;
    `no_exit_door` records that this camera watches no exit (wandering skipped
    here). 404 if the patient has no wandering detector.
    """
    cam = await _owned_camera(session, camera_id, caregiver_id=auth.id)
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="camera_not_found",
        )
    det = await detector_repo.find_by_patient_identifier(
        session, cam.patient_id, "wandering", caregiver_id=auth.id,
    )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="wandering_detector_not_found",
        )

    door_config = _normalize_doors(body)

    # Upsert the assignment carrying this camera's door_config. Setting doors
    # implicitly assigns wandering to the camera so it starts running here.
    cd = await session.get(
        CameraDetector, {"camera_id": camera_id, "detector_id": det.id},
    )
    if cd is None:
        cd = CameraDetector(
            camera_id=camera_id,
            detector_id=det.id,
            created_at=datetime.now(UTC),
            door_config=door_config,
        )
        session.add(cd)
    else:
        cd.door_config = door_config
    await session.commit()
    return CameraWanderingDoor(camera_id=camera_id, door_config=door_config)


@router.get(
    "/patients/{patient_id}/wandering-doors",
    response_model=list[CameraWanderingDoor],
)
async def list_wandering_doors(
    patient_id: UUID,
    auth: CaregiverAuth,
    session: SessionDep,
) -> list[CameraWanderingDoor]:
    """Per-camera wandering exit-door config for a patient — one entry per
    camera that has a wandering assignment, so the app can show each sensor's
    status. Cameras with no wandering config simply don't appear."""
    patient = await patient_repo.find_by_id(
        session, patient_id, caregiver_id=auth.id,
    )
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="patient_not_found",
        )
    det = await detector_repo.find_by_patient_identifier(
        session, patient_id, "wandering", caregiver_id=auth.id,
    )
    if det is None:
        return []
    result = await session.execute(
        select(CameraDetector.camera_id, CameraDetector.door_config)
        .join(Camera, Camera.id == CameraDetector.camera_id)
        .where(Camera.patient_id == patient_id, CameraDetector.detector_id == det.id)
    )
    return [
        CameraWanderingDoor(camera_id=cid, door_config=dc)
        for cid, dc in result.all()
    ]


# Fine-detail detectors — accuracy varies with distance and lighting,
# so the caregiver UI surfaces an expectations notice. Two signals so
# BOTH the catalog (key only, no bucket) and compiled detectors (bucket
# known) resolve through one policy:
#   - identifier set: the fine-detail presets (small motion + small
#     object). Also matches catalog keys, since key == identifier.
#   - recipe buckets: catches CUSTOM detectors compiled to fine-motion
#     or rapid whose identifier is custom_<hex>.
_CAVEAT_BUCKETS = frozenset({"fine_motion", "rapid"})
# Small-motion (seizure) / small-object (medication, stove) presets whose
# accuracy genuinely degrades with distance + lighting. NOT choking: it's a
# whole-person distress event (bucket="person", degradation_warning="none"),
# so the "reads small, fine details" notice misdescribes it — its caveat was
# manual, never a derived property.
_CAVEAT_IDENTIFIERS = frozenset({
    "seizure", "medication", "stove_left_on",
})


def _accuracy_caveat(identifier: str | None, bucket: str | None) -> bool:
    """Whether a detector should carry the accuracy-expectations notice.
    Pure (unit-tested); the single policy point for the caveat, shared
    by the detector response and the library catalog."""
    if bucket in _CAVEAT_BUCKETS:
        return True
    return identifier in _CAVEAT_IDENTIFIERS


def _to_response(d: Detector) -> DetectorResponse:
    resp = DetectorResponse.model_validate(d, from_attributes=True)
    return resp.model_copy(update={
        "accuracy_caveat": _accuracy_caveat(d.identifier, d.bucket),
    })


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
            # key == identifier for presets; catalog has no bucket.
            accuracy_caveat=_accuracy_caveat(e.key, None),
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

    now = datetime.now(UTC)
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
        # Catalog scope → the canonical where_rooms (where_scope is gone).
        preset_rooms = None if e.where_scope == "ALL ROOMS" else [e.where_scope]
        det = Detector(
            id=uuid4(),
            patient_id=patient_id,
            name=e.name,
            icon_name=e.icon_name,
            description=e.description,
            category=e.category,
            source="preset",
            alert_trigger=e.alert_trigger,
            is_sustained=e.is_sustained,
            duration_seconds=e.duration_seconds,
            where_rooms=preset_rooms,
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
            schedule_config=None,
            **catalog_v2.row_kwargs(e.key),
        )
        await detector_repo.insert(session, det)
        log.info(
            "[detectors] from-library add key=%s detector=%s patient=%s",
            e.key, det.id, patient_id,
        )
        created.append(det)

    if created:
        # Belt-and-braces reconcile: the per-detector assign above covers
        # the new rows; this repairs any pre-existing link drift for the
        # patient in the same transaction.
        await camera_detector_repo.sync_links_for_patient(session, patient_id)
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
    # v2 P5: measured-zero asks are refused HERE, before any LLM spend,
    # with the plain-language reason the UI shows verbatim.
    reason = compiler_v2.reject_reason(body.name or "", body.prompt)
    if reason is not None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "detector_rejected", "reason": reason},
        )
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
        suggested_name=inferred.suggested_name,
        alert_trigger=inferred.alert_trigger,  # type: ignore[arg-type]
        is_sustained=inferred.is_sustained,
        duration_seconds=inferred.duration_seconds,
        category=inferred.category,      # type: ignore[arg-type]
        icon_name=inferred.icon_name,
        sensitivity=inferred.sensitivity,  # type: ignore[arg-type]
        schedule_config=(
            ScheduleConfig(**inferred.schedule) if inferred.schedule else None
        ),
        episode_gap_seconds=inferred.episode_gap_seconds,
        plan=_compose_plan(
            name=body.name or "", prompt=body.prompt,
            alert_trigger=inferred.alert_trigger,
            is_sustained=inferred.is_sustained,
            category=str(inferred.category),
            duration_seconds=inferred.duration_seconds,
            schedule=inferred.schedule,
        ),
    )


def _is_absence(alert_trigger: object) -> bool:
    """Whether a detector fires on ABSENCE (not detected within its window) —
    the case that takes the conservative compile bias. Accepts an enum or raw
    string."""
    return str(getattr(alert_trigger, "value", alert_trigger)) == "absence"


def _fmt_time(hhmm: str) -> str:
    """"17:00" → "5:00 PM" for caregiver-facing firing copy. Passes the raw
    string through unchanged if it isn't HH:MM."""
    try:
        h_str, m_str = hhmm.split(":")
        h, m = int(h_str), int(m_str)
        suffix = "AM" if h < 12 else "PM"
        return f"{h % 12 or 12}:{m:02d} {suffix}"
    except Exception:
        return hhmm


def _fires_phrase(
    alert_trigger: str, is_sustained: bool, dims: dict,
    schedule: dict | None, duration_seconds: int | None = None,
) -> str:
    """The one-line plain-English firing behavior — the single place this
    copy is derived, so the create-review plan and any preview can never
    disagree. Caregiver language only, keyed on the alert trigger."""
    if alert_trigger == "absence":
        deadline = None
        if schedule:
            deadline = schedule.get("deadline") or schedule.get("window_end")
        if deadline:
            return f"Only if it doesn't happen by {_fmt_time(deadline)}"
        return "Only if it doesn't happen during the scheduled window"
    if alert_trigger == "trailing":
        return "When the activity finishes"
    if alert_trigger == "duration":
        if duration_seconds and duration_seconds >= 60:
            return f"Only if it goes on longer than {duration_seconds // 60} minutes"
        return "Only if it goes on too long"
    # leading: M-of-N corroboration reads as a short wait; a sustained
    # activity is one alert per session; an instant event fires on-sight.
    entry = (dims.get("eventing") or {}).get("entry") or {}
    m = int(entry.get("m", 1) or 1)
    t = int(entry.get("t_seconds", 0) or 0)
    if m > 1 and t > 0:
        return f"After it's noticed {m} times within {max(1, t // 60)} minutes"
    if is_sustained:
        return "One alert for the whole activity, not repeats"
    return "Right away, the moment it's noticed"


def _compose_plan(
    *, name: str, prompt: str, alert_trigger: str, is_sustained: bool,
    category: str, duration_seconds: int | None = None,
    schedule: dict | None = None,
):
    """Plain-English preview of the compiled behavior (customs
    visibility, Aryaman 2026-07-18). Pure classification — the same
    `classify_custom_dims` the create-time compiler uses, so the
    preview can never disagree with what actually gets built. Any
    failure returns None; the review screen hides the card."""
    from src.schemas.detector import DetectorPlan

    try:
        dims = compiler_v2.classify_custom_dims(
            name=name, user_text=prompt, is_sustained=is_sustained,
            category=category,
        )
        # Caregiver-facing "who" — warm, name-free (the infer endpoint has
        # no patient scope). "Your patient only" read clinically; "the
        # person you care for" is the app's own voice. The "Sees a close-up
        # of each person" media row was dropped entirely: how Memoria frames
        # the shot is an internal detail a caregiver shouldn't have to parse
        # (Aryaman 2026-07-26).
        watches = {
            "patient": "Only the person you care for",
            "anyone": "Anyone in view",
            "environment": "The room itself (no person needed)",
        }[dims["subject_scope"]]
        # Caregiver language only (Aryaman 2026-07-18): no internal
        # vocabulary — a non-technical reader must understand every line
        # without explanation. Shared with the type-change preview.
        fires = _fires_phrase(
            alert_trigger, is_sustained, dims, schedule, duration_seconds,
        )
        checks = (
            "Every few minutes"
            if dims["duty_cycle"] == "thinned"
            else "Continuously"
        )
        return DetectorPlan(
            watches=watches, fires=fires, checks=checks,
            listens=bool(dims.get("needs_audio")),
        )
    except Exception:
        log.debug("detector plan composition failed", exc_info=True)
        return None


@router.post(
    "/patients/{patient_id}/detectors",
    response_model=DetectorResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(RateLimit(CREATE_DETECTOR_RATE_LIMIT))],
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
    reject = compiler_v2.reject_reason(body.name, body.custom_prompt or "")
    if reject is not None:
        # v2 P5 hard-reject: never create a row for a detector that
        # measured zero — the caregiver gets the honest reason instead
        # of an alert they can't trust.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "detector_rejected", "reason": reject},
        )
    # 404 if the parent patient isn't owned by the caller — never leak
    # patient existence.
    patient = await patient_repo.find_by_id(session, patient_id, caregiver_id=auth.id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="patient_not_found",
        )

    now = datetime.now(UTC)
    # where_rooms is the canonical room scope (None = every room).
    resolved_rooms = body.where_rooms
    det = Detector(
        id=uuid4(),
        patient_id=patient_id,
        name=body.name,
        icon_name=body.icon_name,
        description=body.description,
        category=body.category.value,
        source="custom",
        alert_trigger=body.alert_trigger.value,
        is_sustained=body.is_sustained,
        duration_seconds=(
            body.duration_seconds
            if body.alert_trigger == AlertTrigger.duration else None
        ),
        where_rooms=resolved_rooms,
        when_scope=body.when_scope,
        sensitivity=body.sensitivity.value,
        is_on=body.is_on,
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
        # Null for reactive; the evaluation window for a schedule trigger.
        schedule_config=(
            body.schedule_config.model_dump() if body.schedule_config else None
        ),
        # Null unless sustained; the rising-edge episode gap.
        episode_gap_seconds=body.episode_gap_seconds,
    )
    await detector_repo.insert(session, det)

    # Links derive from where_rooms via the reconciler — except a custom
    # with NO room scope, which the reconciler deliberately skips (opt-in
    # semantics): initialize it to every camera once, here; the caregiver
    # prunes from /cameras/[id] afterwards.
    if camera_detector_repo.rooms_mean_all(resolved_rooms):
        await camera_detector_repo.set_detector_rooms(
            session, det.id, patient_id, resolved_rooms,
        )
    await camera_detector_repo.sync_links_for_patient(session, patient_id)
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
            absence=_is_absence(body.alert_trigger.value),
            is_sustained=det.is_sustained,
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
    absence: bool,
    is_sustained: bool,
    category: str,
    settings: Settings,
    identifier: str | None = None,
    refinements: list[str] | None = None,
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
            user_text=user_text, absence=absence, category=category,
            settings=settings,
            identifier=identifier, refinements=refinements,
        )
    except Exception as e:
        log.error(
            "detector_compile_background.failed detector_id=%s name=%r err=%s",
            detector_id, name, e,
        )
        return

    try:
        # Reserved-key guard (loud compile failure beats the silent
        # backend-wins overwrite at merge time) + v2 dims so the
        # scheduler treats this custom exactly like a built-in.
        compiler_v2.validate_schema_fields(compiled.schema_json)
        dims = compiler_v2.classify_custom_dims(
            name=name, user_text=user_text, is_sustained=is_sustained,
            category=category,
        )
    except compiler_v2.DetectorRejected as e:
        log.error(
            "detector_compile_background.rejected detector_id=%s reason=%s",
            detector_id, e.reason,
        )
        return

    # Caregiver-facing summary of the fresh fragment (customs
    # visibility): best-effort, regenerated on every compile so it can
    # never describe a stale prompt. None on failure = card hidden.
    criteria = await summarize_detector_prompt(
        fragment=compiled.prompt_fragment, settings=settings,
    )

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
                user_criteria=criteria,
                updated_at=datetime.now(UTC),
                **dims,
            )
        )
        await session.commit()
    log.info(
        "detector_compile_background.ok detector_id=%s identifier=%s criteria=%s",
        detector_id, compiled.identifier, bool(criteria),
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
    settings: SettingsDep,
) -> DetectorResponse:
    fields = body.model_dump(exclude_unset=True)
    # Enum → string for the ORM string columns.
    for k in ("category", "sensitivity", "alert_trigger"):
        if k in fields and fields[k] is not None:
            fields[k] = fields[k].value
    rescope_rooms = "where_rooms" in fields
    det = await detector_repo.update_partial(
        session, detector_id, caregiver_id=auth.id, fields=fields
    )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="detector_not_found",
        )
    # Replace the camera assignments to match the new room set (atomic:
    # deselected rooms stop, newly-selected rooms start). Custom detectors
    # only — presets (esp. wandering) carry per-camera door_config on their
    # CameraDetector rows that a blanket replace would wipe, and rooms
    # editing is a custom-detector affordance in the UI anyway.
    if (
        rescope_rooms
        and det.source == "custom"
        and camera_detector_repo.rooms_mean_all(fields["where_rooms"])
    ):
        # Custom rescoped to ALL ROOMS: re-initialize the opt-in set (the
        # reconciler skips null-scoped customs by design).
        await camera_detector_repo.set_detector_rooms(
            session, det.id, det.patient_id, fields["where_rooms"],
        )
    if rescope_rooms:
        # Re-derive links from the new where_rooms for ALL sources. For
        # presets this is the door_config-safe path set_detector_rooms
        # can't provide: the reconciler only inserts missing / deletes
        # stale links, never rewrites kept rows, so per-camera wandering
        # door_config survives a rescope. NOT run on plain is_on toggles
        # (rescope_rooms is False there) — the hot path stays link-free.
        await camera_detector_repo.sync_links_for_patient(
            session, det.patient_id,
        )
    await session.commit()
    # Recompile-on-edit (bug fix 2026-07-18): an edited custom_prompt
    # used to update the stored TEXT while the detector kept running on
    # the OLD compiled instructions forever. Now any prompt change on a
    # custom detector re-runs the background compile — identifier is
    # preserved so analyses history stays keyed; accumulated
    # refinements still apply on top of the new description.
    if (
        "custom_prompt" in fields
        and det.source == "custom"
        and det.custom_prompt
    ):
        asyncio.create_task(
            _compile_and_update_detector(
                detector_id=det.id,
                name=det.name,
                description=det.description or "Custom detector",
                user_text=det.custom_prompt,
                absence=_is_absence(det.alert_trigger),
                is_sustained=det.is_sustained,
                category=det.category,
                settings=settings,
                identifier=det.identifier,
                refinements=list(det.refinements or []),
            )
        )
    return _to_response(det)


@router.post(
    "/detectors/{detector_id}/refine",
    response_model=DetectorResponse,
)
async def refine_detector(
    detector_id: UUID,
    auth: CaregiverAuth,
    body: RefineDetectorRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> DetectorResponse:
    """One plain-words change request from the caregiver ("don't count
    eating as choking"). Appends to the detector's refinement list and
    recompiles with the FULL accumulated list — the compiler applies
    every refinement on top of the original description, never weakens
    the safety bias, and the fresh summary shows the result. Customs
    only: preset prompts are curated, not user-writable."""
    det = await detector_repo.find_by_id(
        session, detector_id, caregiver_id=auth.id,
    )
    if det is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="detector_not_found",
        )
    if det.source != "custom" or not det.custom_prompt:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="preset_not_refinable",
        )
    refinements = [*(det.refinements or []), body.feedback.strip()]

    # Re-infer the alert TYPE from the original ask PLUS every refinement, so a
    # plain-words change that implies a different trigger ("alert me when he
    # FINISHES eating, not when he starts" -> trailing) actually moves the
    # detector's type, not just its detection criteria. That's the whole point
    # of NL refine being flexible. Best-effort: if the re-inference call fails
    # we keep the current trigger rather than block the refine.
    combined_ask = (
        f"{det.custom_prompt}\n\nChanges the caregiver has asked for:\n"
        + "\n".join(f"- {r}" for r in refinements)
    )
    try:
        reinf = await infer_detector_config(
            name=det.name, prompt=combined_ask, settings=settings,
        )
        new_trigger = str(getattr(reinf.alert_trigger, "value", reinf.alert_trigger))
        new_sustained = bool(reinf.is_sustained)
        new_duration = reinf.duration_seconds if new_trigger == "duration" else None
        new_gap = reinf.episode_gap_seconds if new_sustained else None
        new_schedule = reinf.schedule if new_trigger == "absence" else None
        new_when = reinf.when_scope        # None if nothing usable was inferred
        new_where_hint = reinf.where_hint  # None if no place was named
    except Exception:
        log.debug("refine: trigger re-inference failed; keeping current trigger", exc_info=True)
        new_trigger = det.alert_trigger
        new_sustained = det.is_sustained
        new_duration = det.duration_seconds
        new_gap = det.episode_gap_seconds
        new_schedule = det.schedule_config
        new_when = None
        new_where_hint = None

    # WHEN: apply the inferred window only when the model gave one; otherwise
    # keep the detector's current window (never blow away a user-set scope).
    when_scope = new_when if new_when else det.when_scope

    # WHERE: if the caregiver named a place ("only in the kitchen"), match it to
    # one of THIS patient's real camera rooms (exact, then a loose contains
    # fallback). Only re-scope when it maps to an actual room, since a detector
    # can't run where there's no camera; otherwise keep the current rooms.
    where_rooms = det.where_rooms
    if new_where_hint:
        patient_rooms = (
            await session.execute(
                select(Camera.room).where(
                    Camera.patient_id == det.patient_id,
                    Camera.room.isnot(None),
                )
            )
        ).scalars().all()
        hint = new_where_hint.strip().lower()
        match = next(
            (r for r in patient_rooms if r and r.strip().lower() == hint), None,
        ) or next(
            (r for r in patient_rooms
             if r and (hint in r.strip().lower() or r.strip().lower() in hint)),
            None,
        )
        if match:
            where_rooms = [match]

    # Dry-run ("Re-generate"): compile WITH this feedback applied and hand
    # back the resulting criteria + the re-inferred trigger, but write nothing
    # — no refinement stored, no row update, so the live detector keeps running
    # its current instructions until the caregiver hits Save.
    # Synchronous (the user is waiting).
    if body.preview:
        try:
            compiled = await compile_custom_detector(
                name=det.name,
                description=det.description or "Custom detector",
                user_text=det.custom_prompt,
                absence=_is_absence(new_trigger),
                category=det.category,
                settings=settings,
                identifier=det.identifier,
                refinements=refinements,
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="detector_preview_unavailable",
            ) from e
        criteria = await summarize_detector_prompt(
            fragment=compiled.prompt_fragment, settings=settings,
        )
        # Return the unchanged row with the PREVIEWED criteria + re-inferred
        # trigger swapped in; persistence is skipped entirely.
        return _to_response(det).model_copy(update={
            "user_criteria": criteria,
            "alert_trigger": AlertTrigger(new_trigger),
            "is_sustained": new_sustained,
            "duration_seconds": new_duration,
            "when_scope": when_scope,
            "where_rooms": where_rooms,
        })

    old_rooms = det.where_rooms
    det = await detector_repo.update_partial(
        session, detector_id, caregiver_id=auth.id,
        fields={
            "refinements": refinements,
            "alert_trigger": new_trigger,
            "is_sustained": new_sustained,
            "duration_seconds": new_duration,
            "episode_gap_seconds": new_gap,
            "schedule_config": new_schedule,
            "when_scope": when_scope,
            "where_rooms": where_rooms,
        },
    )
    # Re-assign cameras when the NL refine re-scoped the rooms, then
    # reconcile so links always derive from where_rooms.
    if where_rooms != old_rooms:
        if det.source == "custom" and camera_detector_repo.rooms_mean_all(
            where_rooms
        ):
            await camera_detector_repo.set_detector_rooms(
                session, det.id, det.patient_id, where_rooms,
            )
        await camera_detector_repo.sync_links_for_patient(
            session, det.patient_id,
        )
    await session.commit()
    asyncio.create_task(
        _compile_and_update_detector(
            detector_id=det.id,
            name=det.name,
            description=det.description or "Custom detector",
            user_text=det.custom_prompt,
            absence=_is_absence(new_trigger),
            is_sustained=new_sustained,
            category=det.category,
            settings=settings,
            identifier=det.identifier,
            refinements=refinements,
        )
    )
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
