"""Clip ingest endpoint.

The wall phone POSTs a 15s MP4 every ~10s. Auth: camera api-key.
Form fields mirror the dashboard exactly so the wall phone code in
`nemara-camera-app` only needs the URL changed.

Pipeline (synchronous in Phase 0):
  1. Read MP4 + form fields.
  2. Load this camera's enabled detector specs.
  3. Call Gemini → fall back to OpenAI on any Gemini failure.
  4. Apply tick to state machine; UPDATE cameras row.
  5. INSERT analyses row.
  6. Evaluate analysis for detector promotions; INSERT events rows.
  7. Return the dashboard-compatible response shape.

Phase 2 will move steps 2-6 into a long-running asyncio worker that
subscribes to LiveKit rooms — same internal modules, different
input loop.
"""

import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from src.api.deps import CameraAuth, SessionDep, SettingsDep
from src.schemas.clips import IngestResponse
from src.services.detection.analyses_store import add_analysis
from src.services.detection.event_promotion import maybe_promote_analysis
from src.services.detection.providers.gemini import gemini_provider
from src.services.detection.providers.openai import openai_provider
from src.services.detection.providers.types import ProviderResult
from src.services.detection.registry import get_enabled_detectors_for_camera
from src.services.detection.state_machine import apply_tick_update

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["clips"])


@router.post(
    "/clips/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
)
async def ingest_clip(
    cam: CameraAuth,
    session: SessionDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    camera_id: str = Form(...),
    timestamp: str = Form(...),
    motion_in_clip: str = Form(...),
) -> IngestResponse:
    # Defense-in-depth: the api-key resolves a camera, and the form
    # carries a camera_id. Reject mismatched submissions even though
    # the api-key path already has the authoritative camera.
    try:
        form_camera_id = UUID(camera_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="camera_id is not a UUID",
        ) from e
    if form_camera_id != cam.id:
        log.info(
            "ingest.reject reason=camera_id_mismatch api_key_camera=%s form_camera=%s",
            cam.id, form_camera_id,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="camera_id_mismatch",
        )

    if motion_in_clip not in ("true", "false"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='motion_in_clip must be "true" or "false"',
        )
    motion_bool = motion_in_clip == "true"

    try:
        captured_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="timestamp must be ISO-8601",
        ) from e

    mp4_bytes = await file.read()
    if not mp4_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file is empty",
        )

    detectors = await get_enabled_detectors_for_camera(session, cam.id)
    log_tag = f"ingest cam={cam.id}"
    log.info("%s START size=%dKB detectors=%s motion=%s prior_state=%s",
             log_tag, len(mp4_bytes) // 1024,
             [d.identifier for d in detectors], motion_bool, cam.state)

    # Closed-state short-circuit: skip the (expensive) LLM call when
    # nobody's been around. Any motion forces a re-evaluation in case
    # someone walked in; without motion the state machine just bumps
    # the counter and we return early. Matches the dashboard's intent
    # and the plan's `test_closed_camera_skips_llm` requirement.
    if cam.state == "closed" and not motion_bool:
        state_after = await apply_tick_update(
            session,
            camera_id=cam.id,
            people_present=False,
            motion_in_clip=False,
        )
        # Synthetic analyses row so downstream consumers (active-camera
        # resolution for Talk routing) still see a tick. provider tag
        # `skipped` is distinct from real LLM-provider values.
        # Inserted with a minimal description so it's still grep-able
        # in Studio.
        synthetic_fields = {
            "people_present": False,
            "skipped_reason": "camera_closed",
        }
        analysis = await add_analysis(
            session,
            camera_id=cam.id,
            provider="skipped",
            description="(closed — LLM skipped)",
            detector_fields=synthetic_fields,
            people_present=False,
            people_count=0,
            motion_in_clip=False,
            state_after=state_after.state,
            captured_at=captured_at,
        )
        await session.commit()
        log.info("%s SKIP closed-state, no motion — counter=%s/%s",
                 log_tag, state_after.no_people_count, state_after.close_threshold)
        return IngestResponse(
            id=analysis.id,
            provider="skipped",
            analysis=analysis.description,
            people_present=False,
            people_count=0,
            detector_fields=synthetic_fields,
            state=state_after.state,
            no_people_count=state_after.no_people_count,
            close_threshold=state_after.close_threshold,
        )

    # Provider fallback: Gemini → OpenAI. Either may raise; if Gemini
    # fails we try OpenAI; if both fail, return 502.
    used_provider: str
    result: ProviderResult
    try:
        used_provider = "gemini"
        result = await gemini_provider(
            mp4=mp4_bytes, detectors=detectors, log_tag=log_tag, settings=settings,
        )
    except Exception as gem_err:
        log.warning("%s gemini failed, trying openai: %s", log_tag, gem_err)
        try:
            used_provider = "openai"
            result = await openai_provider(
                mp4=mp4_bytes, detectors=detectors, log_tag=log_tag, settings=settings,
            )
        except Exception as oa_err:
            log.error("%s both providers failed: gemini=%s openai=%s",
                      log_tag, gem_err, oa_err)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="both_providers_failed",
            ) from oa_err

    # State machine + analyses + event promotion. All inside the same
    # session; commit at the end.
    state_after = await apply_tick_update(
        session,
        camera_id=cam.id,
        people_present=result.people_present,
        motion_in_clip=motion_bool,
    )
    analysis = await add_analysis(
        session,
        camera_id=cam.id,
        provider=used_provider,
        description=result.description,
        detector_fields=result.detector_fields,
        people_present=result.people_present,
        people_count=result.people_count,
        motion_in_clip=motion_bool,
        state_after=state_after.state,
        captured_at=captured_at,
    )
    promotion = await maybe_promote_analysis(
        session,
        patient_id=cam.patient_id,
        camera_id=cam.id,
        occurred_at=captured_at,
        description=result.description,
        detector_fields=result.detector_fields,
    )
    await session.commit()

    log.info(
        "%s DONE provider=%s people=%s(%s) motion=%s state=%s/%s "
        "events_fired=%d dedup=%d",
        log_tag, used_provider, result.people_present, result.people_count,
        motion_bool, state_after.state, state_after.no_people_count,
        len(promotion.fired), len(promotion.skipped_for_dedup),
    )

    return IngestResponse(
        id=analysis.id,
        provider=used_provider,
        analysis=analysis.description,
        people_present=result.people_present,
        people_count=result.people_count,
        detector_fields=result.detector_fields,
        state=state_after.state,
        no_people_count=state_after.no_people_count,
        close_threshold=state_after.close_threshold,
    )
