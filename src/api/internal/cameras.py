"""Support-only /internal camera routes (room-derived privacy).

- POST /internal/cameras/{id}/reclassify — the misclassification escape
  hatch (room-privacy plan ⚑3). Privacy is locked with no user toggle,
  so a misclassified living room needs a support-side remedy:

    * no body / `{}`        → clear the camera's classification
      (room_classified_at + confidence) so the room classifier re-runs
      on the next frames. `room_class` + `privacy_enabled` are KEPT
      until the new verdict lands (fail-closed — a bathroom stays
      private through the window, a wrongly-private living room unlocks
      only once a fresh verdict says so).
    * `{"room_class": "other"}` (or bathroom/bedroom) → forced override:
      finalize that class immediately (privacy_enabled derived from it)
      and stamp room_classified_at so the classifier does NOT re-run.

  A privacy flip to true triggers the per-camera live sever, same as a
  classifier finalization.

Auth (INTERNAL_SERVICE_KEY pattern): callers present the shared
`NEMARA_INTERNAL_SERVICE_KEY` in the `X-Internal-Key` header. Fails
CLOSED outside env=local/test when the key is unset — an unconfigured
prod deploy rejects every /internal call rather than opening them.
Never caregiver-JWT-reachable: this is an ops surface, not a product
one.
"""

import logging
import secrets
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict

from src.api.deps import SessionDep, SettingsDep
from src.db.camera import Camera
from src.schemas.camera import RoomClass
from src.services import room_classifier
from src.services.livekit_privacy import enforce_privacy_for_camera
from src.services.room_classifier import PRIVATE_ROOM_CLASSES

log = logging.getLogger(__name__)

router = APIRouter(prefix="/internal", tags=["internal"])


def _require_internal_key(settings, x_internal_key: str) -> None:
    """Shared-secret guard for /internal routes. Compare in constant
    time; fail closed (403) when the key is unset outside local/test.
    """
    expected = settings.internal_service_key
    if not expected:
        if settings.env in ("local", "test"):
            return  # local dev convenience — no key needed on a laptop
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="internal_key_not_configured",
        )
    if not secrets.compare_digest(x_internal_key, expected):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="invalid_internal_key",
        )


class ReclassifyRequest(BaseModel):
    """Body of POST /internal/cameras/{id}/reclassify. Empty body =
    clear-and-rerun; `room_class` set = forced override."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    room_class: RoomClass | None = None


class ReclassifyResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    camera_id: UUID
    room_class: RoomClass | None
    room_class_confidence: float | None
    room_classified_at: datetime | None
    privacy_enabled: bool


@router.post(
    "/cameras/{camera_id}/reclassify",
    response_model=ReclassifyResponse,
    status_code=status.HTTP_200_OK,
)
async def reclassify_camera(
    camera_id: UUID,
    session: SessionDep,
    settings: SettingsDep,
    body: ReclassifyRequest | None = None,
    x_internal_key: str = Header(default=""),
) -> ReclassifyResponse:
    _require_internal_key(settings, x_internal_key)
    cam = await session.get(Camera, camera_id)
    if cam is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="camera_not_found",
        )

    forced = body.room_class if body is not None else None
    privacy_flipped_on = False
    if forced is None:
        # Clear-and-rerun. Verdict fields survive until the classifier
        # writes a fresh one (fail-closed, see module docstring).
        cam.room_class_confidence = None
        cam.room_classified_at = None
        log.info(
            "internal.reclassify cleared camera=%s (classifier re-arms)",
            camera_id,
        )
    else:
        new_privacy = forced in PRIVATE_ROOM_CLASSES
        privacy_flipped_on = new_privacy and not cam.privacy_enabled
        cam.room_class = forced
        cam.room_class_confidence = None  # human verdict — no model confidence
        cam.room_classified_at = datetime.now(UTC)
        cam.privacy_enabled = new_privacy
        log.info(
            "internal.reclassify override camera=%s room_class=%s "
            "privacy_enabled=%s",
            camera_id, forced, new_privacy,
        )
    livekit_room = cam.livekit_room
    livekit_identity = cam.livekit_identity
    await session.commit()

    # Either path restarts the in-process attempt ledger: cleared
    # cameras must classify from attempt 1; overridden cameras must not
    # have a stale in-flight attempt racing the human verdict (the
    # classifier's room_classified_at-IS-NULL write guard already makes
    # that race lose, this just stops the wasted VLM call).
    room_classifier.reset_attempts(camera_id)

    if privacy_flipped_on:
        # Same instant defense-in-depth sever as a classifier
        # finalization; best-effort (helper swallows LiveKit errors),
        # publisher-side track permissions are the durable cut.
        await enforce_privacy_for_camera(
            settings, room=livekit_room, camera_identity=livekit_identity,
        )

    return ReclassifyResponse(
        camera_id=cam.id,
        room_class=cam.room_class,  # type: ignore[arg-type]
        room_class_confidence=cam.room_class_confidence,
        room_classified_at=cam.room_classified_at,
        privacy_enabled=cam.privacy_enabled,
    )
