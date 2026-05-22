"""Pydantic boundary models for the LiveKit-token endpoint.

`POST /v1/livekit/token` is dual-auth:
  - camera api_key → may only request `role='camera'` for its own room
  - caregiver JWT → may request `role='viewer'` or `'broadcaster'`
                    for any room owned by their patients

Returns `{token, url}` — same shape the dashboard's
`/api/livekit/token` returns, so caller code in the camera app and
caregiver app is trivially portable.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


# Same camera-role naming the dashboard uses (`'camera' | 'viewer'`).
# `'broadcaster'` is new for nemara — caregiver Talk mode (Phase 3).
LiveKitRole = Literal["camera", "viewer", "broadcaster"]


class MintTokenRequest(BaseModel):
    """Body of POST /v1/livekit/token."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Room name is server-validated against the authenticated principal:
    # for `camera` role, must match the camera's `livekit_room`.
    # For `viewer`/`broadcaster`, must be one of the caregiver's
    # patients' rooms (`patient_<patient_id>`).
    room: Annotated[str, Field(min_length=1, max_length=120)]
    # Identity is informational on the LiveKit side; we still let the
    # caller specify it so the dashboard's "Cam 01" label can survive.
    # Server enforces the prefix for camera identities (`cam_<...>`).
    identity: Annotated[str, Field(min_length=1, max_length=120)]
    role: LiveKitRole


class MintTokenResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    url: str
    # ISO-8601 expiry — clients refresh at TTL/2 to keep ahead of the
    # 1-hour grant. Computed server-side from `_TTL_SECONDS` in
    # `src/api/livekit.py`, not pulled out of the minted JWT.
    expires_at: str
