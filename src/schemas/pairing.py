"""Pydantic boundary models for the camera pairing flow.

Pairing is a three-step handshake:

1. Caregiver app (caregiver-JWT auth) posts to
   `/v1/patients/{patient_id}/cameras/pairing-code` with the room
   metadata it wants the new camera to register with. Server inserts
   a `cameras` row with `online=false`, mints a 6-digit
   `pairing_code` valid for `NEMARA_PAIRING_CODE_TTL_SECONDS`, returns
   the code + camera_id.

2. Wall phone (unauthenticated; the code IS the auth) posts to
   `/v1/cameras/pair` with the code. Server generates a 32-byte
   `api_key`, bcrypt-hashes it, clears the code/expiry, returns the
   api_key to the phone along with everything the phone needs to
   start publishing to LiveKit.

3. Wall phone uses the api_key as its bearer token for every
   subsequent request (clip ingest, LiveKit token refresh, etc.).
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# Room name + icon match the existing camera UI fields — see
# `cameras` table in `supabase/schema.sql`.
RoomName = Annotated[str, Field(min_length=1, max_length=80)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]


class CreatePairingCodeRequest(BaseModel):
    """Body of POST /v1/patients/{patient_id}/cameras/pairing-code.
    Mints a new code and inserts a pre-paired `cameras` row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    room: RoomName
    icon_name: IconName = "tv"


class CreateAccountPairingCodeRequest(BaseModel):
    """Body of POST /v1/cameras/pairing-code.

    Same as above minus the patient in the URL. Coverage is an ordinary
    optional field here, which is the whole point: a phone paired from
    the sensor roster names nobody and arrives unassigned, exactly like
    an installed sensor, while one paired from a resident's Live view
    passes that resident and arrives watching them. One endpoint, and
    attribution lives in the same concept for every device class.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    room: RoomName
    icon_name: IconName = "tv"
    # Empty is the roster case and is entirely legal.
    patient_ids: list[UUID] = []


class CreatePairingCodeResponse(BaseModel):
    """Returned to the caregiver app after a code is minted. The code
    is short-lived; the caregiver app should countdown to the expiry
    and refresh if the user takes too long.
    """

    model_config = ConfigDict(frozen=True)

    pairing_code: str  # 6 digits
    expires_at: datetime
    camera_id: UUID


class PairRequest(BaseModel):
    """Body of POST /v1/cameras/pair (unauthenticated).
    The pairing_code IS the proof of authorization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pairing_code: Annotated[str, Field(pattern=r"^\d{6}$")]


class PairResponse(BaseModel):
    """Handed back to the wall phone after a successful exchange.
    `api_key` is the long-lived bearer the phone uses for every
    subsequent request; the server only stores its bcrypt hash.
    """

    model_config = ConfigDict(frozen=True)

    api_key: str                # 64 hex chars (32 random bytes)
    camera_id: UUID
    patient_id: UUID
    livekit_url: str
    livekit_identity: str
    livekit_room: str
    backend_base_url: str
