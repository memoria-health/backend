"""Pydantic boundary models for the Pi discovery flow.

Discovery is a three-step handshake:

  1. Pi runs `scripts/pi/announce_daemon.py` on a 60 s tick. It
     POSTs to `/v1/cameras/pi-announce` with a stable hardware ID
     and its current MJPEG URL. The endpoint is unauthenticated
     because the Pi has no credentials before pairing.

  2. Caregiver opens the "Add a camera" screen. The app calls
     `GET /v1/cameras/discovered` (caregiver-authenticated) to list
     recently-seen unclaimed Pis. The picker shows
     `device_label` so the caregiver can match a record to the
     physical device.

  3. Caregiver picks one and assigns it to a patient + room. The app
     calls `POST /v1/cameras/pi-provision` with the discovery_id
     instead of a direct pi_stream_url. The backend looks up the
     discovery, copies the URLs into the new cameras row, marks the
     discovery claimed.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# MAC-address-like identifier, e.g. "b8:27:eb:a4:b7:c2". Cap at 64
# to leave room for non-MAC schemes in the future.
HardwareId = Annotated[str, Field(min_length=1, max_length=64)]

# Device label, e.g. "Nemara Camera A4B7C2" (last 6 of MAC).
DeviceLabel = Annotated[str, Field(min_length=1, max_length=120)]

# Mirrors cameras.pi_stream_url constraint.
DiscoveryUrl = Annotated[str, Field(min_length=8, max_length=2048)]


class PiAnnounceRequest(BaseModel):
    """Body of POST /v1/cameras/pi-announce. Sent by the Pi-side
    announce daemon on a 60 s cadence (or on boot if the IP changes).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    hardware_id: HardwareId
    device_label: DeviceLabel
    current_url: DiscoveryUrl
    current_audio_url: DiscoveryUrl | None = None


class PiAnnounceResponse(BaseModel):
    """Returned to the Pi after a successful announcement.

    `paired` flips to True after a caregiver provisions this
    discovery; that's the Pi's signal to switch from the announce-
    loop (poll for pairing) to the push-frames loop (stream MJPEG
    out to /v1/cameras/me/frames).

    `ingest_url` and `ingest_token` are delivered ONCE, on the
    announce immediately following provisioning. The backend nulls
    `pi_discoveries.pending_ingest_token` in the same transaction so
    a second read returns `ingest_token=None`. The Pi is expected to
    write the token to /etc/nemara/ingest.env on receipt; if the
    response is lost in transit the caregiver re-provisions
    (which mints a fresh token).
    """

    model_config = ConfigDict(frozen=True)

    discovery_id: UUID
    last_seen_at: datetime
    paired: bool = False
    ingest_url: str | None = None
    ingest_token: str | None = None


class DiscoveredPi(BaseModel):
    """One unclaimed discovered Pi, as shown to the caregiver in
    the "Add a camera" picker.
    """

    model_config = ConfigDict(frozen=True)

    discovery_id: UUID
    hardware_id: HardwareId
    device_label: DeviceLabel
    current_url: DiscoveryUrl
    current_audio_url: DiscoveryUrl | None = None
    first_seen_at: datetime
    last_seen_at: datetime


class DiscoveredPisResponse(BaseModel):
    """Top-level shape for GET /v1/cameras/discovered."""

    model_config = ConfigDict(frozen=True)

    pis: list[DiscoveredPi]
