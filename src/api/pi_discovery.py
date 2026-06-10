"""Pi camera discovery API surface.

  - POST /v1/cameras/pi-announce       (unauthenticated, called by Pi)
  - GET  /v1/cameras/discovered        (caregiver-authenticated)

The announce endpoint is intentionally unauthenticated because the
Pi device has no credentials before pairing. The WiFi captive-portal
provisioning flow (scripts/pi/wifi_ap_provision.py) is the trust
anchor in the customer flow: physical proximity to write WiFi
credentials. v2 will add a per-device secret negotiated at setup
that the announcement carries, so the backend can attribute
discoveries to specific caregivers.

The discovered endpoint is caregiver-authenticated and filters to
recently-seen unclaimed Pis so stale devices fall off the picker.
There is currently no per-caregiver scoping (every caregiver sees
all unclaimed Pis); this is acceptable at the v1 customer scale of
10-20 devices.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep
from src.repos import pi_discovery_repo
from src.schemas.pi_discovery import (
    DiscoveredPi,
    DiscoveredPisResponse,
    PiAnnounceRequest,
    PiAnnounceResponse,
)

router = APIRouter(prefix="/v1", tags=["pi-discovery"])

# Window during which an announcement is "recent enough" to surface
# in the discovered list. 5 minutes is well above the 60 s announce
# cadence on the Pi side, so a single dropped announcement does not
# cause the camera to vanish from the picker.
_DISCOVERY_WINDOW_SECONDS: int = 5 * 60


@router.post(
    "/cameras/pi-announce",
    response_model=PiAnnounceResponse,
    status_code=status.HTTP_200_OK,
)
async def announce_pi(
    body: PiAnnounceRequest,
    session: SessionDep,
    settings: SettingsDep,
) -> PiAnnounceResponse:
    """Upsert a discovery row keyed by hardware_id.

    Called by the Pi-side announce daemon. No authentication; the
    Pi has no credentials before pairing. The hardware_id (MAC
    address) is the de-dupe key; re-announcing just refreshes
    last_seen_at and current_url.

    If a caregiver has already provisioned this Pi, the response
    carries the ingest URL + plaintext bearer token so the Pi can
    switch from the announce loop to the push-frames loop. The
    token is delivered exactly once (consume_pending_ingest_token
    nulls it after read); a Pi that misses the response forces a
    re-provision to mint a fresh token.
    """
    discovery = await pi_discovery_repo.upsert_announcement(
        session,
        hardware_id=body.hardware_id,
        device_label=body.device_label,
        current_url=body.current_url,
        current_audio_url=body.current_audio_url,
    )
    pending = await pi_discovery_repo.consume_pending_ingest_token(
        session, hardware_id=body.hardware_id,
    )
    paired = discovery.claimed_at is not None
    ingest_token: str | None = None
    ingest_url: str | None = None
    if pending is not None:
        _camera_id, ingest_token = pending
        ingest_url = f"{settings.public_base_url}/v1/cameras/me/frames"
    await session.commit()
    return PiAnnounceResponse(
        discovery_id=discovery.id,
        last_seen_at=discovery.last_seen_at,
        paired=paired,
        ingest_url=ingest_url,
        ingest_token=ingest_token,
    )


@router.get(
    "/cameras/discovered",
    response_model=DiscoveredPisResponse,
)
async def list_discovered(
    auth: CaregiverAuth,
    session: SessionDep,
) -> DiscoveredPisResponse:
    """Unclaimed Pi devices seen within the last
    `_DISCOVERY_WINDOW_SECONDS`. Caregiver-authenticated so the
    endpoint cannot be scraped by anonymous clients; otherwise no
    per-caregiver filtering at v1 (see module docstring).
    """
    rows = await pi_discovery_repo.list_unclaimed_recent(
        session, seen_within_seconds=_DISCOVERY_WINDOW_SECONDS,
    )
    return DiscoveredPisResponse(
        pis=[
            DiscoveredPi(
                discovery_id=row.id,
                hardware_id=row.hardware_id,
                device_label=row.device_label,
                current_url=row.current_url,
                current_audio_url=row.current_audio_url,
                first_seen_at=row.first_seen_at,
                last_seen_at=row.last_seen_at,
            )
            for row in rows
        ]
    )
