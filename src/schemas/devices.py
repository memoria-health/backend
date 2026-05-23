"""Pydantic boundary models for `/v1/me/devices`.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Services/NemaraAPIClient.swift`'s
`RegisterDeviceTokenRequest`. The hex-encoded APNs token comes off
`application(_:didRegisterForRemoteNotificationsWithDeviceToken:)`;
we don't validate length here because Apple's token length has
changed historically.
"""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

ApnsToken = Annotated[str, Field(min_length=1, max_length=512)]
BundleId = Annotated[str, Field(min_length=1, max_length=255)]


class RegisterDeviceRequest(BaseModel):
    """Body of `POST /v1/me/devices`. The caregiver app posts this on
    launch and whenever APNs reissues a token (rare but possible)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    apns_token: ApnsToken
    bundle_id: BundleId
    environment: Literal["sandbox", "production"]


class DeviceResponse(BaseModel):
    """Returned from `POST /v1/me/devices`. iOS doesn't strictly need
    the full row back, but matching wolf's pattern of returning the
    canonical row lets the caregiver app cache state without an extra
    GET."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    caregiver_id: UUID
    apns_token: ApnsToken
    bundle_id: BundleId
    environment: Literal["sandbox", "production"]
    created_at: datetime
    updated_at: datetime
