"""Pydantic boundary model for cameras.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/Camera.swift`.
The iOS app only lists + deletes; v0 has no create body type.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Room = Annotated[str, Field(min_length=1, max_length=80)]
IconName = Annotated[str, Field(min_length=1, max_length=80)]


class CameraResponse(BaseModel):
    """Full camera row."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    patient_id: UUID
    display_id: int
    room: Room
    icon_name: IconName
    online: bool
    last_seen_at: datetime | None
    created_at: datetime
