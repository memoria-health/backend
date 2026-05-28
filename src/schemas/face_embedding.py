"""Pydantic boundary models for the face-embedding API surface.

Mirrors `nemara-caregiver-app/NemaraCaregiver/Models/FaceEmbedding.swift`
and `nemara-web-app/lib/api/types.ts` — keep the three in sync.

Embeddings are never returned over the wire. The 512-dim float32
buffer is internal to the Re-ID pipeline and useless to clients;
exposing it would leak a privacy-sensitive face template. The boundary
type carries only the metadata fields the dashboard cards render.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class FaceEmbeddingSource(str, Enum):
    """How this embedding entered the gallery. Caregiver-uploaded
    selfies vs auto-harvested wall-cam crops are distinguished here so
    the dashboard can show "12 selfies + 47 harvested" separately.
    """
    selfie = "selfie"
    harvested = "harvested"


class FaceEmbeddingStatus(str, Enum):
    """Processing status. Only meaningful for selfie rows — harvested
    rows are always written ready.
    """
    processing = "processing"
    ready = "ready"
    failed = "failed"


class FaceEmbeddingResponse(BaseModel):
    """A single row in the patient's face gallery."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    patient_id: UUID
    source: FaceEmbeddingSource
    status: FaceEmbeddingStatus
    det_score: float = Field(ge=0.0, le=1.0)
    failure_reason: str | None = None
    created_at: datetime


class FaceEnrollmentSummary(BaseModel):
    """Status-card data, shared verbatim between iOS PatientSettings
    card and web PatientIdentityStatus card.

    `ready_count` is the count of ready embeddings, broken down by
    source. `last_update` is the newest created_at among ready rows
    (or None when the gallery is empty).

    `processing_count` is selfies that haven't been embedded yet —
    surfaces during the brief window after a caregiver submits
    enrollment and before SCRFD + ArcFace finish.
    """

    model_config = ConfigDict(frozen=True)

    selfie_count: int = Field(ge=0)
    harvested_count: int = Field(ge=0)
    processing_count: int = Field(ge=0)
    failed_count: int = Field(ge=0)
    last_update: datetime | None = None
