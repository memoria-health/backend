"""ORM model for the `patient_face_embeddings` table.

Maps 1:1 with `supabase/schema.sql` §patient_face_embeddings. Each row
is a single 512-dim ArcFace embedding used by the Re-ID inference
service to identify the patient at detector tick time.

Two row kinds, distinguished by `source`:
  - 'selfie': caregiver-uploaded during onboarding (10-15 angle-prompted
              captures). `source_path` is the Supabase Storage key.
  - 'harvested': inserted by `services/reid/harvester.py` when Re-ID
                 locks a track with very high confidence. Closes the
                 selfie-to-wall-cam domain gap over time. `source_path`
                 is NULL (we keep only the embedding, not the frame).

`status` is non-trivial only for the 'selfie' path which processes
asynchronously after multipart upload. 'harvested' rows are always
written with status='ready' inline.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import REAL, DateTime, ForeignKey, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class PatientFaceEmbedding(Base):
    __tablename__ = "patient_face_embeddings"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"),
    )
    # 512-dim float32 ArcFace embedding, L2-normalized, stored as raw bytes
    # (~2KB / row). Decoded via np.frombuffer(embedding, dtype=np.float32)
    # by the repo + the inference service. Kept opaque at the ORM layer
    # so the model doesn't drag numpy as a dependency.
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    det_score: Mapped[float] = mapped_column(REAL)
    source: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    source_path: Mapped[str | None] = mapped_column(String, default=None)
    failure_reason: Mapped[str | None] = mapped_column(String, default=None)
