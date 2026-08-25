"""ORM model for the `camera_patients` join table.

Which patients a sensor watches. A sensor belongs to an account and
covers 0..N of that account's patients, as equals — there is no primary.
There was an `is_primary` column until 2026-08-22; it existed only to
choose whose LiveKit room a shared sensor published into, and
`apply_account_room_delta.sql` dropped it when the room became the
account's.

Coverage is pure attribution. Nothing about the stream reads it: a sensor
with no rows here still publishes, and adding or removing a patient never
moves the video. What it DOES decide is which patients' detectors run on
this sensor and who an event is about.

The table came from `apply_camera_ownership_delta.sql` and was reached
only through raw SQL until this model, which existed nowhere while
`camera_detector_repo` needed to join against it.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class CameraPatient(Base):
    __tablename__ = "camera_patients"

    camera_id: Mapped[UUID] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True,
    )
    patient_id: Mapped[UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), primary_key=True,
    )
    # Display order for coverage, and the tiebreak `cameras.patient_id`
    # uses while that compatibility column still exists.
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
