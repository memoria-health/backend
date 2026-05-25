"""Repository layer — pure async SQLAlchemy queries, no business logic.

Each repo exports module-level functions that take an `AsyncSession`
and read or write a single aggregate. Transactions are owned by the
caller (typically a service).
"""

from src.repos import (
    camera_detector_repo,
    camera_repo,
    caregiver_repo,
    detector_repo,
    event_repo,
    patient_repo,
)

__all__ = [
    "camera_detector_repo",
    "camera_repo",
    "caregiver_repo",
    "detector_repo",
    "event_repo",
    "patient_repo",
]
