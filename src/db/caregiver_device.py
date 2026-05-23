"""ORM model for the `caregiver_devices` table.

One row per APNs device token. A caregiver may have multiple device
tokens registered (phone + iPad + a re-installed app on the same
phone). `apns_token` is globally unique so a re-registration from the
same device idempotently replaces the prior row via the upsert path
in `src/repos/caregiver_device_repo.py`.

Read by `src/services/push.py` when a detector event fires; written
by `POST /v1/me/devices` from the caregiver-app at launch + on token
refresh.
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class CaregiverDevice(Base):
    __tablename__ = "caregiver_devices"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    caregiver_id: Mapped[UUID] = mapped_column(
        ForeignKey("caregivers.id", ondelete="CASCADE"),
    )
    apns_token: Mapped[str] = mapped_column(String, unique=True)
    bundle_id: Mapped[str] = mapped_column(String)
    # "sandbox" for Xcode dev / TestFlight-debug; "production" for
    # TestFlight release + App Store. Different APNs endpoints accept
    # different tokens; mixing them returns BadDeviceToken.
    environment: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
