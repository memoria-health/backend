"""ORM model for the `cameras` table.

Rows are inserted by the caregiver app's "Pair a new phone" flow:
`POST /v1/patients/{id}/cameras/pairing-code` creates a row with a
6-digit `pairing_code` valid for 10 minutes. The wall phone then
calls `POST /v1/cameras/pair {pairing_code}` to exchange the code
for a long-lived (bcrypt-hashed) `api_key`. After pairing, the row
holds:
  - LiveKit identity + room (camera knows where to publish)
  - the detector state machine's persistent fields (state, no_people_count)
"""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    patient_id: Mapped[UUID] = mapped_column(ForeignKey("patients.id", ondelete="CASCADE"))
    display_id: Mapped[int] = mapped_column(Integer)
    room: Mapped[str] = mapped_column(String(80))
    icon_name: Mapped[str] = mapped_column(String(80))
    online: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)

    # ── Pairing flow ───────────────────────────────────────────────
    # `pairing_code` + `pairing_expires_at` are populated when the
    # caregiver mints a code and cleared when the wall phone exchanges
    # it. `api_key_hash` is the bcrypt hash of the long-lived API key
    # handed back to the phone — used for every clip-ingest auth and
    # for LiveKit token minting on the camera path.
    pairing_code: Mapped[str | None] = mapped_column(String(6), default=None)
    pairing_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
    api_key_hash: Mapped[str | None] = mapped_column(String(80), default=None)

    # ── LiveKit identity ───────────────────────────────────────────
    # Set when the caregiver mints a pairing code so they can identify
    # the camera in dashboards even before the phone connects.
    livekit_identity: Mapped[str | None] = mapped_column(String(80), default=None)
    livekit_room: Mapped[str | None] = mapped_column(String(120), default=None)

    # ── Detector state machine ─────────────────────────────────────
    # See `src/detectors/state_machine.py`. `closed` skips the LLM call;
    # `no_people_count` increments each tick with no people + no motion
    # and resets to 0 when activity returns.
    state: Mapped[str] = mapped_column(String(8), default="open")
    no_people_count: Mapped[int] = mapped_column(Integer, default=0)
    # DB column is NOT NULL with `default now()`. SQLAlchemy's
    # dataclass mode requires a Python-side default for non-trailing
    # columns, so we type as `datetime | None` and rely on the DB to
    # fill in. All current call sites pass `state_changed_at=now()`
    # explicitly; this is the safety net.
    state_changed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        server_default=text("now()"),
        default=None,
    )
    close_threshold: Mapped[int] = mapped_column(Integer, default=6)
    battery_percent: Mapped[int | None] = mapped_column(Integer, default=None)
