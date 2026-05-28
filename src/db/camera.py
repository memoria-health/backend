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

    # ── User-facing "off" toggle ─────────────────────────────────
    # Per-camera disable controlled from the web dashboard / iOS
    # caregiver app. When true, the detector worker stays subscribed
    # to the LiveKit room (keeps the frame buffer warm) but the
    # per-tick loop short-circuits BEFORE the LLM call — see
    # `services/detection/camera_ingest.py:_run_tick`. Web hides
    # disabled cameras from the live grid + analyses feed.
    disabled: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )

    # ── AI paused (LLM-only, video still streams) ────────────────
    # Lighter than `disabled`: when true, the tick still short-circuits
    # before the LLM call (synthetic "skipped" row with
    # skipped_reason='ai_paused') but the live tile STAYS VISIBLE in
    # the grid and video keeps streaming. Lets the caregiver watch a
    # room without spending on Gemini calls.
    ai_paused: Mapped[bool] = mapped_column(
        Boolean, server_default=text("false"), default=False,
    )

    # ── Desired connection state (privacy / battery / cost) ─────────
    # Stronger than `disabled`: when 'disconnected', the wall phone
    # itself LEAVES the LiveKit room. No video bytes are encoded,
    # uploaded, or seen by LiveKit Cloud. The phone polls
    # /v1/cameras/me/desired-state every 1s and reconciles by calling
    # room.connect() / room.disconnect() to match. Reconcile latency
    # ~ 1-2s after the caregiver flips this back to 'connected'.
    # Two-value enum enforced at the DB layer via CHECK constraint;
    # Pydantic Literal also enforces at the API boundary.
    desired_state: Mapped[str] = mapped_column(
        String(16), server_default=text("'connected'"), default="connected",
    )

    # ── Source type (iPhone vs Pi) ───────────────────────────────────
    # Discriminates the ingest path. 'iphone' rows are paired via the
    # existing /v1/cameras/pair flow and publish directly to LiveKit.
    # 'pi' rows are provisioned via /v1/cameras/pi-provision (Phase 4),
    # carry a `pi_stream_url`, and have their LiveKit publishing
    # handled by the backend's pi_ingest worker process. From the
    # detection pipeline's perspective both look identical.
    source_type: Mapped[str] = mapped_column(
        String(16), server_default=text("'iphone'"), default="iphone",
    )

    # ── Pi MJPEG endpoint (vestigial) ────────────────────────────────
    # Was the URL the backend pulled MJPEG from in the pull-mode design.
    # Push refactor (Pi POSTs frames outbound to /v1/cameras/me/frames)
    # made this column unused; kept nullable for backwards compat and
    # for diagnostic queries. NOT referenced by the relay path.
    pi_stream_url: Mapped[str | None] = mapped_column(String(2048), default=None)

    # ── Pi PCM audio endpoint (vestigial) ────────────────────────────
    # Same status as pi_stream_url after the push refactor. Kept
    # nullable, unused by the ingest path.
    pi_audio_url: Mapped[str | None] = mapped_column(String(2048), default=None)

    # ── Ingest bearer token (push auth) ──────────────────────────────
    # Hashed bearer token the Pi presents on
    # POST /v1/cameras/me/frames. Generated at provision time, the
    # plaintext value is returned ONCE in the next announce response,
    # never stored or logged server-side outside this hash. SHA-256
    # hex digest (64 chars). Required for source_type='pi', forbidden
    # for source_type='iphone' (iPhones use their own per-session
    # LiveKit access tokens minted via /v1/livekit/token).
    ingest_token_hash: Mapped[str | None] = mapped_column(String(64), default=None)
    ingest_token_created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None,
    )
