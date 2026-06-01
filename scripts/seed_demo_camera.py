"""Seed (or refresh) the App Review demo camera.

Creates one camera row holding the demo pairing code under the demo
caregiver's first patient, so an Apple reviewer can pair a fresh NemaraCam
install using NEMARA_DEMO_PAIRING_CODE. The backend pair endpoint
(src/api/pairing.py) special-cases this code: it pairs this row without
expiring or clearing the code, so it stays reusable.

Idempotent: re-running resets the row to an unpaired, pairable state
(api_key_hash = NULL, code + far-future expiry restored).

Run AFTER the prod schema migration + deploy (the Camera ORM model uses
`source_type`, which older prod schemas lack):

    NEMARA_DEMO_PAIRING_CODE=424242 uv run python scripts/seed_demo_camera.py

or explicitly:

    uv run python scripts/seed_demo_camera.py --email jane@patel.co --code 424242

Reads the same database the backend uses (NEMARA_DB_URL).
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sqlalchemy import func, select

from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.api.pairing import DEMO_CAMERA_IDENTITY
from src.settings import get_settings

DEFAULT_EMAIL = "jane@patel.co"
DEMO_ROOM = "Demo Room"


async def seed(email: str, code: str) -> None:
    sm = get_sessionmaker()
    async with sm() as session:
        cg = (
            await session.execute(
                select(Caregiver).where(Caregiver.email == email.lower())
            )
        ).scalar_one_or_none()
        if cg is None:
            raise SystemExit(f"no caregiver with email {email!r}")

        patient = (
            await session.execute(
                select(Patient)
                .where(Patient.caregiver_id == cg.id)
                .order_by(Patient.created_at)
                .limit(1)
            )
        ).scalar_one_or_none()
        if patient is None:
            raise SystemExit(
                f"caregiver {email!r} has no patient to attach the demo camera to"
            )

        # Idempotent: reuse the existing demo camera, matched by its stable
        # DEMO_CAMERA_IDENTITY (NOT pairing_code, which the app rotates).
        cam = (
            await session.execute(
                select(Camera).where(Camera.livekit_identity == DEMO_CAMERA_IDENTITY)
            )
        ).scalar_one_or_none()

        now = datetime.now(timezone.utc)
        far_future = now + timedelta(days=3650)
        if cam is None:
            max_display = (
                await session.execute(
                    select(func.coalesce(func.max(Camera.display_id), 0)).where(
                        Camera.patient_id == patient.id
                    )
                )
            ).scalar_one()
            cid = uuid4()
            cam = Camera(
                id=cid,
                patient_id=patient.id,
                display_id=max_display + 1,
                room=DEMO_ROOM,
                icon_name="tv",
                online=False,
                created_at=now,
                pairing_code=code,
                pairing_expires_at=far_future,
                api_key_hash=None,
                livekit_identity=DEMO_CAMERA_IDENTITY,
                livekit_room=f"patient_{patient.id}",
                state="open",
                no_people_count=0,
                state_changed_at=now,
                close_threshold=6,
                battery_percent=None,
                desired_state="disconnected",
                source_type="iphone",
                pi_stream_url=None,
            )
            session.add(cam)
            action = "created"
        else:
            # Reset to a pairable state so the demo code works again.
            cam.pairing_code = code
            cam.pairing_expires_at = far_future
            cam.api_key_hash = None
            action = "reset"

        await session.commit()
        print(
            f"demo camera {action}: id={cam.id} patient={patient.id} "
            f"({patient.preferred_name}) room={cam.room!r} code={code}"
        )


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=DEFAULT_EMAIL)
    ap.add_argument("--code", default=settings.demo_pairing_code or "424242")
    args = ap.parse_args()
    asyncio.run(seed(args.email, args.code))


if __name__ == "__main__":
    main()
