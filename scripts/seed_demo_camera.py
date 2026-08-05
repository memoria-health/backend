"""Seed (or refresh) the App Review demo camera.

Creates one camera row holding the demo pairing code under a DEDICATED
demo patient, so an Apple reviewer can pair a fresh NemaraCam install
using NEMARA_DEMO_PAIRING_CODE. The backend pair endpoint
(src/api/pairing.py) special-cases this code: it pairs this row without
expiring or clearing the code, so it stays reusable.

The dedicated patient matters. This script used to attach the demo
camera to the caregiver's FIRST REAL PATIENT, which made a six-digit,
never-expiring, reusable code into a live credential for a real family:
pairing returns that patient's id and LiveKit room, and the api_key it
hands back mints a room token whose grants include can_subscribe — the
other cameras in that person's home. The demo patient has no events and
no cameras but this one, so the code now reaches nothing real.

Idempotent: re-running resets the row to an unpaired, pairable state
(api_key_hash = NULL, code + far-future expiry restored), and re-points
a demo camera found attached to any other patient.

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
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import func, select

from src.api.pairing import DEMO_CAMERA_IDENTITY
from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.repos import access
from src.settings import get_settings

DEFAULT_EMAIL = "jane@patel.co"
DEMO_ROOM = "Demo Room"
# Stable marker for the dedicated demo patient. Matched on, so changing
# it strands the old row and seeds a second one.
DEMO_PATIENT_NAME = "Memoria Demo Patient"


def now_utc() -> datetime:
    return datetime.now(UTC)


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

        # Find-or-create a DEDICATED demo patient.
        #
        # This used to take the caregiver's FIRST REAL PATIENT (ordered by
        # created_at). That made the demo code a live credential for a real
        # family: /v1/cameras/pair with the demo code returns the api_key,
        # the real patient_id, and livekit_room = patient_<real id>; that
        # api_key then mints a LiveKit token via /v1/livekit/token whose
        # grants include can_subscribe, i.e. the other cameras in that
        # person's home. A six-digit, never-expiring, reusable code is not
        # a credential anyone should be able to guess into a real home.
        #
        # The demo patient carries no cameras but the demo one, and no
        # events, so pairing it reaches nothing real.
        patient = (
            await session.execute(
                select(Patient).where(
                    Patient.caregiver_id == cg.id,
                    Patient.full_name == DEMO_PATIENT_NAME,
                )
            )
        ).scalar_one_or_none()
        if patient is None:
            patient = Patient(
                id=uuid4(),
                caregiver_id=cg.id,
                full_name=DEMO_PATIENT_NAME,
                preferred_name="Demo",
                relation="other",
                state="calm",
                status_line="Demo account for App Review.",
                cameras_paused=False,
                created_at=now_utc(),
                updated_at=now_utc(),
            )
            session.add(patient)
            await session.flush()
            # Patient visibility is membership-scoped, so the demo
            # caregiver needs an owner row or the patient is invisible
            # to the very account a reviewer signs in with.
            await access.add_member(
                session, patient_id=patient.id, caregiver_id=cg.id, role="owner",
            )
            print(f"created dedicated demo patient {patient.id}")

        # Idempotent: reuse the existing demo camera, matched by its stable
        # DEMO_CAMERA_IDENTITY (NOT pairing_code, which the app rotates).
        cam = (
            await session.execute(
                select(Camera).where(Camera.livekit_identity == DEMO_CAMERA_IDENTITY)
            )
        ).scalar_one_or_none()

        # A demo camera left pointing at a REAL patient is the exact bug
        # this script now exists to prevent, so a re-run must repair that
        # state rather than quietly preserve it.
        if cam is not None and cam.patient_id != patient.id:
            print(
                f"WARNING: demo camera was attached to patient "
                f"{cam.patient_id} — re-pointing it at the demo patient "
                f"{patient.id}. Rotate NEMARA_DEMO_PAIRING_CODE as well: the "
                f"old code was a working credential for that patient's room."
            )
            cam.patient_id = patient.id
            cam.livekit_room = f"patient_{patient.id}"
            cam.api_key_hash = None

        now = datetime.now(UTC)
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


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})


def _db_host(url: str) -> str:
    """Host portion of NEMARA_DB_URL, without ever touching the password."""
    after_scheme = url.split("://", 1)[-1]
    authority = after_scheme.split("/", 1)[0]
    hostport = authority.rsplit("@", 1)[-1]  # drop user:pass@
    return hostport.rsplit(":", 1)[0].strip("[]")


def main() -> None:
    settings = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", default=DEFAULT_EMAIL)
    ap.add_argument("--code", default=settings.demo_pairing_code or "424242")
    ap.add_argument(
        "--remote",
        action="store_true",
        help="required to write to a non-local database (e.g. production)",
    )
    args = ap.parse_args()

    # This script writes wherever NEMARA_DB_URL happens to point, with no
    # signal about where that is. Running it against local left rows behind
    # that broke three tests; running it against production is what it is
    # FOR, and deserves more friction than an unlucky shell history.
    #
    # Same shape as the `supabase/.temp/project-ref` problem: a command
    # that silently targets whatever database the environment names. Print
    # the host (never the credentials) and make a remote write explicit.
    host = _db_host(settings.db_url)
    if host not in _LOCAL_HOSTS and not args.remote:
        raise SystemExit(
            f"refusing to seed: NEMARA_DB_URL points at {host!r}, not a local "
            f"database.\nThis writes a pairable camera row with a reusable "
            f"code. If that is genuinely what you want, re-run with --remote."
        )
    print(f"seeding against {host}")
    asyncio.run(seed(args.email, args.code))


if __name__ == "__main__":
    main()
