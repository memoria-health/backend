"""Local-dev seed script.

Populates the local Supabase + Postgres with a single demo caregiver,
one patient (with the 4 auto-seeded preset detectors), 4 cameras
covering the typical home layout, and 7 events spanning the iOS Events
tab's state matrix (unopened safety, opened safety, unopened routine,
reviewed safety, etc).

Run via `make db-reset` (which calls this after the SQL schema apply)
or directly: `uv run python scripts/seed.py`.

Wire shape:
1. Create a Supabase auth user via the GoTrue admin REST API. We need
   service_role auth to do this; the local-dev anon key won't work.
2. Provision the caregiver row by reusing `services.onboarding`. This
   keeps the seed in lock-step with what real signups do.
3. Insert patient + cameras + detectors + events directly through the
   ORM (bypassing the route handlers — there's no JWT to construct
   here). Auto-seed-on-patient-create still runs because we go through
   the service layer.

This script is idempotent: if the demo caregiver email already exists,
we look up the existing user, wipe the demo patient (cascades to
cameras/detectors/events), and re-create. Safe to call repeatedly.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from uuid import UUID, uuid4

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.schemas.auth import SupabasePrincipal
from src.schemas.caregiver import CreateCaregiverRequest
from src.services import onboarding
from src.settings import get_settings

# ─── demo identity ───────────────────────────────────────────────
DEMO_EMAIL = "jane@patel.co"
DEMO_PASSWORD = "Mar9aret!2024"   # local-dev only; never used in prod
DEMO_DISPLAY_NAME = "Jane Patel"


async def upsert_auth_user(*, email: str, password: str) -> UUID:
    """Create the Supabase auth user via the admin REST API, or return
    the existing user's id if the email already exists.

    Uses the service_role JWT — the only one allowed to call admin
    endpoints. The local-dev key is fine; production keys never go
    into this script.
    """
    settings = get_settings()
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Content-Type": "application/json",
    }
    admin_url = f"{settings.supabase_url.rstrip('/')}/auth/v1/admin/users"

    async with httpx.AsyncClient(timeout=10.0) as client:
        # GoTrue's admin endpoint is `?email=` for lookup; falls back to
        # create otherwise.
        list_resp = await client.get(admin_url, headers=headers, params={"email": email})
        if list_resp.status_code == 200:
            users = list_resp.json().get("users", [])
            if users:
                return UUID(users[0]["id"])

        create_resp = await client.post(
            admin_url,
            headers=headers,
            json={
                "email": email,
                "password": password,
                "email_confirm": True,        # skip the confirmation email loop locally
            },
        )
        create_resp.raise_for_status()
        return UUID(create_resp.json()["id"])


# ─── demo patient + child rows ───────────────────────────────────


async def seed_patient(session: AsyncSession, caregiver_id: UUID) -> Patient:
    """Provision a patient + cameras + detectors + events for the demo
    caregiver. Idempotent: deletes any prior patient owned by this
    caregiver first (cascades to children).
    """
    # Wipe prior demo patient(s) for this caregiver. Cascades clean up
    # cameras/detectors/events via DB-level ON DELETE CASCADE.
    existing = await session.execute(
        select(Patient).where(Patient.caregiver_id == caregiver_id)
    )
    for p in existing.scalars().all():
        await session.delete(p)
    await session.flush()

    now = datetime.now(timezone.utc)
    patient = Patient(
        id=uuid4(),
        caregiver_id=caregiver_id,
        full_name="Margaret O'Brien",
        preferred_name="Maggie",
        date_of_birth=date(1951, 3, 18),
        gender="Female",
        relation="parent",
        home_address="84 Birch Lane, Cambridge MA",
        state="critical",
        status_line="Possible fall, living room",
        last_event_at=now - timedelta(minutes=42),
        battery_percent=82,
        cameras_paused=False,
        created_at=now,
        updated_at=now,
    )
    session.add(patient)
    await session.flush()

    # Detectors — call the same service the POST /v1/patients route uses.
    await onboarding.seed_detectors_for_patient(session, patient.id)

    # Cameras — four typical home rooms.
    cam_specs = [
        (1, "Living room", "tv"),
        (2, "Kitchen",     "fork.knife"),
        (3, "Bedroom",     "bed.double"),
        (4, "Hallway",     "door.left.hand.open"),
    ]
    for display_id, room, icon in cam_specs:
        session.add(
            Camera(
                id=uuid4(),
                patient_id=patient.id,
                display_id=display_id,
                room=room,
                icon_name=icon,
                online=True,
                last_seen_at=now,
                created_at=now,
            )
        )

    # Events — covers the iOS Events tab's state matrix.
    event_specs = [
        # (category, source, icon, title, location, mins_ago, has_clip, narrative, opened, reviewed, choice, notes)
        ("safety",  "preset", "figure.fall",         "Possible fall",     "Living room",     42,
         True,
         "A sharp downward motion at 14:32, followed by three minutes of stillness on the floor near the coffee table. No spoken response when we tried room audio at 14:33.",
         False, False, None, None),
        ("safety",  "preset", "figure.walk.motion",  "Agitation episode", "Bedroom",         120,
         True,
         "Maggie paced from the doorway to the window for about four minutes around 13:10. Voice raised twice without clear words. Settled with the radio.",
         True,  False, None, None),
        ("routine", "preset", "pills",               "Medication",        "Kitchen",         420,
         False,
         "Maggie took her morning pills at 08:05, with tea, sitting at the kitchen counter — exactly her usual pattern.",
         False, False, None, None),
        ("safety",  "custom", "moon.stars",          "Late kitchen visit","Kitchen",         780,
         True,
         "Brief light in the kitchen at 02:30. Maggie got a glass of water and returned to bed within four minutes.",
         True,  True,  "correct", None),
        ("safety",  "preset", "door.left.hand.open", "Front door opened", "Hallway",         1380,
         True,
         "Front door opened at 16:15. Maggie greeted someone (the home aide on her usual Tuesday visit), they came in, door closed within 20 seconds.",
         True,  True,  "correct", None),
        ("routine", "preset", "pills",               "Medication",        "Kitchen",         1440,
         False,
         "Evening pills taken at 20:05.",
         True,  True,  "correct", None),
        ("safety",  "preset", "figure.walk.motion",  "Agitation episode", "Living room",     4320,
         True,
         "A brief pacing episode around the news hour — calmed within five minutes.",
         True,  True,  "correct", None),
    ]
    for (
        category, source, icon, title, location, mins_ago, has_clip,
        narrative, opened, reviewed, choice, notes,
    ) in event_specs:
        occurred = now - timedelta(minutes=mins_ago)
        session.add(
            Event(
                id=uuid4(),
                patient_id=patient.id,
                detector_id=None,
                camera_id=None,
                category=category,
                source=source,
                icon_name=icon,
                title=title,
                location=location,
                occurred_at=occurred,
                has_clip=has_clip,
                narrative=narrative,
                opened=opened,
                reviewed=reviewed,
                review_choice=choice,
                review_notes=notes,
                reviewed_at=occurred if reviewed else None,
                created_at=occurred,
            )
        )

    return patient


# ─── entry point ─────────────────────────────────────────────────


async def main() -> None:
    print(f"[seed] upserting auth user {DEMO_EMAIL}...")
    user_id = await upsert_auth_user(email=DEMO_EMAIL, password=DEMO_PASSWORD)
    print(f"[seed] auth user id: {user_id}")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        # Provision caregiver row through the same service the API uses.
        principal = SupabasePrincipal(id=user_id, email=DEMO_EMAIL)
        await onboarding.provision_caregiver(
            session,
            principal,
            CreateCaregiverRequest(display_name=DEMO_DISPLAY_NAME),
        )

        # Patient + cameras + detectors + events.
        patient = await seed_patient(session, user_id)
        await session.commit()
        print(f"[seed] patient {patient.id} ({patient.preferred_name}) provisioned")
        print("[seed] done.")


if __name__ == "__main__":
    asyncio.run(main())
