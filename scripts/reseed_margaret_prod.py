"""One-off ops script: reseed ONLY the demo caregiver's "Margaret O'Brien"
patient, leaving their OTHER patients untouched.

WHY THIS EXISTS (and why it does NOT reuse seed.py):
  * scripts/seed.py wipes ALL patients owned by the demo caregiver. In
    prod that demo account owns THREE (Joe Thomas, Margaret O'Brien, Soph).
  * seed.py resolves the demo user via the GoTrue admin REST API with a
    `?email=` filter. That filter is IGNORED by the admin LIST endpoint in
    prod — it returns the FIRST user of the whole list, which is an
    UNRELATED REAL account. Running seed.py against prod would therefore
    operate on the wrong caregiver and destroy real data.

So this script:
  1. Resolves the demo caregiver by a DIRECT DB query on auth.users.email
     (the DB does not lie), and PINS it to the verified id as a guard.
  2. Finds the SINGLE "Margaret O'Brien" under that caregiver. Aborts
     unless exactly one.
  3. Deletes only that patient (DB cascade removes her cameras / detectors
     / events) and recreates her clean — identical spec to seed.py.

Joe Thomas + Soph are never touched; no other account is ever in scope.

Run against PROD with prod env injected:
    PYTHONPATH=. railway run uv run python scripts/reseed_margaret_prod.py
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, text

from src.db.camera import Camera
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services import onboarding
from src.settings import get_settings

DEMO_EMAIL = "jane@patel.co"
# Verified via direct DB inspection: jane@patel.co -> this caregiver, which
# owns Margaret O'Brien / Joe Thomas / Soph. Pinned so a lookup that ever
# resolves to a different account aborts instead of touching real data.
EXPECTED_CAREGIVER_ID = "9d06193c-e7cd-4657-bed2-100258d199a3"
TARGET_FULL_NAME = "Margaret O'Brien"


def _roster(patients) -> list[str]:
    return [f"{p.full_name} ({p.preferred_name})" for p in patients]


async def main() -> None:
    settings = get_settings()
    print(f"[reseed] supabase_url={settings.supabase_url}")

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        # Resolve the demo caregiver by the DB, NOT the admin API.
        row = (
            await session.execute(
                text("select id from auth.users where email = :email"),
                {"email": DEMO_EMAIL},
            )
        ).first()
        if row is None:
            raise SystemExit(f"ABORT: no auth user with email {DEMO_EMAIL}")
        caregiver_id = UUID(str(row[0]))
        print(f"[reseed] demo caregiver {DEMO_EMAIL} -> {caregiver_id}")

        if str(caregiver_id) != EXPECTED_CAREGIVER_ID:
            raise SystemExit(
                f"ABORT: resolved caregiver {caregiver_id} != expected "
                f"{EXPECTED_CAREGIVER_ID} — refusing to touch an unverified account."
            )

        before = (
            await session.execute(
                select(Patient).where(Patient.caregiver_id == caregiver_id)
            )
        ).scalars().all()
        print(f"[reseed] BEFORE — caregiver owns {len(before)}: {_roster(before)}")

        targets = [p for p in before if p.full_name == TARGET_FULL_NAME]
        if len(targets) != 1:
            raise SystemExit(
                f"ABORT: expected exactly 1 '{TARGET_FULL_NAME}', found {len(targets)}."
            )
        old = targets[0]
        keep = [p.full_name for p in before if p.id != old.id]
        print(f"[reseed] deleting ONLY Margaret {old.id}; preserving {keep}")
        await session.delete(old)
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
            state="calm",
            status_line="All quiet",
            last_event_at=None,
            cameras_paused=False,
            created_at=now,
            updated_at=now,
        )
        session.add(patient)
        await session.flush()

        await onboarding.seed_detectors_for_patient(session, patient.id)

        for display_id, room, icon in [
            (1, "Living room", "tv"),
            (2, "Kitchen", "fork.knife"),
            (3, "Bedroom", "bed.double"),
            (4, "Hallway", "door.left.hand.open"),
        ]:
            session.add(
                Camera(
                    id=uuid4(),
                    patient_id=patient.id,
                    caregiver_id=patient.caregiver_id,
                    display_id=display_id,
                    room=room,
                    icon_name=icon,
                    online=True,
                    created_at=now,
                )
            )

        await session.commit()

        after = (
            await session.execute(
                select(Patient).where(Patient.caregiver_id == caregiver_id)
            )
        ).scalars().all()
        print(f"[reseed] AFTER  — caregiver owns {len(after)}: {_roster(after)}")
        print(f"[reseed] new Margaret patient id: {patient.id}")
        print("[reseed] done.")


if __name__ == "__main__":
    asyncio.run(main())
