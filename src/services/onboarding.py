"""Caregiver and patient onboarding business logic.

Caregiver:
- `provision_caregiver` — idempotent first-login upsert (POST /v1/me).
- `update_caregiver` — partial PATCH path.
- `to_response` — ORM → wire shape.

Patient:
- `seed_detectors_for_patient` — writes the four preset detectors
  (Falls / Agitation / Wandering / Medication adherence) right after a
  new patient row lands. Same shape the iOS mock store uses today.
"""

from collections.abc import Sequence
from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver
from src.db.detector import Detector
from src.repos import caregiver_repo, detector_repo
from src.schemas.auth import SupabasePrincipal
from src.schemas.caregiver import (
    CaregiverResponse,
    CreateCaregiverRequest,
    UpdateCaregiverRequest,
)


def to_response(cg: Caregiver) -> CaregiverResponse:
    """Adapt the ORM row to the wire shape. Wire field names are
    snake_case (Pydantic default); iOS maps to camelCase via explicit
    `CodingKeys` on each Codable model.
    """
    return CaregiverResponse(
        id=cg.id,
        display_name=cg.display_name,
        email=cg.email,
        phone=cg.phone,
        created_at=cg.created_at,
        updated_at=cg.updated_at,
    )


async def provision_caregiver(
    session: AsyncSession,
    auth: SupabasePrincipal,
    body: CreateCaregiverRequest,
) -> Caregiver:
    """Idempotently provision a caregiver row for the JWT principal.

    If a row already exists for `auth.id`, return it unchanged — the
    request body is intentionally ignored in that case (matches range's
    expectation of a safe re-call).
    """
    existing = await caregiver_repo.find_by_id(session, auth.id)
    if existing is not None:
        return existing

    now = datetime.now(timezone.utc)
    cg = Caregiver(
        id=auth.id,
        display_name=body.display_name,
        email=auth.email,                # source of truth = JWT, not body
        created_at=now,
        updated_at=now,
        phone=body.phone,
    )
    return await caregiver_repo.insert(session, cg)


async def update_caregiver(
    session: AsyncSession,
    auth: SupabasePrincipal,
    body: UpdateCaregiverRequest,
) -> Caregiver | None:
    """Apply a partial update. Returns None if the caregiver row is
    missing (the caller should map this to a 404).
    """
    # `exclude_unset=True` distinguishes "field omitted" (don't touch)
    # from explicit `null` (clear the column). `phone=null` clears phone;
    # omitting `phone` leaves it as-is.
    fields = body.model_dump(exclude_unset=True)
    return await caregiver_repo.update_partial(session, auth.id, fields=fields)


# ─── Patient onboarding ─────────────────────────────────────────


# Field shapes mirror MockData.swift's preset detectors 1:1 so the
# iOS UI doesn't need a different code path for "freshly server-seeded"
# vs "freshly mock-store-seeded".
_PRESET_DETECTORS: tuple[dict[str, str], ...] = (
    {
        "name": "Falls",
        "icon_name": "figure.fall",
        "description": "Sudden fall events across all rooms.",
        "category": "safety",
        "shape": "point_in_time",
        "where_scope": "ALL ROOMS",
        "when_scope": "ALWAYS",
    },
    {
        "name": "Agitation",
        "icon_name": "figure.walk.motion",
        "description": "Pacing or restless body language sustained over time.",
        "category": "safety",
        "shape": "state",
        "where_scope": "ALL ROOMS",
        "when_scope": "ALWAYS",
    },
    {
        "name": "Wandering",
        "icon_name": "door.left.hand.open",
        "description": "Main door opening — front entry only.",
        "category": "safety",
        "shape": "point_in_time",
        "where_scope": "HALLWAY",
        "when_scope": "ALWAYS",
    },
    {
        "name": "Medication adherence",
        "icon_name": "pills",
        "description": "Confirms the morning and evening pill routine at the kitchen counter.",
        "category": "routine",
        "shape": "point_in_time",
        "where_scope": "KITCHEN",
        "when_scope": "8 AM, 8 PM ±1H",
    },
)


async def seed_detectors_for_patient(
    session: AsyncSession, patient_id: UUID
) -> Sequence[Detector]:
    """Create the four preset detectors for a freshly-created patient.

    Caller owns the transaction (no commit here). Detectors land in
    insertion order, which becomes the iOS list order.
    """
    now = datetime.now(timezone.utc)
    inserted: list[Detector] = []
    for preset in _PRESET_DETECTORS:
        det = Detector(
            id=uuid4(),
            patient_id=patient_id,
            name=preset["name"],
            icon_name=preset["icon_name"],
            description=preset["description"],
            category=preset["category"],
            source="preset",
            shape=preset["shape"],
            where_scope=preset["where_scope"],
            when_scope=preset["when_scope"],
            sensitivity="standard",
            is_on=True,
            created_at=now,
            updated_at=now,
            custom_prompt=None,
        )
        inserted.append(await detector_repo.insert(session, det))
    return inserted
