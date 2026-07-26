"""Caregiver and patient onboarding business logic.

Caregiver:
- `provision_caregiver` — idempotent first-login upsert (POST /v1/me).
- `update_caregiver` — partial PATCH path.
- `to_response` — ORM → wire shape.

Patient:
- `seed_detectors_for_patient` — writes the seeded preset detectors right after
  a new patient row lands. The set is the `seeded` subset of the detector
  catalog (`src/services/detection/catalog.py`) for the patient's cohort:
  under-18 → child set (Falls / Climbing / Distress); 18+ or unknown birthday →
  older-adult set (Falls / Agitation / Wandering / Medication adherence). The
  catalog is the single source of truth — the detection-library picker offers
  these SAME entries (pre-checked, listed under "More detectors" — only Falls is
  in the "Recommended" group), so onboarding and the library can't disagree.
"""

import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
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
from src.services.detection import catalog, catalog_v2

_log = logging.getLogger(__name__)


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
        onboarded_web=cg.onboarded_web,
        timezone=cg.timezone,
        show_live_analyses=cg.show_live_analyses,
        tos_accepted_at=cg.tos_accepted_at,
        tos_version=cg.tos_version,
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
        # iOS gates account creation on ToS acceptance, so a version arrives
        # here → stamp the acceptance time. Web omits it (accepts later via
        # the PATCH gate), so the row is born un-accepted and the web gate
        # blocks the app until it PATCHes a version.
        tos_version=body.tos_version,
        tos_accepted_at=now if body.tos_version else None,
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
    # ToS acceptance: when the client sends the version it accepted (via the
    # web accept-terms gate), stamp the acceptance time server-side. Mirrors
    # the patient data-collection consent (data_collection_decided_at); the
    # timestamp is never client-writable.
    if "tos_version" in fields:
        fields["tos_accepted_at"] = datetime.now(timezone.utc)
    return await caregiver_repo.update_partial(session, auth.id, fields=fields)


# ─── Patient onboarding ─────────────────────────────────────────


async def seed_detectors_for_patient(
    session: AsyncSession,
    patient_id: UUID,
    date_of_birth: date | None = None,
) -> Sequence[Detector]:
    """Create the recommended preset detectors for a freshly-created patient.

    The set is `catalog.seeded_for_cohort(...)` for the patient's cohort
    (under 18 → child set; 18+ or unknown birthday → adult set). An absent
    `date_of_birth` falls back to the adult set, preserving prior behaviour.

    Caller owns the transaction (no commit here). Detectors land in catalog
    order, which becomes the iOS list order. Each row carries the runtime
    contract (identifier / prompt_fragment / schema_json) the detector pipeline
    reads when building Gemini/OpenAI calls.
    """
    minor = catalog.is_minor(date_of_birth)
    entries = catalog.seeded_for_cohort(minor)
    if not entries:
        # The catalog should always have at least the seeded set; an empty
        # result means a malformed catalog, not a normal state. Fail loud.
        raise RuntimeError(
            f"[onboarding] catalog returned no seeded detectors for "
            f"patient {patient_id} (minor={minor}); refusing to seed nothing"
        )
    # Stamp each preset with a microsecond-apart `created_at` so the
    # `ORDER BY created_at ASC, id ASC` in `detector_repo.list_for_patient`
    # returns them in insertion order deterministically. Without this, rows
    # share `created_at` and Postgres's heap-scan order can drift after
    # autovacuum (caused intermittent test flakes).
    base = datetime.now(timezone.utc)
    inserted: list[Detector] = []
    for i, e in enumerate(entries):
        now = base + timedelta(microseconds=i)
        det = Detector(
            id=uuid4(),
            patient_id=patient_id,
            name=e.name,
            icon_name=e.icon_name,
            description=e.description,
            category=e.category,
            source="preset",
            shape=e.shape,
            where_scope=e.where_scope,
            when_scope=e.when_scope,
            sensitivity="standard",
            is_on=True,
            created_at=now,
            updated_at=now,
            custom_prompt=None,
            identifier=e.key,
            prompt_fragment=e.prompt_fragment,
            schema_json=e.schema_json,
            always_on=False,
            **catalog_v2.row_kwargs(e.key),
        )
        inserted.append(await detector_repo.insert(session, det))
    _log.info(
        "[onboarding] seeded %d preset detectors for patient %s (minor=%s): %s",
        len(inserted), patient_id, minor, [e.key for e in entries],
    )
    return inserted
