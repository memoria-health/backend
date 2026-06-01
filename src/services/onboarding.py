"""Caregiver and patient onboarding business logic.

Caregiver:
- `provision_caregiver` — idempotent first-login upsert (POST /v1/me).
- `update_caregiver` — partial PATCH path.
- `to_response` — ORM → wire shape.

Patient:
- `seed_detectors_for_patient` — writes the preset detectors right after a
  new patient row lands. The set is age-based: under-18 patients get the
  child set (Falls / Climbing / Distress); 18+ or unknown birthday get the
  older-adult set (Falls / Agitation / Wandering / Medication adherence).
"""

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


# Preset prompt fragments + JSON-schema field definitions.
#
# `fall` + `wandering` are ported verbatim from
# `nemara-temp-dashboard/lib/detectors.ts` — those prompts have been
# tuned. `agitation` + `medication` are new (dashboard didn't have
# them); the structure mirrors fall: `<id>_detected: bool`,
# `<id>_confidence: 0–1`, `<id>_reasoning: short sentence`.
#
# `people_present` is NOT seeded as a per-patient row. It's always
# implicitly included by the detector pipeline (see
# `src/detectors/prompt.py:BASE_PROMPT`). Keeping it out of the
# per-patient detectors table prevents an "always-on" entry from
# cluttering the caregiver-facing detector list.

_FALL_PROMPT_FRAGMENT = (
    "fall_detected: boolean. TRUE if a person falls to the ground in the "
    "clip — they collapse, slip, trip, stumble, sag, or drop and end up down "
    "on the floor. If someone goes down to the ground, that's a fall. FALSE "
    "otherwise: they stay upright, walk around or out of the room normally, "
    "or deliberately and slowly sit, lie down, or kneel.\n\n"
    "The camera is mounted high near the ceiling, so a person walking toward "
    "it and shrinking out the bottom of the frame is passing BENEATH the "
    "camera (normal navigation), not a fall.\n\n"
    "fall_confidence: number 0–1. 0 = no fall. 0.7+ = a person clearly went "
    "down to the floor.\n\n"
    "fall_reasoning: 1 short sentence describing what you saw. Always present."
)

_WANDERING_PROMPT_FRAGMENT = (
    "wandering_detected: boolean. TRUE only if, within this 15-second clip, "
    "you SEE a person walk through the RIGHT-SIDE doorway and LEAVE the room — "
    "they were inside the room earlier in the clip, and by the end of the clip "
    "they are no longer inside it. Only OUTBOUND exits through the right door count.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - The person exits through the LEFT-SIDE doorway. The left door leads "
    "to another room inside the house and is NOT a wandering exit. Only the "
    "right door counts.\n"
    "  - The person was outside the room and walked IN. Entering doesn't "
    "count, only leaving does.\n"
    "  - The person only approaches the door, touches it, opens it, or looks "
    "toward it, but does not actually walk through. Intent doesn't count.\n"
    "  - The person walks out of camera frame in a direction that isn't a "
    "clearly visible doorway (e.g. steps behind furniture or off to the side).\n"
    "  - The person leaves and returns within the same 15-second clip.\n"
    "  - No person is visible at any point.\n"
    "\n"
    "It only counts when you literally see them, starting inside the room, "
    "open / step through the RIGHT door and stay out. Exits via the left door "
    "(internal room-to-room) are FALSE.\n"
    "\n"
    "wandering_confidence: number 0–1. 0 = no exit seen. 0.7+ = clearly "
    "walked through the doorway and did not come back. Anything in between "
    "for ambiguous cases.\n"
    "wandering_reasoning: one short sentence describing exactly what you "
    "observed — for example: 'person opened the front door and walked "
    "through at 8s', 'person entered the room from outside, did not leave "
    "(not wandering)', 'person approached the door but did not open it', "
    "'no person visible'."
)

_AGITATION_PROMPT_FRAGMENT = (
    "agitation_detected: boolean. True if a person displays sustained restless body language over the clip — pacing back and forth, fidgeting with clothing/objects in a repetitive nervous way, repeatedly standing/sitting, wringing hands, sharp jerky gestures, or other movement patterns suggesting distress/anxiety. The behavior must be SUSTAINED across multiple seconds of the clip — a single moment of motion is not agitation. "
    "Audio is supporting evidence only: agitated vocalizations (shouting, sustained crying, distressed or repetitive verbalizations, loud groans) can corroborate a visual judgment of agitation, but audio alone is NEVER sufficient to set this TRUE — there must also be visible restless body language in the clip. If the audio sounds distressed but the person is visually calm, off-camera, or not visible at all, set FALSE. "
    "Set FALSE if the person is moving normally (walking through, doing a task with purpose, sitting still, sleeping), or if no person is visible.\n"
    "agitation_confidence: number 0–1. 0 = clearly calm/normal/no person. 0.3–0.6 = some restless motion but ambiguous (could be normal activity). 0.7+ = clear sustained agitation; raise within this band when distressed audio also corroborates the visual signal.\n"
    "agitation_reasoning: 1 short sentence explaining your judgment, noting briefly whether audio reinforced or contradicted the visual (e.g. 'person paced from doorway to window four times across the clip, audio also muttering anxiously', 'person sat reading throughout — calm, no audio cues', 'audible shouting but no person visible — not agitation', 'no person visible')."
)

_MEDICATION_PROMPT_FRAGMENT = (
    "medication_taken: boolean. True if the clip shows clear medication-taking behavior — picking up a pill bottle / blister pack / weekly pill organizer, placing a pill in the mouth, drinking water immediately after, OR sitting at the kitchen counter with a pill organizer open. Set FALSE if no medication-related interaction is visible, the person is just in the kitchen for other reasons (cooking, drinks), or no person is visible.\n"
    "medication_confidence: number 0–1. 0 = no medication-related behavior visible. 0.3–0.6 = ambiguous (e.g. person at kitchen counter but pills not clearly visible). 0.7+ = clear medication-taking sequence.\n"
    "medication_reasoning: 1 short sentence explaining your judgment (e.g. 'person opened weekly pill organizer and swallowed pills with water', 'person at kitchen counter making tea, no pills visible', 'no person visible')."
)


# ── Child-specific prompts (used for under-18 patients) ──────────────
# Falls is reused verbatim from the adult set (children fall too, same
# camera geometry). Climbing + Distress are new and child-oriented. Their
# wording is a sensible starting point — detection accuracy should be
# field-tuned, same as any prompt.

_CLIMBING_PROMPT_FRAGMENT = (
    "climbing_detected: boolean. TRUE if a child is climbing onto something "
    "they shouldn't — pulling up onto furniture (sofa back, table, kitchen "
    "counter, shelf, windowsill, railing), scaling a crib rail or safety gate, "
    "or already standing / perched on an elevated surface in a way that risks "
    "a fall. The climbing motion OR the child clearly perched on the elevated "
    "surface must be visible in the clip.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child is sitting, standing, or playing normally on the floor;\n"
    "  - an adult is holding or lifting the child;\n"
    "  - the child is seated in a high chair, crib, or stroller in normal use;\n"
    "  - no child is visible;\n"
    "  - you only hear sounds without seeing the climbing.\n"
    "\n"
    "climbing_confidence: number 0–1. 0 = no climbing. 0.3–0.6 = ambiguous "
    "(child near furniture but not clearly climbing). 0.7+ = clearly climbing "
    "onto or perched on an elevated surface.\n"
    "climbing_reasoning: 1 short sentence describing what you saw (e.g. "
    "'toddler pulled up onto the coffee table at 5s', 'child sitting on the "
    "floor playing — no climbing', 'no child visible')."
)

_DISTRESS_PROMPT_FRAGMENT = (
    "distress_detected: boolean. TRUE if a child shows sustained distress "
    "across the clip — crying, wailing, or visibly upset (scrunched / red "
    "face, flailing, reaching out while crying) for multiple seconds. Audio "
    "(crying / wailing) is SUPPORTING evidence that can corroborate a visual "
    "judgment, but audio alone is NEVER sufficient — if the child is visually "
    "calm, off-camera, or not visible, set FALSE.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child is calm, playing, laughing, or sleeping;\n"
    "  - the upset is a single brief moment rather than sustained;\n"
    "  - no child is visible.\n"
    "\n"
    "distress_confidence: number 0–1. 0 = calm / no child. 0.3–0.6 = some "
    "fussing but ambiguous. 0.7+ = clear sustained distress; raise within the "
    "band when crying audio corroborates the visual signal.\n"
    "distress_reasoning: 1 short sentence (e.g. 'child crying and reaching out "
    "for ~8s, audio wailing', 'child playing happily — calm', 'audible fussing "
    "but no child visible — not distress', 'no child visible')."
)


def _detection_schema(identifier: str) -> dict[str, dict[str, str]]:
    """Per-detector JSON-schema field map for detectors whose schema
    follows the standard `<id>_detected / _confidence / _reasoning`
    shape. Used by 3 of the 4 preset prompts.
    """
    return {
        f"{identifier}_detected":   {"type": "boolean"},
        f"{identifier}_confidence": {"type": "number"},
        f"{identifier}_reasoning":  {"type": "string"},
    }


_PRESET_DETECTORS_ADULT: tuple[dict[str, object], ...] = (
    {
        "identifier":      "fall",
        "name":            "Falls",
        "icon_name":       "figure.fall",
        "description":     "Sudden fall events across all rooms.",
        "category":        "safety",
        "shape":           "point_in_time",
        "where_scope":     "ALL ROOMS",
        "when_scope":      "ALWAYS",
        "prompt_fragment": _FALL_PROMPT_FRAGMENT,
        "schema_json":     _detection_schema("fall"),
    },
    {
        "identifier":      "agitation",
        "name":            "Agitation",
        "icon_name":       "figure.walk.motion",
        "description":     "Pacing or restless body language sustained over time.",
        "category":        "safety",
        "shape":           "state",
        "where_scope":     "ALL ROOMS",
        "when_scope":      "ALWAYS",
        "prompt_fragment": _AGITATION_PROMPT_FRAGMENT,
        "schema_json":     _detection_schema("agitation"),
    },
    {
        "identifier":      "wandering",
        "name":            "Wandering",
        "icon_name":       "door.left.hand.open",
        "description":     "Main door opening — front entry only.",
        "category":        "safety",
        "shape":           "point_in_time",
        "where_scope":     "HALLWAY",
        "when_scope":      "ALWAYS",
        "prompt_fragment": _WANDERING_PROMPT_FRAGMENT,
        "schema_json":     _detection_schema("wandering"),
    },
    {
        "identifier":      "medication",
        "name":            "Medication adherence",
        "icon_name":       "pills",
        "description":     "Confirms the morning and evening pill routine at the kitchen counter.",
        "category":        "routine",
        "shape":           "point_in_time",
        "where_scope":     "KITCHEN",
        "when_scope":      "8 AM, 8 PM ±1H",
        "prompt_fragment": _MEDICATION_PROMPT_FRAGMENT,
        # medication uses `medication_taken` not `_detected`; explicit
        # schema rather than the helper.
        "schema_json": {
            "medication_taken":      {"type": "boolean"},
            "medication_confidence": {"type": "number"},
            "medication_reasoning":  {"type": "string"},
        },
    },
)


# Child set (under-18 patients). Falls reused verbatim; Climbing + Distress
# are child-oriented. Distinct identifiers so they never collide with the
# adult set under the per-patient unique-identifier constraint.
_PRESET_DETECTORS_CHILD: tuple[dict[str, object], ...] = (
    {
        "identifier":      "fall",
        "name":            "Falls",
        "icon_name":       "figure.fall",
        "description":     "Sudden fall events across all rooms.",
        "category":        "safety",
        "shape":           "point_in_time",
        "where_scope":     "ALL ROOMS",
        "when_scope":      "ALWAYS",
        "prompt_fragment": _FALL_PROMPT_FRAGMENT,
        "schema_json":     _detection_schema("fall"),
    },
    {
        "identifier":      "climbing",
        "name":            "Climbing",
        "icon_name":       "figure.climbing",
        "description":     "Climbing onto furniture or elevated surfaces.",
        "category":        "safety",
        "shape":           "state",
        "where_scope":     "ALL ROOMS",
        "when_scope":      "ALWAYS",
        "prompt_fragment": _CLIMBING_PROMPT_FRAGMENT,
        "schema_json":     _detection_schema("climbing"),
    },
    {
        "identifier":      "distress",
        "name":            "Distress",
        "icon_name":       "exclamationmark.bubble",
        "description":     "Sustained crying or visible distress.",
        "category":        "safety",
        "shape":           "state",
        "where_scope":     "ALL ROOMS",
        "when_scope":      "ALWAYS",
        "prompt_fragment": _DISTRESS_PROMPT_FRAGMENT,
        "schema_json":     _detection_schema("distress"),
    },
)


def _is_minor(date_of_birth: date) -> bool:
    """True if the patient is under 18 as of today (UTC). Subtracts a year
    when this year's birthday hasn't occurred yet."""
    today = datetime.now(timezone.utc).date()
    age = today.year - date_of_birth.year - (
        (today.month, today.day) < (date_of_birth.month, date_of_birth.day)
    )
    return age < 18


async def seed_detectors_for_patient(
    session: AsyncSession,
    patient_id: UUID,
    date_of_birth: date | None = None,
) -> Sequence[Detector]:
    """Create the preset detectors for a freshly-created patient.

    The set is chosen by age at creation: under 18 → the child set
    (Falls / Climbing / Distress); 18+ or unknown birthday → the
    older-adult set (Falls / Agitation / Wandering / Medication). An
    absent `date_of_birth` falls back to the adult set, preserving the
    prior default behavior.

    Caller owns the transaction (no commit here). Detectors land in
    insertion order, which becomes the iOS list order. Each row carries
    the runtime contract (identifier / prompt_fragment / schema_json)
    the detector pipeline reads when building Gemini/OpenAI calls.
    """
    presets = (
        _PRESET_DETECTORS_CHILD
        if date_of_birth is not None and _is_minor(date_of_birth)
        else _PRESET_DETECTORS_ADULT
    )
    # Stamp each preset with a microsecond-apart `created_at` so the
    # `ORDER BY created_at ASC` in `detector_repo.list_for_patient`
    # returns them in insertion order deterministically. Without this,
    # rows share `created_at` and Postgres's heap-scan order can drift
    # after autovacuum (caused intermittent test flakes).
    base = datetime.now(timezone.utc)
    inserted: list[Detector] = []
    for i, preset in enumerate(presets):
        now = base + timedelta(microseconds=i)
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
            identifier=preset["identifier"],
            prompt_fragment=preset["prompt_fragment"],
            schema_json=preset["schema_json"],
            always_on=False,
            is_builtin=True,
        )
        inserted.append(await detector_repo.insert(session, det))
    return inserted
