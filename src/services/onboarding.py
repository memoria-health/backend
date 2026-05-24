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
from datetime import datetime, timedelta, timezone
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
    "The camera is wall-mounted with a fixed view — a person can fall PARTIALLY OR FULLY OUT OF FRAME during a real fall (e.g., they collapse below the camera's downward field of view).\n\n"
    "fall_detected: boolean. Be INCLUSIVE. Set TRUE for any of:\n"
    "  - a person fully collapsing, slipping, tripping, stumbling hard, sliding down to the floor;\n"
    "  - suddenly dropping to the ground from standing / a chair / a bed;\n"
    "  - any motion where a person ends up on the ground in a way that does not look fully controlled;\n"
    "  - **a person who was visible upright (or upright moving) then suddenly disappears downward out of the bottom of the frame** — treat this as a likely fall;\n"
    "  - a person becomes partially visible (just legs, just torso slumped) after having been fully upright a moment before;\n"
    "  - a person's body language changes from controlled posture to limp / horizontal in a fast, unbroken motion.\n"
    "Set FALSE only if the person clearly remains upright and stable the entire clip; OR walks out of frame sideways at normal walking pace; OR is never visible at all; OR deliberately and slowly sits / lies down in a controlled way.\n\n"
    "fall_confidence: number 0–1. How confident you are a fall (including partial / out-of-frame / near-fall) occurred. 0 = clearly nothing fall-like. 0.3–0.6 = ambiguous (e.g., person disappears but motion was sideways, could be either). 0.7+ = clear fall or clear sudden downward disappearance.\n\n"
    "fall_reasoning: 1 short sentence explaining your fall judgment — what specifically you saw or didn't see (e.g. 'person stayed seated throughout', 'person dropped from standing to floor around the middle of the clip', 'person was visible standing then suddenly disappeared downward out of frame — likely fell below camera view', 'person walked out of frame to the right at normal pace, no fall', 'no person visible at any point'). Always present."
)

_WANDERING_PROMPT_FRAGMENT = (
    "wandering_detected: boolean. True ONLY if, during this 15-second clip, "
    "a person was clearly present in the room at some point AND has actually "
    "LEFT THE ROOM by the end of the clip — they walked through a doorway, "
    "hallway, or other exit, and are no longer in the room.\n"
    "\n"
    "CRITICAL — distinguish 'left the room' from 'walked out of camera frame':\n"
    "  - 'Left the room' means physically went through a door / out of the "
    "room's physical space. The room itself no longer contains them.\n"
    "  - 'Out of camera frame' means they could still be in the room but "
    "behind the camera, behind furniture, or just off-screen. NOT wandering.\n"
    "  - If you cannot distinguish the two with confidence, set FALSE and "
    "reflect uncertainty in the confidence score.\n"
    "\n"
    "Set TRUE when:\n"
    "  - A person walked through a clearly visible doorway / opening to "
    "another room and did not return within the clip.\n"
    "  - A person was visible inside the room earlier in the clip and is "
    "no longer in the room (not just out of frame) by the end.\n"
    "\n"
    "Set FALSE when:\n"
    "  - The person stayed in the room throughout (still visible, or moved "
    "out of frame but clearly still in the room — sounds, partial visibility, "
    "or movement returning quickly).\n"
    "  - The person left and CAME BACK within the same clip — not wandering, "
    "they're still in the room overall.\n"
    "  - The person approached / touched / opened a door but did not actually "
    "leave through it. Don't pre-trigger on intent; the next clip will catch "
    "the actual exit (15-second clips overlap by 5 seconds).\n"
    "  - No person is visible at any point.\n"
    "  - The person walked out of frame in a direction that's not clearly a "
    "doorway or exit (could be just stepping behind something).\n"
    "\n"
    "wandering_confidence: number 0–1. 0 = clearly no one left the room. "
    "0.3–0.6 = ambiguous (e.g. person walked out of frame but unclear if "
    "they went through a door). 0.7+ = clearly walked out a doorway and "
    "did not return.\n"
    "wandering_reasoning: 1 short sentence describing what you observed — "
    "for example: 'person walked through the doorway on the left at 8s and "
    "did not return', 'person stayed seated throughout', 'person stepped "
    "off camera but partial visibility and sounds indicate still in room', "
    "'no person visible'."
)

_AGITATION_PROMPT_FRAGMENT = (
    "agitation_detected: boolean. True if a person displays sustained restless body language over the clip — pacing back and forth, fidgeting with clothing/objects in a repetitive nervous way, repeatedly standing/sitting, wringing hands, sharp jerky gestures, or other movement patterns suggesting distress/anxiety. The behavior must be SUSTAINED across multiple seconds of the clip — a single moment of motion is not agitation. Set FALSE if the person is moving normally (walking through, doing a task with purpose, sitting still, sleeping), or if no person is visible.\n"
    "agitation_confidence: number 0–1. 0 = clearly calm/normal/no person. 0.3–0.6 = some restless motion but ambiguous (could be normal activity). 0.7+ = clear sustained agitation.\n"
    "agitation_reasoning: 1 short sentence explaining your judgment (e.g. 'person paced from doorway to window four times across the clip', 'person sat reading throughout — calm', 'no person visible')."
)

_MEDICATION_PROMPT_FRAGMENT = (
    "medication_taken: boolean. True if the clip shows clear medication-taking behavior — picking up a pill bottle / blister pack / weekly pill organizer, placing a pill in the mouth, drinking water immediately after, OR sitting at the kitchen counter with a pill organizer open. Set FALSE if no medication-related interaction is visible, the person is just in the kitchen for other reasons (cooking, drinks), or no person is visible.\n"
    "medication_confidence: number 0–1. 0 = no medication-related behavior visible. 0.3–0.6 = ambiguous (e.g. person at kitchen counter but pills not clearly visible). 0.7+ = clear medication-taking sequence.\n"
    "medication_reasoning: 1 short sentence explaining your judgment (e.g. 'person opened weekly pill organizer and swallowed pills with water', 'person at kitchen counter making tea, no pills visible', 'no person visible')."
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


_PRESET_DETECTORS: tuple[dict[str, object], ...] = (
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


async def seed_detectors_for_patient(
    session: AsyncSession, patient_id: UUID
) -> Sequence[Detector]:
    """Create the four preset detectors for a freshly-created patient.

    Caller owns the transaction (no commit here). Detectors land in
    insertion order, which becomes the iOS list order. Each row carries
    the runtime contract (identifier / prompt_fragment / schema_json)
    the detector pipeline reads when building Gemini/OpenAI calls.
    """
    # Stamp each preset with a microsecond-apart `created_at` so the
    # `ORDER BY created_at ASC` in `detector_repo.list_for_patient`
    # returns them in insertion order deterministically. Without this,
    # all four rows share `created_at` and Postgres's heap-scan order
    # can drift after autovacuum (caused intermittent test flakes).
    base = datetime.now(timezone.utc)
    inserted: list[Detector] = []
    for i, preset in enumerate(_PRESET_DETECTORS):
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
