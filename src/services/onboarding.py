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

# NOTE (2026-06-04): the agitation + medication fragments below were just
# revised to be precision-first — agitation now fires only on clear,
# unambiguous physical OR verbal distress (two independent channels), and
# medication gates hard on unmistakably identifying a medicine container.
# TODO (2026-06-05): collect more live clips and re-test BOTH detectors
# against these revised prompts before tuning thresholds or wording further.
_AGITATION_PROMPT_FRAGMENT = (
    "agitation_detected: boolean. Prioritize PRECISION over recall: set "
    "TRUE only when there is CLEAR, OBVIOUS evidence of genuine "
    "distress that leaves no room for doubt. Casual or ordinary "
    "activity is NOT agitation. When unsure, set FALSE.\n"
    "\n"
    "There are two independent kinds of evidence. EITHER one alone is "
    "enough — but only when it is unambiguous:\n"
    "\n"
    "PHYSICAL — clear, obvious physical distress, such as:\n"
    "  - PACING in the clearest sense: repeatedly walking back and "
    "forth in an agitated, purposeless way — NOT simply crossing the "
    "room or walking over to do something;\n"
    "  - frantic running or rushing about in evident distress;\n"
    "  - frantic, disordered rummaging — agitatedly tearing through or "
    "tossing things, NOT calm searching or tidying;\n"
    "  - throwing objects, or banging / striking / slamming surfaces or "
    "objects forcefully and angrily.\n"
    "The action must clearly read as distress or anger. If it has an "
    "ordinary explanation — exercise, dancing, play, sport, swatting an "
    "insect, normal chores, reaching for or carrying something, moving around the house — it is "
    "NOT agitation.\n"
    "\n"
    "VERBAL — clear distress in BOTH tone and content, from the person "
    "on camera:\n"
    "  - TONE / affect: shouting, sharp, harsh, or hostile voice, or "
    "sustained crying / groaning / yelling; AND\n"
    "  - CONTENT: angry, hostile, or distressed words — cursing, "
    "threats, distressed demands, or calls for help.\n"
    "Both tone AND content must point to distress. Ordinary "
    "conversation does NOT count — a normal phone call, a minor "
    "argument or debate, or an animated explanation has a "
    "conversational tone and topical content, not hostility or "
    "distress.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the movement is ordinary or calm — walking to a destination, "
    "doing chores, searching calmly, exercising, dancing, playing, or "
    "swatting an insect;\n"
    "  - there is motion but you cannot clearly tell it is distress "
    "rather than ordinary activity;\n"
    "  - the only voice is ordinary conversation, argument, or "
    "explanation, or it likely comes from a TV / radio / phone / device "
    "or cannot be attributed to the person on camera;\n"
    "  - the person is sitting calmly, resting, or sleeping;\n"
    "  - no person is visible, or the person is too far / too dark or "
    "the audio too unclear to judge — when you cannot tell, set "
    "FALSE.\n"
    "\n"
    "agitation_confidence: number 0–1. 0 = calm / ordinary activity / "
    "no person / not observable. 0.3–0.6 = some raised voice or "
    "vigorous motion, but you cannot rule out ordinary activity (it "
    "will not alert here — that is intended). 0.7+ = clear, unambiguous "
    "physical distress OR clear distressed tone AND content; raise "
    "toward 0.9 when both occur together.\n"
    "agitation_reasoning: 1 short sentence stating the clear evidence "
    "you saw or heard, or why it was not agitation (e.g. 'threw objects "
    "and banged the table repeatedly in anger', 'paced frantically back "
    "and forth for ~8s with no purpose', 'shouted curses and threats "
    "while sitting still', 'forceful arm motion but it was swatting a "
    "bug — not distress', 'raised voice but an ordinary phone "
    "argument', 'no person visible')."
)

_MEDICATION_PROMPT_FRAGMENT = (
    "medication_taken: boolean. Set TRUE only with CLEAR, OBVIOUS proof "
    "that the person took medication, by one of these two patterns. Do "
    "not speculate about other things they might ingest or other "
    "containers they might handle.\n"
    "\n"
    "VISUAL GATE (oral). Set FALSE unless at least ONE of these is "
    "clearly present in view: (a) a typical pill container — a pill "
    "bottle, blister / foil pack, or weekly pill organizer; OR (b) an "
    "unmistakable pill, tablet, or capsule. A generic cup, mug, glass, "
    "food package, or other non-medicine container does NOT count. If "
    "neither a typical pill container nor a clearly visible pill is in "
    "view, set FALSE — never infer medication from a drink or a "
    "hand-to-mouth motion alone.\n"
    "\n"
    "ORAL — all three steps visible, with the drink either just before "
    "or just after the pill:\n"
    "  - they handle an obvious medicine container — a pill bottle, a "
    "blister / foil pack, or a weekly pill organizer;\n"
    "  - they take medicine out and put it in their mouth;\n"
    "  - they drink a liquid right before or right after (pill then "
    "drink, or drink then pill).\n"
    "\n"
    "INJECTION — they clearly give themselves an injection with an "
    "obvious device (insulin pen or syringe) against the body.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - neither a typical pill container nor a clearly visible pill is "
    "in view (a drink or a hand-to-mouth motion alone is not enough);\n"
    "  - you cannot clearly see the pill placed in the mouth (or, for "
    "injection, the device against the body);\n"
    "  - only part of the sequence is visible — e.g. a drink alone, or "
    "holding a container without taking anything out;\n"
    "  - the person eats food or drinks a beverage with no medicine "
    "container and no pill visible;\n"
    "  - they handle some other, non-medicine container (food, a mug, a "
    "drink bottle);\n"
    "  - no person is visible, or the action is too far or unclear to "
    "confirm.\n"
    "\n"
    "medication_confidence: number 0–1. 0 = no medication behavior / "
    "unclear. 0.3–0.6 = partial or ambiguous (e.g. handles a bottle but "
    "the pill-to-mouth is not clearly seen). 0.7+ = the full pattern "
    "above is clearly visible.\n"
    "medication_reasoning: 1 short sentence stating the proof you saw "
    "(e.g. 'opened a weekly pill organizer, placed a pill in mouth, "
    "drank water', 'injected with an insulin pen in the abdomen', "
    "'drank from a glass but no pills or medicine container visible', "
    "'no person visible')."
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
