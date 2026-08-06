"""Prompt + JSON-schema assembly for the detector pipeline.

Ports `nemara-temp-dashboard/lib/detectors.ts:buildPromptAndSchema`
verbatim. The base prompt + always-on `people_present` block are
hardcoded here (rather than seeded as a per-patient detector row) so
the caregiver-facing detector list stays focused on user-tunable
detectors.

Both providers (Gemini and OpenAI) consume the same prompt + schema
shape — see `providers/gemini.py` and `providers/openai.py` for the
respective wire formats they wrap it in.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DetectorSpec:
    """The bits of a `detectors` row the prompt+schema builder needs."""

    identifier: str
    name: str
    prompt_fragment: str
    schema_fields: dict[str, dict[str, str]]
    always_on: bool = False


# ── Hardcoded prompts (matches dashboard `lib/detectors.ts`) ───────

BASE_PROMPT = (
    "You are monitoring a home camera to look after the person in view. "
    "Respond with a JSON object containing every field below, FILLING THEM "
    "IN THE ORDER GIVEN.\n\n"
    # ── Evidence before verdict (2026-08-06) ──────────────────────────
    # The prompt used to open with `description`, defined as justifying the
    # detections ("justify whatever you detect below"). That is verdict-first
    # reasoning: the model settles on a label and then writes prose to support
    # it. It is the mechanism behind the 08-05/06 false positives — the
    # narrative was accurate right up to the verb ("crouches down and FALLS
    # onto the floor", "bends to clean the table then COLLAPSES"), because the
    # sentence existed to justify a conclusion already reached.
    #
    # `observed` runs first and is deliberately verdict-free. Fields are
    # generated in order, so committing to neutral observations first means
    # the detections below are conditioned on evidence rather than the reverse.
    # Measured on the 08-05 clips: forcing transcribe-then-attribute-then-
    # describe before the question took flash-lite from firing on ordinary
    # conversation to 8/8 correct, and recovered a real call for help that the
    # verdict-first shape scored at 0.2 confidence.
    # These are three SEPARATE fields, not one `observed` blob. The blob was
    # tried first (2026-08-06) and silently failed: asked for an AUDIO line
    # and a VIDEO line in one string, the model wrote only the video half —
    # "A person is seated in a chair facing a desk with a laptop." — and then
    # fired calling_for_help claiming 'At 00:08, the person clearly says
    # "help me"'. A single string can be satisfied by any sentence, so it
    # gets satisfied by the easy one. A required field cannot be skipped.
    "heard: transcribe any speech VERBATIM. If words cannot be made out, "
    "write exactly UNINTELLIGIBLE. If there is no sound, write SILENCE. "
    "Never substitute a plausible-sounding phrase for one you did not hear, "
    "and never translate into another language. This is the ONLY place "
    "speech may be quoted — if a phrase does not appear here, no detector "
    "below may claim it was said.\n"
    "audio_source: exactly one of 'person_on_camera' (you can see them "
    "speaking), 'device' (a TV, phone, laptop or speaker — including video "
    "or music being played), 'other_room' (off-camera), 'none' (no speech), "
    "'cannot_tell'. A person watching a video is 'device'.\n"
    "seen: body position and how it changes, in plain physical terms "
    "(standing, crouching, kneeling, lying, sitting, walking toward). Do NOT "
    "write fell, collapsed, fainted, took, swallowed or drank — those are "
    "conclusions and belong to the detectors below, not here. Write UNCLEAR "
    "for anything you cannot resolve at this distance or resolution; "
    "guessing here corrupts every field after it.\n\n"
    "description: write this for the caregiver who could not watch live — "
    "from it alone they should understand exactly what "
    "happened, start to finish, and why it was flagged. So justify whatever "
    "you detect below: describe what the person did and said and the "
    "specific things about them that led to your conclusion, making clear "
    "what about the person made each detector you mark true. Justify ONLY "
    "the detectors you mark true. Never write what did not happen and never "
    "mention a detector you mark false: no sentences like 'no signs of "
    "distress were observed' or 'there was no fall' — leaving something "
    "unmentioned already says it did not happen. Every sentence must report "
    "something actually seen or heard in the clip. Mention an "
    "object only when it matters to the detection, and include a specific "
    "detail about it (such as which medication was taken) only when you can "
    "credibly see it — never guess or invent details. Skip surroundings and "
    "incidental objects that don't change how the event is read. Use as many "
    "sentences as the moment needs. Always present."
)

# always_on=true detector. NOT seeded into per-patient detectors rows
# — included implicitly by `build_prompt_and_schema`. Same prompt as
# the dashboard's `PEOPLE_PRESENT_PROMPT`.
PEOPLE_PRESENT_SPEC = DetectorSpec(
    identifier="people_present",
    name="People Present",
    always_on=True,
    prompt_fragment=(
        "people_present: boolean. True if any human is at least "
        "partially visible in any frame (including just legs, just a "
        "torso, just an arm). False ONLY if you are confident no human "
        "is in any frame at all.\n"
        "people_count: integer (number). Best-estimate count of "
        "distinct people visible. 0 if people_present is false."
    ),
    schema_fields={
        "people_present": {"type": "boolean"},
        "people_count":   {"type": "number"},
    },
)


# ── Builder ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BuiltPrompt:
    """Output of `build_prompt_and_schema`. Providers wrap this in their
    own request shape (Gemini: responseSchema; OpenAI: json_schema).
    """

    prompt: str
    schema_properties: dict[str, dict[str, str]]
    schema_required: list[str]


# Prepended to the prompt (right after BASE_PROMPT) when the frame-diff motion
# overlay is active, so the model knows what the red tint is. Without this,
# detectors misread the red as blood / a real object.
_MOTION_OVERLAY_NOTE = (
    "MOTION HIGHLIGHT: a semi-transparent RED tint has been overlaid onto the "
    "areas of each frame that MOVED since the previous frame. It marks MOTION "
    "only — it is NOT blood, a light, or any real object, so never report the red "
    "itself as an object or injury. Read it as a motion cue: more red = more "
    "movement there. A body part that stays RED across many frames while staying "
    "in roughly the SAME place (not travelling across the frame) means fast, "
    "repetitive, LOCALIZED motion — a hand shaking or trembling, or a limb "
    "convulsing. A red region that moves ACROSS the frame is ordinary movement "
    "(walking, reaching, gesturing). Judge the actual behaviour from the "
    "underlying scene; the tint only tells you where and how much motion is.\n\n"
)


def build_prompt_and_schema(
    detectors: Sequence[DetectorSpec], *, motion_overlay: bool = False,
) -> BuiltPrompt:
    """Assemble the final prompt + JSON-schema for the LLM call.

    `BASE_PROMPT` is always included; the caller-provided detectors
    are appended in order. Raises on duplicate schema field names
    across detectors (signals a config bug — two detectors output
    the same key).

    `PEOPLE_PRESENT_SPEC` is merged in as always-on: the LLM is the
    generic presence detector. `camera_ingest` reads `result.people_present`
    as the looser-than-YOLO presence signal, with the re-ID face match
    (the patient / POI) as the override. (It was briefly removed in favor
    of YOLO-derived presence, but YOLO body count misses people the LLM
    catches, so the LLM verdict is authoritative for generic presence and
    re-ID's face match is authoritative for "is it the patient.")

    Verbatim port of `lib/detectors.ts:buildPromptAndSchema` minus
    the always-on merge.
    """
    parts: list[str] = [BASE_PROMPT]
    if motion_overlay:
        parts.append(_MOTION_OVERLAY_NOTE)
    schema_properties: dict[str, dict[str, str]] = {
        # FIRST, deliberately: structured output is generated in property
        # order, so this is what makes "evidence before verdict" real rather
        # than advisory. See the note in BASE_PROMPT.
        "heard": {
            "type": "string",
            "description": (
                "Verbatim speech only, or UNINTELLIGIBLE, or SILENCE. Never "
                "a plausible substitute, never translated. The only place "
                "speech may be quoted."
            ),
        },
        "audio_source": {
            "type": "string",
            "enum": [
                "person_on_camera", "device", "other_room", "none",
                "cannot_tell",
            ],
        },
        "seen": {
            "type": "string",
            "description": (
                "Plain physical body position only. No conclusions: never "
                "'fell', 'collapsed', 'took a pill', 'drank'. UNCLEAR for "
                "anything not resolvable."
            ),
        },
        "description": {
            "type": "string",
            # Schema-level reinforcement of the BASE_PROMPT rule —
            # flash-lite kept writing "no signs of X" filler when the
            # rule lived only mid-prompt.
            "description": (
                "Only what actually happened and was seen or heard. "
                "Never mention absences or detectors that are false — "
                "no sentences like 'no signs of distress' or 'there "
                "was no fall'."
            ),
        },
    }
    schema_required: list[str] = ["heard", "audio_source", "seen", "description"]

    # Always-on people_present block, RE-ENABLED. The LLM is the generic
    # presence detector again (camera_ingest's gate reads result.people_present
    # — looser than YOLO body count, with the re-ID face match as the POI
    # override). Prepended so its prompt_fragment + schema_fields are always
    # emitted; its identifier stays in `seen_identifiers` so a caller that also
    # forwards it is deduped (no duplicate schema field).
    all_specs: list[DetectorSpec] = [PEOPLE_PRESENT_SPEC]
    seen_identifiers: set[str] = {PEOPLE_PRESENT_SPEC.identifier}
    for d in detectors:
        if d.identifier in seen_identifiers:
            continue
        seen_identifiers.add(d.identifier)
        all_specs.append(d)

    for d in all_specs:
        parts.append(d.prompt_fragment)
        for name, field in d.schema_fields.items():
            if name in schema_properties:
                raise ValueError(
                    f"[detectors] duplicate schema field '{name}' across "
                    f"detectors (introduced by '{d.identifier}')"
                )
            schema_properties[name] = field
            schema_required.append(name)

    # One-event-per-clip pick. When TWO OR MORE firing detectors are
    # configured, a single action can satisfy several of them (a fall is
    # also a general emergency; agitation overlaps calling-for-help), which
    # would otherwise file one event per matching detector. Ask the model
    # to name the SINGLE detection that dominates the clip; event_promotion
    # fires only that one (falling back to highest confidence when this is
    # empty/invalid). Omitted with 0–1 firing detectors — no conflict is
    # possible, so the common single-detector prompt is left untouched.
    # `people_present` is always-on (not a firing detector) and excluded.
    firing_ids = [s.identifier for s in all_specs if not s.always_on]
    if len(firing_ids) >= 2:
        parts.append(
            "primary_detection: string. Several detectors above can be true "
            "for a single clip when one action overlaps their definitions "
            "(for example a fall is also a general emergency, or agitation "
            "overlaps calling for help). Of the detectors you marked "
            "detected=true, return the ONE identifier whose triggering "
            "action or sound is the dominant event in this clip. Valid "
            "identifiers: " + ", ".join(firing_ids) + ". Return an empty "
            "string if you marked no detector detected=true."
        )
        schema_properties["primary_detection"] = {"type": "string"}
        schema_required.append("primary_detection")

    return BuiltPrompt(
        prompt="\n\n".join(parts),
        schema_properties=schema_properties,
        schema_required=schema_required,
    )


def build_identity_addendum(patient_in_clip: bool) -> str:
    """Return the prompt addendum to append when Re-ID has identified
    the patient in the current clip. Empty string otherwise.

    Two states:

      - patient_in_clip=True  -> tell the LLM to focus detector
                                 outcomes on the patient
      - patient_in_clip=False -> no addendum; today's behavior (score
                                 any visible person, including for
                                 safety detectors)

    Identification is carried purely as text guidance — no visual box
    is drawn on the frames on any provider.
    """
    if not patient_in_clip:
        return ""
    return (
        "\n\n"
        "Re-ID has identified the patient in this clip with high "
        "confidence. Focus detector outcomes on the patient. Other "
        "people may be present but should not be the subject of "
        "behavioral findings unless they directly interact with the "
        "patient. Safety findings (fall, agitation, distress) should "
        "still be reported for the patient even if they happen while "
        "the patient is partially out of frame."
    )


def detector_spec_from_row(row: Any) -> DetectorSpec:
    """Adapt an ORM `Detector` row to a `DetectorSpec`. Skips rows
    without an `identifier` (legacy / not-yet-migrated rows).
    """
    if not row.identifier or not row.prompt_fragment or not row.schema_json:
        raise ValueError(
            f"[detectors] detector {row.id} ({row.name}) is missing "
            f"runtime contract fields (identifier / prompt_fragment / "
            f"schema_json). Seed via `onboarding.seed_detectors_for_patient` "
            f"or backfill before running the pipeline."
        )
    return DetectorSpec(
        identifier=row.identifier,
        name=row.name,
        prompt_fragment=row.prompt_fragment,
        schema_fields=row.schema_json,
        always_on=row.always_on,
    )
