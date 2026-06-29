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
    "You are monitoring a home security camera. Respond with a JSON "
    "object containing every field below.\n\n"
    "description: 1–2 sentence summary of what's happening (visual). "
    "Always present."
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


def build_prompt_and_schema(detectors: Sequence[DetectorSpec]) -> BuiltPrompt:
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
    schema_properties: dict[str, dict[str, str]] = {
        "description": {"type": "string"},
    }
    schema_required: list[str] = ["description"]

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

    Two states (per the plan's "two prompt states" design):

      - patient_in_clip=True  -> tell the LLM about the green box AND
                                 to focus on the patient
      - patient_in_clip=False -> no addendum; today's behavior (score
                                 any visible person, including for
                                 safety detectors)

    The OpenAI path actually draws the green rectangle on every frame
    via `frame_annotate.annotate_jpeg`. Gemini accepts the raw MP4
    and the text guidance carries it — the box-on-frames pass would
    require a decode/re-encode loop we haven't committed to yet.
    """
    if not patient_in_clip:
        return ""
    return (
        "\n\n"
        "Re-ID has identified the patient in this clip with high "
        "confidence. The patient is marked with a thin green "
        "rectangle drawn around their face on every frame where the "
        "rectangle is visible. Focus detector outcomes on the person "
        "inside the green rectangle. Other people may be present but "
        "should not be the subject of behavioral findings unless they "
        "directly interact with the patient. Safety findings (fall, "
        "agitation, distress) should still be reported for the "
        "patient even if they happen while the patient is partially "
        "out of frame."
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
