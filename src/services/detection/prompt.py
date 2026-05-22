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

    `BASE_PROMPT` + `PEOPLE_PRESENT_SPEC` are always included; the
    caller-provided detectors are appended in order. Raises on
    duplicate schema field names across detectors (signals a config
    bug — two detectors output the same key).

    Verbatim port of `lib/detectors.ts:buildPromptAndSchema`.
    """
    parts: list[str] = [BASE_PROMPT]
    schema_properties: dict[str, dict[str, str]] = {
        "description": {"type": "string"},
    }
    schema_required: list[str] = ["description"]

    # `people_present` is alwaysOn — merged unconditionally so callers
    # don't have to pass it through. Seed `seen_identifiers` with it
    # BEFORE the loop so a duplicate `people_present` in `detectors`
    # is caught by the single dedupe check.
    seen_identifiers: set[str] = {PEOPLE_PRESENT_SPEC.identifier}
    all_specs: list[DetectorSpec] = [PEOPLE_PRESENT_SPEC]
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

    return BuiltPrompt(
        prompt="\n\n".join(parts),
        schema_properties=schema_properties,
        schema_required=schema_required,
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
