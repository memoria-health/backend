"""Prompt + JSON-schema builder unit tests.

Port of `nemara-temp-dashboard/tests/unit/detectors.test.mjs` adapted
to the Python `DetectorSpec` / `build_prompt_and_schema` API.

The builder appends caller-provided detectors after `BASE_PROMPT`,
dedupes by identifier, raises on schema-field collisions, and keeps
`description` as the first required field.

Note: the always-on `PEOPLE_PRESENT_SPEC` IS merged into every
prompt/schema — the LLM is the generic presence detector and
`camera_ingest` reads `result.people_present` (with re-ID's face match
as the POI override). A caller that also forwards the spec is deduped
by identifier, so it still appears exactly once.
"""

from __future__ import annotations

import pytest

from src.services.detection.prompt import (
    BASE_PROMPT,
    PEOPLE_PRESENT_SPEC,
    DetectorSpec,
    build_prompt_and_schema,
)


def _fall_spec() -> DetectorSpec:
    return DetectorSpec(
        identifier="fall",
        name="Falls",
        prompt_fragment=(
            "fall_detected: boolean.\nfall_confidence: number.\nfall_reasoning: string."
        ),
        schema_fields={
            "fall_detected":   {"type": "boolean"},
            "fall_confidence": {"type": "number"},
            "fall_reasoning":  {"type": "string"},
        },
    )


def _wandering_spec() -> DetectorSpec:
    return DetectorSpec(
        identifier="wandering",
        name="Wandering",
        prompt_fragment=(
            "wandering_detected: boolean.\n"
            "wandering_confidence: number.\n"
            "wandering_reasoning: string."
        ),
        schema_fields={
            "wandering_detected":   {"type": "boolean"},
            "wandering_confidence": {"type": "number"},
            "wandering_reasoning":  {"type": "string"},
        },
    )


def test_people_present_is_always_merged_into_prompt_and_schema() -> None:
    """The always-on PEOPLE_PRESENT_SPEC is merged into every build: the
    LLM is the generic presence detector (camera_ingest reads
    `result.people_present`, with re-ID's face match as the POI override).
    """
    result = build_prompt_and_schema([_fall_spec()])

    assert result.prompt.startswith(BASE_PROMPT)
    # The people_present prompt fragment is always present.
    assert PEOPLE_PRESENT_SPEC.prompt_fragment in result.prompt
    # The fall detector's fragments are present too.
    assert "fall_detected" in result.prompt
    assert "fall_confidence" in result.prompt
    assert "fall_reasoning" in result.prompt

    # Schema contains description + people_* + fall_*.
    assert set(result.schema_properties.keys()) == {
        "description",
        "people_present", "people_count",
        "fall_detected", "fall_confidence", "fall_reasoning",
    }
    # Every property must be in required, and vice-versa.
    assert set(result.schema_required) == set(result.schema_properties.keys())

    # description must be first in required.
    assert result.schema_required[0] == "description"

    # Type-match the inner field shapes.
    assert result.schema_properties["description"] == {"type": "string"}
    assert result.schema_properties["people_present"] == {"type": "boolean"}
    assert result.schema_properties["people_count"] == {"type": "number"}
    assert result.schema_properties["fall_detected"] == {"type": "boolean"}
    assert result.schema_properties["fall_confidence"] == {"type": "number"}
    assert result.schema_properties["fall_reasoning"] == {"type": "string"}


def test_duplicate_detector_identifier_only_included_once() -> None:
    """If the caller (eg. a buggy registry) passes the same detector
    twice, dedupe-by-identifier kicks in and we don't double up the
    schema (which would otherwise trigger the duplicate-field guard).
    """
    fall = _fall_spec()
    result = build_prompt_and_schema([fall, fall])

    assert set(result.schema_properties.keys()) == {
        "description",
        "people_present", "people_count",
        "fall_detected", "fall_confidence", "fall_reasoning",
    }
    assert set(result.schema_required) == set(result.schema_properties.keys())

    # The fall prompt fragment must appear EXACTLY once in the prompt.
    assert result.prompt.count(fall.prompt_fragment) == 1


def test_duplicate_schema_field_across_detectors_raises_value_error() -> None:
    """Two different detectors that emit the same schema key is a
    config bug — the builder must refuse to silently overwrite.
    """
    fall = _fall_spec()
    colliding = DetectorSpec(
        identifier="fall_clone",
        name="Fall Clone",
        prompt_fragment="fall_detected: boolean.",
        schema_fields={
            "fall_detected": {"type": "boolean"},   # collides with `fall`
        },
    )

    with pytest.raises(ValueError) as exc:
        build_prompt_and_schema([fall, colliding])
    # Error mentions both the field and the offender's identifier.
    assert "fall_detected" in str(exc.value)
    assert "fall_clone" in str(exc.value)


def test_description_is_always_the_first_required_field() -> None:
    """description must lead the required[] list — providers (Gemini's
    responseSchema, OpenAI's json_schema) preserve order, so this
    influences how the LLM is steered.
    """
    # Empty input
    r0 = build_prompt_and_schema([])
    assert r0.schema_required[0] == "description"

    # Many inputs
    r1 = build_prompt_and_schema([_fall_spec(), _wandering_spec()])
    assert r1.schema_required[0] == "description"


def test_empty_detector_list_still_includes_people_present() -> None:
    """No per-patient detectors → BASE_PROMPT + description + the
    always-on people_present block (nothing else).
    """
    result = build_prompt_and_schema([])

    assert result.prompt.startswith(BASE_PROMPT)
    assert PEOPLE_PRESENT_SPEC.prompt_fragment in result.prompt

    assert set(result.schema_properties.keys()) == {
        "description", "people_present", "people_count",
    }
    assert set(result.schema_required) == set(result.schema_properties.keys())
    assert result.schema_required[0] == "description"


def test_primary_detection_added_when_two_or_more_firing_detectors() -> None:
    """With ≥2 firing detectors a single action can satisfy several, so the
    builder asks the model to name the dominant one via `primary_detection`.
    """
    result = build_prompt_and_schema([_fall_spec(), _wandering_spec()])

    assert result.schema_properties["primary_detection"] == {"type": "string"}
    assert "primary_detection" in result.schema_required
    # The instruction names the candidate identifiers the model may return.
    assert "primary_detection" in result.prompt
    assert "fall" in result.prompt
    assert "wandering" in result.prompt
    # description still leads the required list.
    assert result.schema_required[0] == "description"


def test_primary_detection_omitted_for_single_or_no_firing_detector() -> None:
    """0–1 firing detectors → no conflict possible → field omitted, so the
    common single-detector prompt/schema is left exactly as before.
    """
    one = build_prompt_and_schema([_fall_spec()])
    assert "primary_detection" not in one.schema_properties
    assert "primary_detection" not in one.schema_required

    none = build_prompt_and_schema([])
    assert "primary_detection" not in none.schema_properties
    assert "primary_detection" not in none.schema_required


def test_caller_passing_people_present_explicitly_is_deduped() -> None:
    """A caller that also forwards `PEOPLE_PRESENT_SPEC` must NOT cause
    schema regrowth or a duplicate-field error — it's dedupe-seeded, so
    it still appears exactly once.
    """
    result = build_prompt_and_schema([PEOPLE_PRESENT_SPEC, _fall_spec()])

    assert set(result.schema_properties.keys()) == {
        "description",
        "people_present", "people_count",
        "fall_detected", "fall_confidence", "fall_reasoning",
    }
    # Always-on, but it appears EXACTLY once even when forwarded.
    assert result.prompt.count(PEOPLE_PRESENT_SPEC.prompt_fragment) == 1
