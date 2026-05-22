"""Prompt + JSON-schema builder unit tests.

Port of `nemara-temp-dashboard/tests/unit/detectors.test.mjs` adapted
to the Python `DetectorSpec` / `build_prompt_and_schema` API.

The builder always merges in the always-on `people_present` block,
dedupes by identifier, raises on schema-field collisions, and keeps
`description` as the first required field.
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


def test_people_present_is_always_merged_in_even_when_not_passed() -> None:
    """The caller never has to pass people_present; it's seeded by the
    builder so the JSON schema always carries the people_count and
    people_present columns the state machine + analyses store need.
    """
    result = build_prompt_and_schema([_fall_spec()])

    assert result.prompt.startswith(BASE_PROMPT)
    assert PEOPLE_PRESENT_SPEC.prompt_fragment in result.prompt
    assert "fall_detected" in result.prompt
    assert "fall_confidence" in result.prompt
    assert "fall_reasoning" in result.prompt

    # schema contains description + people_* + fall_*
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
    # Identical-identifier second copy — would raise on duplicate
    # schema field if dedupe didn't fire first.
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


def test_empty_detector_list_still_produces_valid_prompt() -> None:
    """No per-patient detectors → still a valid prompt + schema with
    just BASE_PROMPT + the always-on PEOPLE_PRESENT block.
    """
    result = build_prompt_and_schema([])

    # Prompt body — both BASE_PROMPT and the people_present fragment
    # are present, with the people_present block appearing AFTER the
    # base prompt.
    assert result.prompt.startswith(BASE_PROMPT)
    assert PEOPLE_PRESENT_SPEC.prompt_fragment in result.prompt
    pp_pos = result.prompt.index(PEOPLE_PRESENT_SPEC.prompt_fragment)
    base_pos = result.prompt.index(BASE_PROMPT)
    assert pp_pos > base_pos

    # Schema contains exactly description + people_present + people_count.
    assert set(result.schema_properties.keys()) == {
        "description", "people_present", "people_count",
    }
    assert set(result.schema_required) == set(result.schema_properties.keys())
    assert result.schema_required[0] == "description"


def test_caller_passing_people_present_explicitly_is_deduped() -> None:
    """The caller might mistakenly forward people_present in the list.
    The builder must NOT double-add the schema fields (which would
    otherwise raise on duplicate).
    """
    result = build_prompt_and_schema([PEOPLE_PRESENT_SPEC, _fall_spec()])

    assert set(result.schema_properties.keys()) == {
        "description",
        "people_present", "people_count",
        "fall_detected", "fall_confidence", "fall_reasoning",
    }
    # Fragment appears exactly once.
    assert result.prompt.count(PEOPLE_PRESENT_SPEC.prompt_fragment) == 1
