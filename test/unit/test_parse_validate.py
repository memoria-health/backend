"""parse_and_validate unit tests.

Covers the shared LLM-JSON parser/validator that both providers use.
The schema arg is the exact shape `build_prompt_and_schema` produces;
we synthesize a minimal one here for each test.
"""

from __future__ import annotations

import json

import pytest

from src.services.detection.providers._parse import parse_and_validate

_BASE_PROPERTIES: dict[str, dict[str, str]] = {
    "description":    {"type": "string"},
    "people_present": {"type": "boolean"},
    "people_count":   {"type": "number"},
    "fall_detected":  {"type": "boolean"},
    "fall_confidence": {"type": "number"},
    "fall_reasoning": {"type": "string"},
}
_BASE_REQUIRED: list[str] = list(_BASE_PROPERTIES.keys())


def _valid_payload(**overrides) -> dict:
    """Build a fully-valid payload, then apply overrides."""
    payload = {
        "description": "A person is walking in the living room.",
        "people_present": True,
        "people_count": 1,
        "fall_detected": False,
        "fall_confidence": 0.1,
        "fall_reasoning": "no abnormal motion",
    }
    payload.update(overrides)
    return payload


def test_missing_required_field_raises_value_error() -> None:
    payload = _valid_payload()
    del payload["fall_detected"]
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "fall_detected" in str(exc.value)


def test_wrong_type_boolean_expected_string_given_raises() -> None:
    payload = _valid_payload(fall_detected="yes")  # string, not bool
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "fall_detected" in str(exc.value)


def test_confidence_above_one_clamps_to_one() -> None:
    payload = _valid_payload(fall_confidence=1.5)
    result = parse_and_validate(
        json.dumps(payload),
        provider="gemini",
        required=_BASE_REQUIRED,
        properties=_BASE_PROPERTIES,
    )
    assert result.description == "A person is walking in the living room."
    assert result.people_present is True
    assert result.people_count == 1
    assert result.detector_fields == {
        "fall_detected": False,
        "fall_confidence": 1.0,
        "fall_reasoning": "no abnormal motion",
    }


def test_confidence_below_zero_clamps_to_zero() -> None:
    payload = _valid_payload(fall_confidence=-0.3)
    result = parse_and_validate(
        json.dumps(payload),
        provider="gemini",
        required=_BASE_REQUIRED,
        properties=_BASE_PROPERTIES,
    )
    assert result.description == "A person is walking in the living room."
    assert result.people_present is True
    assert result.people_count == 1
    assert result.detector_fields == {
        "fall_detected": False,
        "fall_confidence": 0.0,
        "fall_reasoning": "no abnormal motion",
    }


def test_string_values_are_trimmed() -> None:
    payload = _valid_payload(
        description="  the room is quiet  ",
        fall_reasoning="  no motion  ",
    )
    result = parse_and_validate(
        json.dumps(payload),
        provider="gemini",
        required=_BASE_REQUIRED,
        properties=_BASE_PROPERTIES,
    )
    assert result.description == "the room is quiet"
    assert result.people_present is True
    assert result.people_count == 1
    assert result.detector_fields == {
        "fall_detected": False,
        "fall_confidence": 0.1,
        "fall_reasoning": "no motion",
    }


def test_bool_is_not_treated_as_number_for_people_count() -> None:
    """Python gotcha: bool is a subclass of int. If the LLM returns
    `true` where a number is expected, we must reject — silently
    coercing to 1 would mask a model error.
    """
    payload = _valid_payload(people_count=True)
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "people_count" in str(exc.value)


def test_bool_is_not_treated_as_number_for_other_number_fields() -> None:
    """Same gotcha guard applies to other declared `number` fields."""
    payload = _valid_payload(fall_confidence=True)
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "fall_confidence" in str(exc.value)


def test_non_json_raw_string_raises_value_error() -> None:
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            "this is not json at all",
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "gemini returned non-JSON" in str(exc.value)


def test_missing_description_raises() -> None:
    payload = _valid_payload()
    del payload["description"]
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "description" in str(exc.value)


def test_empty_description_string_raises() -> None:
    """Whitespace-only description is treated as missing — providers
    must produce real prose for the analyses row to be useful in
    Studio."""
    payload = _valid_payload(description="   ")
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "description" in str(exc.value)


def test_missing_people_present_raises() -> None:
    payload = _valid_payload()
    del payload["people_present"]
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "people_present" in str(exc.value)


def test_missing_people_count_raises() -> None:
    payload = _valid_payload()
    del payload["people_count"]
    with pytest.raises(ValueError) as exc:
        parse_and_validate(
            json.dumps(payload),
            provider="gemini",
            required=_BASE_REQUIRED,
            properties=_BASE_PROPERTIES,
        )
    assert "people_count" in str(exc.value)


def test_extra_fields_land_in_detector_fields_not_dropped() -> None:
    """Anything not in {description, people_present, people_count} ends
    up in `detector_fields` — fall_* etc., and any other extras the
    provider emits.
    """
    payload = _valid_payload(
        fall_detected=True,
        fall_confidence=0.85,
        fall_reasoning="person dropped to the floor",
    )
    # Extra fields beyond the declared schema also flow through.
    payload["speech_heard"] = True
    payload["speech_transcript"] = "help"
    result = parse_and_validate(
        json.dumps(payload),
        provider="gemini",
        required=_BASE_REQUIRED,
        properties=_BASE_PROPERTIES,
    )
    assert result.description == "A person is walking in the living room."
    assert result.people_present is True
    assert result.people_count == 1
    assert result.detector_fields == {
        "fall_detected": True,
        "fall_confidence": 0.85,
        "fall_reasoning": "person dropped to the floor",
        "speech_heard": True,
        "speech_transcript": "help",
    }
