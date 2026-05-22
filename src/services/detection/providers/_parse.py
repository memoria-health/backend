"""Shared LLM JSON response parser + validator.

Verbatim port of `nemara-temp-dashboard/lib/providers/gemini.ts:parseAndValidate`.
Both providers (Gemini, OpenAI) return JSON matching the same schema
the prompt asked for; this validates type-correctness, clamps
`*_confidence` fields to `[0, 1]`, and trims string fields.
"""

import json
import logging
from typing import Any

from src.services.detection.providers.types import ProviderResult

_log = logging.getLogger(__name__)


def parse_and_validate(
    raw: str,
    *,
    provider: str,
    required: list[str],
    properties: dict[str, dict[str, str]],
) -> ProviderResult:
    """Parse the LLM's JSON output, validate required fields against the
    schema, extract `description` / `people_present` / `people_count`,
    and put the rest in `detector_fields`. Clamps `*_confidence` floats
    to `[0, 1]` and trims string values.

    Raises ValueError on any schema mismatch (the providers map these
    to a more specific exception type at their boundary).
    """
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"{provider} returned non-JSON: {e} — raw: {raw[:500]}"
        ) from e

    # Type-check every required field.
    missing: list[str] = []
    for key in required:
        expected = properties.get(key, {}).get("type")
        v = parsed.get(key)
        if expected == "boolean":
            if not isinstance(v, bool):
                missing.append(f"{key}:boolean")
        elif expected == "number":
            # bool is a subclass of int in Python — exclude it.
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                missing.append(f"{key}:number")
        elif expected == "string":
            if not isinstance(v, str):
                missing.append(f"{key}:string")
            elif key == "description" and not v.strip():
                missing.append(f"{key}:non-empty-string")
        else:
            missing.append(f"{key}:unknown-type-{expected}")
    if missing:
        _log.error("[%s] response missing/wrong-typed: %s — raw: %s",
                   provider, ", ".join(missing), raw[:500])
        raise ValueError(
            f"{provider} response missing/wrong-typed: {', '.join(missing)} "
            f"— raw: {raw[:500]}"
        )

    if not isinstance(parsed.get("people_present"), bool):
        raise ValueError(
            f"{provider} response missing people_present (boolean) — raw: {raw[:500]}"
        )
    if isinstance(parsed.get("people_count"), bool) or not isinstance(
        parsed.get("people_count"), (int, float)
    ):
        raise ValueError(
            f"{provider} response missing people_count (number) — raw: {raw[:500]}"
        )

    description = str(parsed["description"]).strip()
    people_present = bool(parsed["people_present"])
    people_count = int(parsed["people_count"])

    detector_fields: dict[str, Any] = {}
    for key, val in parsed.items():
        if key in {"description", "people_present", "people_count"}:
            continue
        detector_fields[key] = val

    # Clamp `*_confidence` to [0, 1], trim strings.
    for key, val in list(detector_fields.items()):
        if key.endswith("_confidence") and isinstance(val, (int, float)) and not isinstance(val, bool):
            detector_fields[key] = max(0.0, min(1.0, float(val)))
        elif isinstance(val, str):
            detector_fields[key] = val.strip()

    return ProviderResult(
        description=description,
        people_present=people_present,
        people_count=people_count,
        detector_fields=detector_fields,
    )
