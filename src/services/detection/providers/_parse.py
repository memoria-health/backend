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

    # `people_present`/`people_count` are no longer required: Re-ID's
    # body pipeline is the source of truth for presence and the prompt
    # no longer asks the LLM to answer them. Older providers / cached
    # responses might still include the fields, so we accept-and-pass-
    # through with type validation; missing or wrong-typed values
    # default to (False, 0) and are discarded upstream when Re-ID ran.
    pp_raw = parsed.get("people_present")
    pc_raw = parsed.get("people_count")
    people_present = bool(pp_raw) if isinstance(pp_raw, bool) else False
    if isinstance(pc_raw, bool) or not isinstance(pc_raw, (int, float)):
        people_count = 0
    else:
        people_count = int(pc_raw)

    description = str(parsed["description"]).strip()

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
