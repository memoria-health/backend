"""`thinking_level` env-string -> ThinkingConfig mapping.

Measured on gemini-3.5-flash-lite (2026-08-01): the model does not think
by default, `minimal`/`low` are no-ops on it, and only `medium`/`high`
produce thought tokens. We send `high`. These tests pin the MAPPING, not
the model behaviour — the point is that a bad env value can never take
detection down, and that we never send the 2.5-era `thinking_budget`
(which lite rejects outright with 400 INVALID_ARGUMENT).
"""

from __future__ import annotations

import pytest
from google.genai import types as genai_types

from src.services.detection.providers.gemini import _thinking_config


@pytest.mark.parametrize("value", ["minimal", "low", "medium", "high"])
def test_every_level_maps_to_the_matching_enum(value: str) -> None:
    cfg = _thinking_config(value)
    assert cfg is not None
    assert cfg.thinking_level == genai_types.ThinkingLevel(value.upper())


@pytest.mark.parametrize("value", ["HIGH", "  High  ", "MeDiUm"])
def test_case_and_whitespace_are_tolerated(value: str) -> None:
    """The value comes from .env, where humans type it."""
    assert _thinking_config(value) is not None


@pytest.mark.parametrize("value", ["", "   ", "none", "off", "true", "0", "extreme"])
def test_unknown_values_send_nothing_rather_than_raising(value: str) -> None:
    """A typo'd env value must fall back to the model default. Raising
    here would take every VLM call down on a one-character mistake."""
    assert _thinking_config(value) is None


def test_never_sends_thinking_budget() -> None:
    """`thinking_budget` is the Gemini-2.5 API; 3.5-flash-lite rejects it
    with 400 INVALID_ARGUMENT even when set to 0. Only `thinking_level`
    may ever be populated."""
    cfg = _thinking_config("high")
    assert cfg is not None
    assert cfg.thinking_budget is None


def test_default_setting_is_a_live_rung() -> None:
    """Guards the .env/Settings default against drifting to one of the
    rungs that measured as a no-op on lite."""
    from src.settings import Settings

    assert Settings().gemini_thinking_level.strip().lower() in {"medium", "high"}
