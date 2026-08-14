"""Confirm-by-evidence-risk (Aryaman, 2026-08-14).

The compiler judges whether a custom detector's evidence is error-prone
(subtle motion, small-object interaction, confounding context) and only
then does the instant-custom re-vote gate apply. The parse must strip
the metadata line from the runtime fragment, and any miss must land on
today's default (gate kept) — the 2026-08-08 confabulation storm is why
the gate can never be dropped by accident.
"""

from __future__ import annotations

from src.services.detection.scheduling.compiler_v2 import apply_confirm_need
from src.services.detector_compiler import _split_confirm_need

IDENT = "custom_ab12cd34"
K2N3 = {"confirm": {"k": 2, "n": 3}}


def _frag(extra: str) -> str:
    return (
        f"{IDENT}_detected: boolean. TRUE when clearly on film.\n"
        f"{IDENT}_confidence: number 0-1.\n"
        f"{IDENT}_reasoning: 1 short sentence.\n"
        f"{extra}"
    ).strip()


# ── _split_confirm_need ──────────────────────────────────────────────


def test_split_high_is_parsed_and_stripped():
    frag, need = _split_confirm_need(
        _frag(f"{IDENT}_confirm_need: high. Small cup, subtle wrist motion."),
        IDENT,
    )
    assert need == "high"
    assert "confirm_need" not in frag
    assert f"{IDENT}_reasoning" in frag          # runtime fields survive


def test_split_low_case_and_quoting_tolerant():
    frag, need = _split_confirm_need(
        _frag(f"  {IDENT}_confirm_need: 'LOW' — whole-body, unmistakable"),
        IDENT,
    )
    assert need == "low"
    assert "confirm_need" not in frag


def test_split_missing_line_returns_none_and_fragment_unchanged():
    src = _frag("")
    frag, need = _split_confirm_need(src, IDENT)
    assert need is None
    assert frag == src


def test_split_garbled_value_returns_none():
    src = _frag(f"{IDENT}_confirm_need: maybe? hard to say")
    frag, need = _split_confirm_need(src, IDENT)
    assert need is None
    assert frag == src                            # nothing stripped


def test_split_mid_fragment_line_is_removed_cleanly():
    src = (
        f"{IDENT}_detected: boolean. TRUE when clearly on film.\n"
        f"{IDENT}_confirm_need: high. Busy scene.\n"
        f"{IDENT}_state_active: boolean. TRUE while preconditions hold."
    )
    frag, need = _split_confirm_need(src, IDENT)
    assert need == "high"
    assert f"{IDENT}_detected" in frag and f"{IDENT}_state_active" in frag
    assert "confirm_need" not in frag


# ── apply_confirm_need ───────────────────────────────────────────────


def test_low_lifts_the_instant_default():
    assert apply_confirm_need(dict(K2N3), "low") is None


def test_high_keeps_the_gate():
    assert apply_confirm_need(dict(K2N3), "high") == K2N3


def test_none_keeps_todays_behavior():
    assert apply_confirm_need(dict(K2N3), None) == K2N3


def test_safety_instants_stay_fire_on_sight():
    assert apply_confirm_need(None, "high") is None   # never ADDS a gate


def test_sustained_entry_windows_pass_through():
    entry = {"entry": {"m": 3, "t_seconds": 300}}
    assert apply_confirm_need(dict(entry), "low") == entry


def test_hand_tuned_eventing_untouched():
    custom = {"confirm": {"k": 1, "n": 2}}
    assert apply_confirm_need(dict(custom), "low") == custom
