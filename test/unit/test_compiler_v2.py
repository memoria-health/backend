"""Unit goldens: compiler v2 (detection v2 P5) — hard rejects,
reserved-key guard, and dims classification. Pure rules, no LLM.
"""

from __future__ import annotations

import pytest

from src.services.detection.scheduling.dims import CROP_BUCKETS
from src.services.detection.scheduling.compiler_v2 import (
    DetectorRejected,
    classify_custom_dims,
    reject_reason,
    validate_schema_fields,
)


# ─── Hard rejects (measured zeros) ───────────────────────────────


@pytest.mark.parametrize("name,text", [
    ("Eye twitch", "alert me if her eye twitches"),
    ("Tremor watch", "notify me about his fine hand tremor"),
    ("Shivering", "tell me when she is shivering"),
    ("Ring check", "is grandma wearing her ring"),
    ("Heel tapping", "he keeps heel tapping when anxious"),
])
def test_measured_zeros_are_rejected_with_plain_language(name, text) -> None:
    reason = reject_reason(name, text)
    assert reason is not None
    assert "testing" in reason  # honesty: cites the measurement


@pytest.mark.parametrize("name,text", [
    ("Fall", "alert me if he falls"),
    ("Seizure-like shaking", "whole body shaking violently"),
    ("Wedding ring box", "she moved the ring box off the shelf"),  # not wearing-a-ring
    ("Watching TV", "watching tv for more than an hour"),
    # CONTRAST phrasing: a seizure ask that merely MENTIONS shivering
    # must never be rejected (P5 review F1 — false reject = permanent
    # detection loss at creation, the forbidden failure).
    ("Seizure", "alert me about violent shaking, not just shivering"),
    ("Convulsions", "whole-body shaking or shivering under a blanket"),
    # Bracelet is wrist-scale, not a measured zero (P5 review F2).
    ("Medical bracelet", "make sure mom is wearing her bracelet"),
])
def test_detectable_asks_are_not_rejected(name, text) -> None:
    assert reject_reason(name, text) is None


def test_pure_shivering_ask_still_rejects() -> None:
    """The contrast exemption must not swallow the real measured zero."""
    assert reject_reason("Shivering", "tell me when she is shivering") is not None


# ─── Reserved-key guard ──────────────────────────────────────────


def test_reserved_keys_and_prefixes_reject_loudly() -> None:
    for bad in ("motion_src", "sig_env_motion", "sched_calls",
                "identity_confidence", "people_present", "rapid_x"):
        with pytest.raises(DetectorRejected):
            validate_schema_fields({bad: {"type": "boolean"}})
    # Normal custom trio passes untouched.
    validate_schema_fields({
        "custom_a1b2_detected": {"type": "boolean"},
        "custom_a1b2_confidence": {"type": "number"},
        "custom_a1b2_reasoning": {"type": "string"},
    })


# ─── Dims classification goldens ─────────────────────────────────


def _dims(name, text, shape="point_in_time", category="safety"):
    return classify_custom_dims(
        name=name, user_text=text, is_sustained=(shape == "state"),
        category=category,
    )


def test_environment_subject_classifies_env_scene() -> None:
    d = _dims("Stove check", "tell me if the stove is left on")
    assert (d["requires_motion"], d["subject_scope"], d["bucket"]) == (
        False, "environment", "scene",
    )


def test_still_activity_classifies_person_still() -> None:
    d = _dims("TV time", "she is watching tv", shape="state", category="routine")
    assert (d["requires_motion"], d["subject_scope"], d["bucket"]) == (
        False, "patient", "person",
    )
    assert d["duty_cycle"] == "thinned"      # routine state


def test_default_is_todays_behavior() -> None:
    d = _dims("Dancing", "alert me when he starts dancing")
    assert (d["requires_motion"], d["subject_scope"], d["bucket"]) == (
        True, "patient", "scene",
    )
    assert d["duty_cycle"] == "every_eligible_tick"
    assert d["eventing"] is None
    assert d["needs_audio"] is False
    assert d["degradation_warning"] == "none"


def test_anyone_subject_and_audio_and_distance_markers() -> None:
    d = _dims("Visitor", "alert me when a visitor calls out for help")
    assert d["subject_scope"] == "anyone"
    assert d["needs_audio"] is True
    d2 = _dims("Fidgeting", "picking at her fingers and fidgeting",
               shape="state", category="routine")
    assert d2["degradation_warning"] == "distance_class"


def test_env_wins_over_anyone_with_genuinely_matching_text() -> None:
    # BOTH regexes match here ("stranger" + "stove") — the env-first
    # precedence is what's under test (P5 review F7a: the old golden's
    # text never matched the anyone-regex at all).
    d = _dims("Stove", "a stranger left the stove on")
    assert d["subject_scope"] == "environment"


def test_still_wins_over_env_with_genuinely_matching_text() -> None:
    d = _dims("TV sleeper", "falls asleep with the tv left on, watching tv",
              shape="state", category="routine")
    assert d["requires_motion"] is False


def test_exit_verbs_keep_wandering_asks_person_scoped() -> None:
    """'leaves through the front door' is a wandering ask, not a
    door-state check — env classification would gate it on motion
    OUTSIDE person boxes post-soak, skipping exactly the tick that
    matters (P5 review F4)."""
    d = _dims("Front door", "alert me if dad leaves through the front door")
    assert (d["requires_motion"], d["subject_scope"], d["bucket"]) == (
        True, "patient", "scene",
    )
    d2 = _dims("Visitor", "a visitor knocks on the door")
    assert d2["subject_scope"] == "anyone"
    assert d2["requires_motion"] is True
    # Pure door-STATE phrasing stays environment.
    d3 = _dims("Door ajar", "the front door is left open at night")
    assert d3["subject_scope"] == "environment"


# ── body-scale actions must crop (2026-08-07) ────────────────────────


@pytest.mark.parametrize("name,text", [
    ("Drinking water", "Ary takes a drink from a cup or bottle"),
    ("Evening meds",   "Check if he takes his pills in the evening"),
    ("Eating",         "Detect when he eats a meal at the table"),
    ("Brushing teeth", "He brushes his teeth in the morning"),
    ("Inhaler",        "He uses his inhaler"),
])
def test_body_scale_actions_compile_to_the_person_bucket(name, text) -> None:
    """Their evidence is a hand, an object, and the gap to a face —
    all at body scale, none at room scale.

    `_STILL_ACTIVITY` used to be the ONLY route to `person`, so every
    ACTIVE body-scale ask fell through to person_motion + scene: a full
    1080p room where the object is a few dozen pixels after
    downsampling. The model then has nothing to read and writes what is
    LIKELY. Measured on the live pilot camera — a custom `drinking
    water` detector produced 11 false fires on 2026-08-06 and 8 in a
    single hour on 08-07, and the fabrication reached all the way into
    the `seen` field ("he holds a drinking cup in his right hand") while
    the subject sat crouched over a laptop with both hands down.

    The preset `medication` had already hit this exact failure from the
    other direction (scheduler.py:309 — "reported a pill being
    swallowed while the man held a camera"), which is why the rule is
    'the bucket decides' and why these must land in a crop bucket.
    """
    dims = classify_custom_dims(
        name=name, user_text=text, is_sustained=False, category="routine",
    )
    assert dims["bucket"] in CROP_BUCKETS, dims
    assert dims["subject_scope"] == "patient"


def test_environment_asks_are_not_dragged_into_the_person_bucket() -> None:
    """The body-scale rule sits AFTER the environment check, so a stove
    or a door still compiles scene/environment — otherwise every
    appliance ask would start demanding a body to crop to and would
    simply stop running."""
    dims = classify_custom_dims(
        name="Stove left on", user_text="Alert if the stove burner is left on",
        is_sustained=False, category="safety",
    )
    assert dims["bucket"] == "scene"
    assert dims["subject_scope"] == "environment"


def test_env_customs_emit_builtin_parity_cooldowns() -> None:
    """Env-classified customs carry the same backstop spacing as the env
    built-ins (Aryaman 2026-08-17): hazard prompts are event-phrased, so
    the cooldown only caps a VLM that mistakes lingering aftermath for a
    fresh event — 1800s hazards, 600s for the fire shape where frequent
    re-alerts while it burns are correct. Non-env customs must NOT carry
    the key at all: the row default (NULL -> 30s global) stays, and a
    recompile never nulls a caregiver-tuned value."""
    spill = classify_custom_dims(
        name="Water spilled", user_text="alert me when they spill water",
        is_sustained=False, category="safety",
    )
    assert spill["subject_scope"] == "environment"
    assert spill["cooldown_seconds"] == 1800

    fire = classify_custom_dims(
        name="Fire", user_text="alert me if anything catches fire",
        is_sustained=False, category="safety",
    )
    assert fire["subject_scope"] == "environment"
    assert fire["cooldown_seconds"] == 600

    fall = classify_custom_dims(
        name="Falls", user_text="alert me if my dad falls",
        is_sustained=False, category="safety",
    )
    assert "cooldown_seconds" not in fall
