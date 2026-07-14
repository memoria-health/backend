"""Unit goldens: compiler v2 (detection v2 P5) — hard rejects,
reserved-key guard, and dims classification. Pure rules, no LLM.
"""

from __future__ import annotations

import pytest

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
])
def test_detectable_asks_are_not_rejected(name, text) -> None:
    assert reject_reason(name, text) is None


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
        name=name, user_text=text, shape=shape, category=category,
    )


def test_environment_subject_classifies_env_snapshot() -> None:
    d = _dims("Stove check", "tell me if the stove is left on")
    assert (d["evidence_branch"], d["subject_scope"], d["bucket"]) == (
        "environment", "environment", "snapshot",
    )


def test_still_activity_classifies_person_still() -> None:
    d = _dims("TV time", "she is watching tv", shape="state", category="routine")
    assert (d["evidence_branch"], d["subject_scope"], d["bucket"]) == (
        "person_still", "patient", "person",
    )
    assert d["duty_cycle"] == "sampled"      # routine state


def test_default_is_todays_behavior() -> None:
    d = _dims("Dancing", "alert me when he starts dancing")
    assert (d["evidence_branch"], d["subject_scope"], d["bucket"]) == (
        "person_motion", "patient", "scene",
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


def test_duration_claim_maps_to_entry_preset_menu() -> None:
    d = _dims("TV too long", "watching tv for more than 15 minutes",
              shape="state", category="routine")
    assert d["eventing"] == {"entry": {"m": 5, "t_seconds": 900}}
    # Point shapes never get entry windows, whatever the words say.
    d2 = _dims("Door", "door open for 15 minutes", shape="point_in_time",
               category="safety")
    assert d2["eventing"] is None


def test_env_wins_over_anyone_and_still_wins_over_env_context() -> None:
    d = _dims("Door watch", "alert if someone leaves the door open")
    assert d["subject_scope"] == "environment"   # the door is the subject
    d2 = _dims("Phone in bed", "on the phone lying in bed",
               shape="state", category="routine")
    assert d2["evidence_branch"] == "person_still"
