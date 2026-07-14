"""Compiler v2 — custom detectors classified into the six dimensions
(detection v2, P5).

Three jobs, all pure rules over the caregiver's words (NO extra LLM
call — the shape/category signal already comes from
`infer_detector_config`, and every mapping here is a conservative
keyword rule whose misses degrade to today's behavior, never to
detection loss; the P6 soak re-litigates any rule with golden data):

  1. HARD REJECTS — the measured zeros. Asks that sit below the
     sensor floor (July 2026 experiments) are refused at creation
     with a plain-language reason instead of shipping a detector
     that can never work.
  2. RESERVED-FIELD CHECK — backend-stamped analyses keys can never
     be claimed by a custom schema (silent-overwrite guard; P2/P3
     review carry-over).
  3. DIMS CLASSIFICATION — evidence_branch / subject_scope / bucket /
     duty_cycle / needs_audio / eventing / degradation_warning for
     the row, so the scheduler treats a compiled custom exactly like
     a built-in. Unclassifiable text falls back to the legacy default
     (wide call, every tick — today's behavior).
"""

from __future__ import annotations

import re

# ── 1. Hard rejects (measured zeros — spec §0/§5, refuse list) ────
# Matched conservatively against name+text; only unambiguous phrasings
# reject. Plain-language reasons are shown verbatim to the caregiver.
_HARD_REJECTS: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"\beye\s*(twitch|flutter|blink(ing)?\s*rapidly)", re.I),
        "Eye twitches and flutters are too small and too fast for the "
        "camera to see reliably — in our testing this could not be "
        "detected at all (0 of 30 attempts). We don't want to give you "
        "an alert you can't trust.",
    ),
    (
        re.compile(
            r"\b(fine|slight|subtle)\s+(hand\s+)?trem(or|bling)|"
            r"\bparkinson'?s?\s+tremor|\bshiver(ing)?\b", re.I,
        ),
        "Fine tremors and shivering move less than a pixel at the "
        "camera's distance and frame rate — our testing showed they "
        "cannot be told apart from normal stillness. A seizure-scale "
        "movement IS detectable (see the Seizure detector).",
    ),
    (
        re.compile(
            r"\b(wearing|wears|has\s+on)\s+(a\s+|her\s+|his\s+|their\s+)?"
            r"(ring|earring|bracelet)\b", re.I,
        ),
        "Ring-sized objects are smaller than what the camera can "
        "resolve from across a room — in testing this failed on every "
        "attempt. Larger wearables (a medical collar, a watch on a "
        "table) can work.",
    ),
    (
        re.compile(r"\bheel\s+tap(ping)?\b", re.I),
        "Heel tapping is below the camera's motion floor at room "
        "distance — our testing could not detect it reliably.",
    ),
)

# ── 2. Reserved analyses keys (backend-stamped; P5 compile check) ──
_RESERVED_PREFIXES = (
    "sig_", "sched_", "identity_", "vlm_", "rapid_",
)
_RESERVED_KEYS = frozenset({
    "motion_src", "people_present_src", "people_present", "people_count",
    "woke_by_backend", "wake_env_motion", "primary_detection",
    "skipped_reason", "description", "speech_heard", "speech_transcript",
})


class DetectorRejected(ValueError):
    """Raised at creation when the ask is physically undetectable or
    its schema collides with reserved keys. `.reason` is caregiver-
    facing plain language."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def reject_reason(name: str, user_text: str) -> str | None:
    """Plain-language rejection for a measured-zero ask, else None."""
    hay = f"{name}\n{user_text}"
    for pattern, reason in _HARD_REJECTS:
        if pattern.search(hay):
            return reason
    return None


def validate_schema_fields(schema_json: dict) -> None:
    """Reject a compiled schema that claims backend-stamped keys —
    the backend silently wins at merge time today; make it a loud
    compile failure instead."""
    for key in schema_json:
        if key in _RESERVED_KEYS or key.startswith(_RESERVED_PREFIXES):
            raise DetectorRejected(
                f"The field name {key!r} is reserved by the system. "
                "Please rephrase the detector so it doesn't use it."
            )


# ── 3. Dims classification (conservative keyword rules) ──────────

_ENV_SUBJECT = re.compile(
    r"\b(stove|oven|burner|kettle|faucet|tap|sink|door|window|gate|"
    r"fridge|freezer|light(s)?|lamp|tv\s+left\s+on|fire|smoke|water\s+"
    r"(leak|running)|spill)\b", re.I,
)
_ANYONE_SUBJECT = re.compile(
    r"\b(visitor|stranger|intruder|anyone|any\s+person|someone\s+"
    r"(new|unknown|else)|delivery|mask(ed)?)\b", re.I,
)
_AUDIO_EVIDENCE = re.compile(
    r"\b(call(s|ing)?\s+(out|for)|yell(s|ing)?|scream(s|ing)?|shout|"
    r"cry(ing)?|says?\b|saying|asks?\b|asking|talk(s|ing)?|moan(s|ing)?|"
    r"groan(s|ing)?|verbal|repeat(s|ing)?\s+(words|questions)|swear)", re.I,
)
_STILL_ACTIVITY = re.compile(
    r"\b(watch(es|ing)?\s+tv|on\s+(the\s+)?phone|read(s|ing)?|"
    r"sit(s|ting)?|sleep(s|ing)?|ly(ing|es)\s+(down|in\s+bed)|"
    r"motionless|not\s+moving|napping)\b", re.I,
)
_DISTANCE_CLASS = re.compile(
    r"\b(pill|tablet|small|tiny|fidget(ing)?|wring(ing)?|pick(s|ing)?\s+"
    r"at|scratch(es|ing)?|finger|hand[- ]?motion|utensil|cigarette|"
    r"remote)\b", re.I,
)
# "for 10 minutes", "more than 5 min", "at least an hour"
_DURATION = re.compile(
    r"\b(?:for|than|least|over)\s+(?:about\s+)?(\d+|an?)\s*"
    r"(hour|hr|minute|min)s?\b", re.I,
)

# Entry preset menu (locked: dials come from a small menu, shown in
# plain words on the review screen — never free-form). Snapped to the
# nearest by the stated duration.
_ENTRY_PRESETS: tuple[tuple[int, int], ...] = (
    (3, 300),    # "a few minutes" — ≥3 qualifying ticks in 5 min
    (5, 900),    # "10-15 minutes"
    (8, 1800),   # "half an hour or more"
)


def _entry_preset_for(user_text: str, shape: str) -> dict | None:
    """States whose CLAIM includes a duration get a windowed entry
    preset; everything else is on-sight (locked decision: entry speed
    is independent of anti-spam, which the episode gap owns)."""
    if shape != "state":
        return None
    m = _DURATION.search(user_text)
    if not m:
        return None
    qty = 1 if m.group(1).lower() in ("a", "an") else int(m.group(1))
    seconds = qty * (3600 if m.group(2).lower().startswith(("hour", "hr")) else 60)
    best = min(_ENTRY_PRESETS, key=lambda p: abs(p[1] - seconds))
    return {"entry": {"m": best[0], "t_seconds": best[1]}}


def classify_custom_dims(
    *, name: str, user_text: str, shape: str, category: str,
) -> dict:
    """Row values for a compiled custom detector. Every rule biased so
    a miss lands on today's behavior (wide call, patient scope, every
    tick), never on a silently-dead detector."""
    hay = f"{name}\n{user_text}"

    if _ENV_SUBJECT.search(hay) and not _STILL_ACTIVITY.search(hay):
        branch, subject, bucket = "environment", "environment", "snapshot"
    elif _STILL_ACTIVITY.search(hay):
        branch, subject, bucket = "person_still", "patient", "person"
    else:
        branch, subject, bucket = "person_motion", "patient", "scene"

    if subject != "environment" and _ANYONE_SUBJECT.search(hay):
        subject = "anyone"

    duty = (
        "sampled"
        if shape == "state" and category == "routine"
        else "every_eligible_tick"
    )

    eventing = _entry_preset_for(user_text, shape)

    return {
        "evidence_branch": branch,
        "subject_scope": subject,
        "bucket": bucket,
        "duty_cycle": duty,
        "needs_audio": bool(_AUDIO_EVIDENCE.search(hay)),
        "eventing": eventing,
        "degradation_warning": (
            "distance_class" if _DISTANCE_CLASS.search(hay) else "none"
        ),
    }
