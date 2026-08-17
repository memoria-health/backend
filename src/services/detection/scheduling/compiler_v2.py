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
  3. DIMS CLASSIFICATION — requires_motion / subject_scope / bucket /
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
# Each entry: (match, unless, reason). `unless` suppresses the reject
# when the ask ALSO carries detectable-scale language — "violent
# shaking, not just shivering" is a seizure ask using shivering as
# CONTRAST, and a false reject is permanent detection loss at
# creation, the one failure this module forbids (P5 review F1).
_SEIZURE_SCALE = re.compile(r"\b(shak(e|es|ing)?|seizure|convuls\w*|violent\w*|whole[- ]?body|thrash\w*)\b", re.I)
_HARD_REJECTS: tuple[tuple[re.Pattern, re.Pattern | None, str], ...] = (
    (
        re.compile(r"\beye\s*(twitch|flutter|blink(ing)?\s*rapidly)", re.I),
        None,
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
        _SEIZURE_SCALE,
        "Fine tremors and shivering move less than a pixel at the "
        "camera's distance and frame rate — our testing showed they "
        "cannot be told apart from normal stillness. A convulsion-scale "
        "movement IS detectable (see the Convulsion detector).",
    ),
    (
        # ring/earring only — a bracelet is wrist-scale, NOT a measured
        # zero (the collar program precedent; P5 review F2).
        re.compile(
            r"\b(wearing|wears|has\s+on)\s+(a\s+|her\s+|his\s+|their\s+)?"
            r"(ring|earring)s?\b", re.I,
        ),
        None,
        "Ring-sized objects are smaller than what the camera can "
        "resolve from across a room — in testing this failed on every "
        "attempt. Larger wearables (a medical collar, a watch on a "
        "table) can work.",
    ),
    (
        re.compile(r"\bheel\s+tap(ping)?\b", re.I),
        None,
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
    for pattern, unless, reason in _HARD_REJECTS:
        if pattern.search(hay) and not (unless and unless.search(hay)):
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

# Hazard NOUNS and hazard EVENTS both live here. Morphology matters:
# real caregivers write in the past tense ("water spilled", "a glass
# broke"), so each verb carries its s/ed/ing forms — the bare-stem
# `spill` missed "spilled" and sent a floor-hazard ask to the patient
# branch (compiler eval 2026-08-17). Glass is deliberately NOT matched
# bare: "drinks a glass of water" is a body-scale person ask, so glass
# only counts alongside breakage language (broke/shatter/shards).
_ENV_SUBJECT = re.compile(
    r"\b(stove|oven|burner|kettle|faucet|tap|sink|door|window|gate|"
    r"fridge|freezer|light(s)?|lamp|tv\s+left\s+on|fire|smoke|"
    r"water\s+(leak(s|ed|ing)?|running)|spill(s|ed|ing)?|"
    r"leak(s|ed|ing)?|puddle(s)?|shard(s)?|shatter(s|ed|ing)?|"
    r"broken\s+glass|glass\s+(break(s|ing)?|broke|shatter(s|ed|ing)?|"
    r"smash\w*))\b", re.I,
)
_ANYONE_SUBJECT = re.compile(
    r"\b(visitor|stranger|intruder|anyone|any\s+person|someone\s+"
    r"(new|unknown|else)|delivery|mask(ed)?)\b", re.I,
)
_AUDIO_EVIDENCE = re.compile(
    r"\b(call(s|ing)?\s+(out|for)|yell(s|ing)?|scream(s|ing)?|shout|"
    r"cry(ing)?|says?\b|saying|asks?\b|asking|talk(s|ing)?|moan(s|ing)?|"
    r"groan(s|ing)?|verbal|repeat(s|ing)?\s+(words|questions)|swear|"
    # Agitation/distress asks: the state is substantially VOCAL (moaning,
    # muttering, repeated questions) — the built-in agitation detector
    # carries needs_audio=True for the same reason, and a custom compiled
    # without audio would judge a shouting spell on picture alone.
    r"agitat(ed|es|ion|ing)?|distress(ed)?)", re.I,
)
_STILL_ACTIVITY = re.compile(
    r"\b(watch(es|ing)?\s+tv|on\s+(the\s+)?phone|read(s|ing)?|"
    r"sit(s|ting)?|sleep(s|ing)?|ly(ing|es)\s+(down|in\s+bed)|"
    r"motionless|not\s+moving|napping)\b", re.I,
)
# Exit/leaving language means the PERSON is the subject even when a
# door is mentioned — 'leaves through the front door' is a wandering
# ask, not a door-state check (P5 review F4: env branch gates on
# motion OUTSIDE person boxes post-soak, which would skip exactly the
# tick that matters).
#
# 'leave' only counts with a PLACE or a direction, mirroring how 'left'
# always did: bare `leav(e|es|ing)` also matched the transitive sense —
# "shatters and LEAVES shards", "leaves the water running" — and that
# vetoed the environment branch on a floor-hazard ask (compiler eval
# 2026-08-17). A leaving-the-house ask always names where or which way.
_EXIT_VERBS = re.compile(
    r"\b(leav(e|es|ing)\s+(through|out|via|for)\b|"
    r"leav(e|es|ing)\s+(the\s+|her\s+|his\s+|their\s+)?"
    r"(house|home|room|building|apartment|bed)\b|"
    r"left\s+(the\s+)?(house|home|room|building)|"
    r"exit(s|ing)?|walk(s|ing)?\s+out|wander(s|ing)?|go(es|ing)?\s+"
    r"out(side)?|knock(s|ing)?)\b", re.I,
)
_DISTANCE_CLASS = re.compile(
    r"\b(pill|tablet|small|tiny|fidget(ing)?|wring(ing)?|pick(s|ing)?\s+"
    r"at|scratch(es|ing)?|finger|hand[- ]?motion|utensil|cigarette|"
    r"remote)\b", re.I,
)
# Actions whose EVIDENCE lives at the person's own scale — a hand, an
# object, the space between a hand and a face — not at room scale.
#
# These must compile to the `person` bucket so the scheduler crops to
# the body. Answering them from a full 1080p room is not a degraded
# answer, it is a wrong one: the object is a few dozen pixels after
# downsampling, so the model has nothing to read and describes what is
# LIKELY instead. That is not hypothetical —
#
#   - `medication` (preset, bucket=person) once ran full-scene at 4fps
#     and "reported a pill being swallowed while the man held a camera"
#     (scheduler.py:309).
#   - a custom `drinking water` ask compiled to SCENE and produced 11
#     false fires on 2026-08-06 and 8 more in one hour on 08-07,
#     inventing a cup in the `seen` field itself while the subject sat
#     with both hands on a laptop (verified against the clips).
#
# The second is why this class exists. `_STILL_ACTIVITY` was the only
# route to `person`, so every ACTIVE body-scale ask — drinking, eating,
# taking medication, grooming — fell through to `person_motion` and
# `scene`, which is the one combination guaranteed to hallucinate.
_BODY_SCALE_ACTION = re.compile(
    r"\b(drink(s|ing)?|sip(s|ping)?|swallow(s|ing)?|eat(s|ing)?|"
    r"chew(s|ing)?|bite(s)?|feed(s|ing)?|"
    # Choking evidence is hands-to-throat and face colour — body scale,
    # exactly why the choking PRESET is bucket=person; a custom "grandpa
    # is choking" must land in the same crop. `meds` is the everyday
    # word for medication ("did he take his meds") the formal stems miss.
    r"chok(e|es|ed|ing)|meds|"
    r"medicat(e|es|ion)|pill(s)?|tablet(s)?|dose|inhaler|"
    r"brush(es|ing)?\s+(teeth|hair)|shav(e|es|ing)|groom(s|ing)?|"
    r"wip(e|es|ing)|rub(s|bing)?|touch(es|ing)?\s+(face|mouth|head)|"
    r"hand\s+to\s+(mouth|face)|to\s+(his|her|their)\s+(mouth|lips|face))\b",
    re.I,
)
def _entry_confidence_for(
    is_sustained: bool, bucket: str, degradation: str, category: str,
) -> dict | None:
    """The entry M-of-N — an INTERNAL confidence gate, NOT a user duration.

    How many corroborating detections we require before believing a
    SUSTAINED state, scaled to how HARD the activity is to see reliably
    (Aryaman 2026-07-26). A caregiver never dials this; it exists to cut
    the false positives that distance + lighting cause on hard-to-detect
    things. Easy-to-see states fire on the first confident detection;
    small/fine-detail or rapid-motion states require corroboration.
    Emergencies must not wait, so safety stakes cap corroboration lower
    than routine.

    Only SUSTAINED detectors accumulate entry evidence — instant events
    fire on a single tick, so their corroboration is the WITHIN-CLIP
    confirm (k-of-N): on a fire, the same clip is re-judged by fresh
    calls that never see the first call's answer, and the fire is
    suppressed unless k of n agree. Independent re-judging is the
    variant the verification literature actually validates — a checker
    that can read the draft happily re-copies the draft's hallucination
    (Chain-of-Verification, arXiv:2309.11495).

    Derived 2026-08-08 from the drinking-water FP episode: an instant
    routine custom fired 35×/day at confidence 0.9-1.0 on confabulated
    drinks, and the confirm dial — named below since P4 but never set
    by any code path — existed precisely for this. Safety instants stay
    on-sight: a real fall must not risk being outvoted by two noisier
    re-checks, the same stakes carve-out the sustained branch makes.
    Cost is bounded by the fire rate, not the tick rate (~n-1 extra
    calls per fire), so a well-behaved detector pays nothing.

    Windows satisfy `t_seconds >= m * scheduler_tick` (~60 s) so M is
    physically reachable. TODO(soak): the exact (m, t) per tier are
    placeholders to calibrate against golden soak data before prod.
    """
    if not is_sustained:
        if category == "safety":
            return None
        return {"confirm": {"k": 2, "n": 3}}
    if degradation == "distance_class":
        m, t = (2, 180) if category == "safety" else (3, 300)
    elif bucket in ("rapid", "fine_motion"):
        m, t = (2, 180)
    else:
        return None  # easy to see → fire on the first confident detection
    return {"entry": {"m": m, "t_seconds": t}}


def apply_confirm_need(eventing: dict | None, confirm_need: str | None) -> dict | None:
    """Relax the instant-custom confirm gate when the COMPILER judged the
    evidence low-risk (Aryaman 2026-08-14: confirmation should follow
    evidence risk — subtle motion, small-object interaction, confounding
    context — not be a blanket default).

    Deliberately narrow: it can ONLY turn the exact instant-non-safety
    default `{"confirm": {"k": 2, "n": 3}}` into None (fire on sight).
    Sustained entry windows, safety instants (already None), and any
    hand-tuned eventing pass through untouched. confirm_need None or
    anything but "low" → unchanged, so a missing/garbled compiler line
    lands on today's behavior — the 2026-08-08 confabulation storm
    (35 fires/day on imagined drinks) is why the gate must never be
    dropped by accident.
    """
    if confirm_need == "low" and eventing == {"confirm": {"k": 2, "n": 3}}:
        return None
    return eventing


def classify_custom_dims(
    *, name: str, user_text: str, is_sustained: bool, category: str,
) -> dict:
    """Row values for a compiled custom detector. Every rule biased so
    a miss lands on today's behavior (wide call, patient scope, every
    tick), never on a silently-dead detector."""
    hay = f"{name}\n{user_text}"

    if (
        _ENV_SUBJECT.search(hay)
        and not _STILL_ACTIVITY.search(hay)
        and not _EXIT_VERBS.search(hay)
    ):
        branch, subject, bucket = "environment", "environment", "scene"
    elif _STILL_ACTIVITY.search(hay):
        branch, subject, bucket = "person_still", "patient", "person"
    elif _BODY_SCALE_ACTION.search(hay):
        # Moving, but the evidence is at body scale — crop to the person.
        # Checked BEFORE the person_motion default, which is what sent
        # every drinking/eating/medication ask to the whole room.
        branch, subject, bucket = "person_motion", "patient", "person"
    else:
        branch, subject, bucket = "person_motion", "patient", "scene"

    if subject != "environment" and _ANYONE_SUBJECT.search(hay):
        subject = "anyone"

    duty = (
        "thinned"
        if is_sustained and category == "routine"
        else "every_eligible_tick"
    )

    # rapid_frequency outranks distance_class: high-frequency motion
    # (tremor-band, 3-6 Hz) sits below VLM video sampling — a temporal
    # ceiling the benchmark measured directly (Convulsion was removed
    # from the catalog over it). Any custom ask that compiles into the
    # rapid/fine_motion lanes must surface a high-FP/FN warning.
    if bucket in ("rapid", "fine_motion"):
        degradation = "rapid_frequency"
    else:
        degradation = "distance_class" if _DISTANCE_CLASS.search(hay) else "none"
    # M-of-N is derived from detection difficulty (bucket + degradation +
    # stakes), never from a user-stated duration.
    eventing = _entry_confidence_for(is_sustained, bucket, degradation, category)

    return {
        # Only person_motion vs not is read downstream (the motion floor);
        # environment/person_still collapse to requires_motion=False.
        "requires_motion": branch == "person_motion",
        "subject_scope": subject,
        "bucket": bucket,
        "duty_cycle": duty,
        "needs_audio": bool(_AUDIO_EVIDENCE.search(hay)),
        "eventing": eventing,
        "degradation_warning": degradation,
    }
