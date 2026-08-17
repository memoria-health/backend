"""Detector catalog — the single source of truth for every built-in detector.

This is the "detection library." Each `CatalogEntry` carries everything the
runtime needs (identifier / prompt_fragment / schema_json) plus the catalog
metadata (cohort, recommended, promotion threshold). Three consumers read from
here, so they can never drift out of sync:

  - `onboarding.seed_detectors_for_patient` — seeds the `recommended` subset for
    the patient's cohort on patient create (the original auto-seed behaviour).
  - `api/detectors.py` — the library list + "add from library" endpoints.
  - `detection.event_promotion` — per-detector confidence thresholds.

Entries are `source="preset"` when instantiated, so they inherit every existing
preset behaviour (auto-attach to newly-paired cameras via
`camera_detector_repo.sync_links_for_patient`, "Possible …"
event titles) with no new branching.

The 6 original presets (fall / agitation / wandering / medication / climbing /
distress) moved here VERBATIM from `onboarding.py`. The 16 library detectors are
new; their prompts follow the same `<id>_detected / _confidence / _reasoning`
contract and need the same field tuning as any prompt.

NOTE: `people_present` is NOT here — it's the always-on block synthesised in
`detection/prompt.py:PEOPLE_PRESENT_SPEC`, never a per-patient row.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

# ══════════════════════════════════════════════════════════════════
# Prompt fragments
# ══════════════════════════════════════════════════════════════════
# The first six are ported VERBATIM from the previous
# `onboarding.py` so seeding output is byte-identical. Do not reword
# them here without intending to change every patient's detector.

# Sleep (2026-07-28, Aryaman's design): the DETECTOR only labels burst
# ticks — ticks are motion-gated, so a still sleeper generates none and
# the tick SILENCE between labeled moments is the sleep signal. The
# nightly state machine (sleep_summary.py) derives bedtime / onset /
# disturbances / out-of-bed / wake from these labels + the gaps. Cost:
# rides the shared scene call on ticks that fire anyway (~1-2% prompt
# growth); a quiet night is $0. Pin to the lite tier when per-detector
# difficulty routing lands — this is an EASY visual task.
_SLEEP_PROMPT_FRAGMENT = (
    # DELIBERATELY UNDER-SPECIFIED (Aryaman 2026-08-03). The previous
    # version enumerated posture, head position and eye state, and
    # crucially allowed "judge by the settled sleeping posture alone"
    # when the face was not visible. The model took that route by
    # default — on the 08-02 night every tick around the reported onset
    # reasoned from posture and NONE mentioned eyes, so it called sleep
    # at 02:19 while he lay awake until ~02:50. The camera has IR and
    # reads eye state clearly (it said "eyes closed" freely from 04:00
    # onward), so the fallback was never needed. The model knows what
    # sleep is; asking plainly beats dictating a checklist it can
    # satisfy without looking.
    "sleep_detected: boolean. TRUE only if the person is genuinely "
    "ASLEEP — unresponsive, not reacting to their surroundings. Lying "
    "still is not enough: someone resting, settling, or lying awake in "
    "the dark is NOT asleep. If you cannot tell, answer FALSE.\n"
    # TRANSITIONS OUTRANK STATES (2026-08-03). Two bed-crossing EVENTS
    # carry the whole night: our state machine knows the person is in bed
    # from `entering_bed` until `leaving_bed` and asks nothing in
    # between, so an occluded sleeper never has to be re-detected.
    #
    # `leaving_bed` already worked — 3 ticks on the 08-02 night, all
    # correct, matching the resident's own account to the minute. But
    # `settling` fired ZERO times across ~1350 ticks, because the old
    # closing line said "pick the LAST state visible" and the last state
    # after getting into bed is simply lying there. The event was being
    # overwritten by the state that followed it. Hence the explicit
    # priority rule below, and renaming `settling` -> `entering_bed` so
    # both crossings read as the events they are.
    "sleep_activity: string, exactly one of: 'entering_bed' (a person "
    "GETS INTO the bed — climbs in, sits down onto it and lies back, "
    "pulls the covers over themselves), 'leaving_bed' (a person GETS "
    "OUT of the bed — pushes the covers off and stands, or is already "
    "out and moving away), 'stirring' (movement while staying lying "
    "down — rolling over, shifting), 'sitting_up' (rising to a sitting "
    "position but STAYING in the bed), 'awake_in_bed' (in bed and "
    "clearly awake — eyes open, talking, on the phone, using a device), "
    "'none' (no bed-related activity visible).\n"
    "PRIORITY: if the person CROSSES INTO or OUT OF the bed at any point "
    "in the clip, report that crossing — 'entering_bed' or "
    "'leaving_bed' — even if they settle into some other state "
    "afterwards. A crossing is never overwritten by what follows it. "
    "Only when no crossing occurs do you report the last state you "
    "saw.\n"
    "sleep_confidence: number 0-1. 0.7+ = what you reported is clearly "
    "visible.\n"
    "sleep_reasoning: 1 short sentence describing what you saw."
)

# Activity (2026-07-28, Aryaman): "actual physical activity and
# movement" — engagement, not raw pixel motion. Seated EXERCISE is
# active; seated fidgeting is sitting. Same transition-labeling economy
# as sleep: labels ride motion ticks, silence persists the last state
# (a posture change is physically a motion event), and the daily
# summarizer time-weights the timeline. record_only — never an event.
# POSTURE DOES NOT DECIDE THIS (2026-08-07, measured live).
#
# The fragment already said "exercising in ANY posture including seated
# weights or floor exercises" and still returned `sitting` — at
# confidence 1.0 — for continuous overhead arm raises, with `seen`
# correctly reading "their arms are raised above their head for the
# duration" and the reasoning saying "sitting passively with their arm
# raised". Adding more instruction in that direction had already been
# tried; the model was not missing the rule, it was taking an exit.
#
# The exit was "regardless of small hand/arm movements" on `sitting`.
# It let repeated whole-arm exercise be filed as a small hand movement
# and stop the analysis there. So the fix is to REMOVE the escape
# hatch, not to restate the rule: `sitting` now requires the body to be
# at rest, and the tie-break is stated explicitly in the direction that
# matters.
#
# Why that direction: chair-based exercise is the prescribed
# intervention for frail older adults precisely because standing and
# walking are not available to them. Filing it as sedentary does not
# just cost precision — it erases the only exercise that cohort does,
# and the aggregate then reports lethargy, which is the alarming
# reading rather than the safe one.
_ACTIVITY_PROMPT_FRAGMENT = (
    "activity_detected: boolean. TRUE if the person is PHYSICALLY ACTIVE "
    "— walking, exercising, doing chores, or otherwise physically "
    "engaged in a task. Exercise COUNTS WHEREVER IT HAPPENS, including "
    "sitting down: repeated arm raises, leg lifts, stretching, marching "
    "in a chair, or using weights or a band while seated are ACTIVE. "
    "FALSE if they are lying down, sitting at rest (reading, watching "
    "TV, using a phone), or standing still.\n"
    "activity_state: string, exactly one of: 'lying', 'sitting_still', "
    "'sitting_moving', 'standing', 'active'.\n"
    "  - 'sitting_still': seated and the body is at rest. Small "
    "incidental hand movement still counts as still — typing, holding a "
    "phone, turning a page.\n"
    "  - 'sitting_moving': seated, but the arms, legs or trunk are "
    "MOVING REPEATEDLY. Report what the body is doing; do NOT try to "
    "decide whether it is exercise or conversation. Overhead arm "
    "raises, leg lifts, marching in a chair, stretching, using a band "
    "or weights, AND animated gesturing all belong here.\n"
    "  - 'standing': upright but not going anywhere.\n"
    "  - 'active': walking, chores, or whole-body movement while "
    "upright.\n"
    "Being seated does not by itself make the state 'sitting_still'. "
    "Pick the state that best describes MOST of the clip.\n"
    "activity_confidence: number 0-1.\n"
    "activity_reasoning: 1 short sentence describing the activity level."
)

_FALL_PROMPT_FRAGMENT = (
    "fall_detected: boolean. TRUE if a person falls to the ground in the "
    "clip — they collapse, slip, trip, stumble, sag, or drop and end up down "
    "on the floor. If someone goes down to the ground, that's a fall. FALSE "
    "otherwise: they stay upright, walk around or out of the room normally, "
    "or deliberately and slowly sit, lie down, or kneel.\n\n"
    "The camera is mounted high near the ceiling, so a person walking toward "
    "it and shrinking out the bottom of the frame is passing BENEATH the "
    "camera (normal navigation), not a fall.\n\n"
    "fall_confidence: number 0–1. 0 = no fall. 0.7+ = a person clearly went "
    "down to the floor.\n\n"
    "fall_reasoning: 1 short sentence describing what you saw. Always present."
)

_WANDERING_PROMPT_FRAGMENT = (
    "wandering_detected: boolean. TRUE ONLY if you SEE the ENTIRE person walk "
    "COMPLETELY out of view through a doorway / door frame — they were fully "
    "inside the room earlier in the clip, and by the END of the clip NO part of "
    "them (body, head, arms, or legs) is visible ANYWHERE in the frame, they left "
    "specifically by passing through a visible doorway or door frame, and they "
    "have not returned. The WHOLE person must be entirely gone from view.\n"
    "\n"
    "Set FALSE if ANY of:\n"
    "  - ANY part of the person is still visible in the frame — even if only "
    "their body remains while their face/head has moved out of view, or only a "
    "limb or partial figure is visible. If you can still see any part of them, "
    "they have NOT left;\n"
    "  - the person is only partially out, standing in the doorway, or halfway "
    "through;\n"
    "  - the person leaves by walking off the EDGE of the frame somewhere that is "
    "NOT a clearly visible doorway/door frame (e.g. off to the side, behind "
    "furniture) — it must be THROUGH a door frame to count;\n"
    "  - the person was outside and walks IN — entering does not count, only "
    "leaving does;\n"
    "  - the person only approaches, touches, opens, or looks toward a door but "
    "does not walk all the way through it;\n"
    "  - the person leaves and returns within the same clip;\n"
    "  - the person stays within the room — moving around inside is not "
    "wandering;\n"
    "  - no person is visible at any point.\n"
    "\n"
    "wandering_confidence: number 0–1. 0 = person still visible / stayed inside / "
    "left by a non-doorway edge. 0.3–0.6 = ambiguous (partly out, or unclear "
    "whether they fully passed through a door frame). 0.7+ = the ENTIRE person "
    "clearly walked out through a doorway and NO part of them remains visible.\n"
    "wandering_reasoning: one short sentence — e.g. 'entire person walked out "
    "through the doorway at 8s, no longer visible in frame', 'person's body still "
    "visible in frame — has not left (not wandering)', 'person walked off the "
    "side of the frame, not through a doorway', 'no person visible'."
)


def _door_region_phrase(box: list[float]) -> str:
    """Human region for a normalized [x, y, w, h] door box — e.g. 'upper left',
    'center', 'lower right'. Used to describe the selected exit door to the VLM."""
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    horiz = "left" if cx < 0.34 else "right" if cx > 0.66 else "center"
    vert = "upper" if cy < 0.34 else "lower" if cy > 0.66 else ""
    if vert and horiz != "center":
        return f"{vert} {horiz}"
    if vert:  # center column
        return f"{vert} center"
    return horiz


def _door_where_phrase(door: dict) -> str:
    """'the upper right of the camera view (the far right hallway)' for one door."""
    region = _door_region_phrase(door["box"])
    where = f"the {region} of the camera view"
    label = str(door.get("label") or "").strip()
    if label:
        where += f" (the {label})"
    return where


def build_wandering_fragment(doors: list[dict] | None) -> str:
    """Return the wandering `prompt_fragment` specialized to the caregiver's
    chosen exit door(s). `doors` is a list of `{box, label, ...}` (may be empty).

    With selected doors, the per-tick VLM is told WHICH doorways are the exterior
    exits so it only fires when the person leaves THROUGH one of them — leaving
    via an interior doorway (bathroom, bedroom) or a frame edge no longer counts.
    With no doors, falls back to the generic any-doorway fragment.
    """
    valid = [d for d in (doors or []) if len(d.get("box") or []) == 4]
    if not valid:
        return _WANDERING_PROMPT_FRAGMENT

    if len(valid) == 1:
        targeting = (
            "THIS ROOM HAS ONE EXTERIOR EXIT DOOR — the ONLY doorway that counts "
            f"as wandering. It is located in {_door_where_phrase(valid[0])}. A "
            "person counts as wandering ONLY if they walk COMPLETELY out of view "
            "THROUGH THAT specific exit doorway. Leaving the camera's view in ANY "
            "other way is NOT wandering — through another doorway (bathroom, "
            "bedroom, closet), off ANY edge of the frame, behind furniture, or "
            "out of sight anywhere else — set FALSE for ALL of those. ONLY a "
            "full exit through that one designated exit door counts. Everything "
            "below still applies to that one exit door.\n\n"
        )
    else:
        listed = "; ".join(
            f"({i + 1}) {_door_where_phrase(d)}" for i, d in enumerate(valid))
        targeting = (
            f"THIS ROOM HAS {len(valid)} EXTERIOR EXIT DOORS — the ONLY doorways "
            f"that count as wandering: {listed}. A person counts as wandering ONLY "
            "if they walk COMPLETELY out of view THROUGH ONE of these specific exit "
            "doorways. Leaving the camera's view in ANY other way is NOT "
            "wandering — through another doorway (bathroom, bedroom, closet), off "
            "ANY edge of the frame, behind furniture, or out of sight anywhere "
            "else — set FALSE for ALL of those. ONLY a full exit through one of "
            "these designated exit doors counts. Everything below still applies "
            "to these exit doors.\n\n"
        )
    return targeting + _WANDERING_PROMPT_FRAGMENT


_AGITATION_PROMPT_FRAGMENT = (
    "agitation_detected: boolean. Prioritize PRECISION over recall: set "
    "TRUE only when there is CLEAR, OBVIOUS evidence of genuine "
    "distress that leaves no room for doubt. Casual or ordinary "
    "activity is NOT agitation. When unsure, set FALSE.\n"
    "\n"
    "There are two independent kinds of evidence. EITHER one alone is "
    "enough — but only when it is unambiguous:\n"
    "\n"
    "PHYSICAL — clear, obvious physical distress, such as:\n"
    "  - PACING in the clearest sense: repeatedly walking back and "
    "forth in an agitated, purposeless way — NOT simply crossing the "
    "room or walking over to do something;\n"
    "  - frantic running or rushing about in evident distress;\n"
    "  - frantic, disordered rummaging — agitatedly tearing through or "
    "tossing things, NOT calm searching or tidying;\n"
    "  - throwing objects, or banging / striking / slamming surfaces or "
    "objects forcefully and angrily.\n"
    "The action must clearly read as distress or anger. If it has an "
    "ordinary explanation — exercise, dancing, play, sport, swatting an "
    "insect, normal chores, reaching for or carrying something, moving around the house — it is "
    "NOT agitation.\n"
    "\n"
    "VERBAL — clear distress in BOTH tone and content, from the person "
    "on camera:\n"
    "  - TONE / affect: shouting, sharp, harsh, or hostile voice, or "
    "sustained crying / groaning / yelling; AND\n"
    "  - CONTENT: angry, hostile, or distressed words — cursing, "
    "threats, distressed demands, or calls for help.\n"
    "Both tone AND content must point to distress. Ordinary "
    "conversation does NOT count — a normal phone call, a minor "
    "argument or debate, or an animated explanation has a "
    "conversational tone and topical content, not hostility or "
    "distress.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the movement is ordinary or calm — walking to a destination, "
    "doing chores, searching calmly, exercising, dancing, playing, or "
    "swatting an insect;\n"
    "  - there is motion but you cannot clearly tell it is distress "
    "rather than ordinary activity;\n"
    "  - the only voice is ordinary conversation, argument, or "
    "explanation, or it likely comes from a TV / radio / phone / device "
    "or cannot be attributed to the person on camera;\n"
    "  - the person is sitting calmly, resting, or sleeping;\n"
    "  - no person is visible, or the person is too far / too dark or "
    "the audio too unclear to judge — when you cannot tell, set "
    "FALSE.\n"
    "\n"
    "agitation_confidence: number 0–1. 0 = calm / ordinary activity / "
    "no person / not observable. 0.3–0.6 = some raised voice or "
    "vigorous motion, but you cannot rule out ordinary activity (it "
    "will not alert here — that is intended). 0.7+ = clear, unambiguous "
    "physical distress OR clear distressed tone AND content; raise "
    "toward 0.9 when both occur together.\n"
    "agitation_reasoning: 1 short sentence stating the clear evidence "
    "you saw or heard, or why it was not agitation (e.g. 'threw objects "
    "and banged the table repeatedly in anger', 'paced frantically back "
    "and forth for ~8s with no purpose', 'shouted curses and threats "
    "while sitting still', 'forceful arm motion but it was swatting a "
    "bug — not distress', 'raised voice but an ordinary phone "
    "argument', 'no person visible')."
)

_MEDICATION_PROMPT_FRAGMENT = (
    "medication_taken: boolean. Set TRUE when there is good evidence the "
    "person took medication, by ANY of the patterns below. Aim to CATCH "
    "genuine medication-taking — including the ordinary case where you can't "
    "see the bottle — but don't fire on plain eating or a bare drink with no "
    "pill-taking gesture at all.\n"
    "\n"
    "Accept TRUE if ANY of:\n"
    "\n"
    "(A) CONTAINER — you can see a medicine container (pill bottle, blister / "
    "foil pack, or weekly pill organizer) AND the person takes something from "
    "it toward their mouth. A drink before, during, or after supports it but "
    "is NOT required.\n"
    "\n"
    "(B) PILL-TAKING GESTURE — the KEY case, and you will usually NOT be able "
    "to see the pill itself: this camera is mounted far away, so a pill is only "
    "a few pixels and effectively invisible. DO NOT require a visible pill or "
    "bottle. Instead read the MOTION: the person brings a hand up to their mouth "
    "to PLACE something on the tongue — fingers or fingertips to the lips, with "
    "the hand NOT holding a cup, glass, or food at that moment — and then tilts "
    "the head back in a short, deliberate swallow. This is followed or "
    "accompanied by a drink of liquid at some point in the clip (before, during, "
    "or a little later). That distinct empty-hand fingertips-to-mouth placing "
    "motion + head-tilt swallow + a drink is ENOUGH to set TRUE — you do not "
    "need to see the pill. What separates this from ordinary drinking: in a "
    "plain drink the person holds a cup/glass/bottle TO their lips; in pill-"
    "taking the bare hand/fingers go to the mouth to place something, separate "
    "from the drinking action.\n"
    "\n"
    "(C) INJECTION — they clearly give themselves an injection with an obvious "
    "device (insulin pen or syringe) against the body.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - just drinking — a cup, glass, or bottle held to the lips — with NO "
    "separate bare-hand fingertips-to-mouth placing gesture;\n"
    "  - they are clearly EATING food — chewing, a handful of snack, a bite of "
    "a meal;\n"
    "  - a hand goes near the face but there is no distinct placing-on-tongue "
    "motion and no drink anywhere in the clip (too ambiguous);\n"
    "  - for injection, you cannot see the device against the body;\n"
    "  - no person is visible, or the action is too far or unclear to judge.\n"
    "\n"
    "medication_confidence: number 0–1. 0 = no medication behavior / unclear. "
    "0.3–0.6 = a hand-to-mouth motion but you didn't clearly see the placing "
    "gesture or the drink. 0.7+ = a clear medicine container, OR a clear "
    "empty-hand fingertips-to-mouth placing + head-tilt swallow together with a "
    "drink (even though the pill itself isn't visible).\n"
    "medication_reasoning: 1 short sentence stating the motion you saw (e.g. "
    "'brought fingertips to mouth to place something, tilted head back, then "
    "drank water — pill not visible at this distance but the gesture is clear', "
    "'opened a weekly pill organizer and took a pill', 'only drank from a glass, "
    "no separate hand-to-mouth placing gesture', 'no person visible')."
)




# ── New library prompts (14) — starting points, field-tune like any prompt ──








_CALLING_FOR_HELP_PROMPT_FRAGMENT = (
    # REWRITTEN 2026-08-06 (Aryaman). The old rule required "BOTH a distressed
    # delivery AND help-seeking content". Measured against the eval set, that
    # cost us the only true positive we have: the model transcribed "Help me"
    # correctly, attributed it to the person on camera correctly, and then
    # declined because "the delivery is calm and monotone, lacking the
    # distressed tone required". That is the exact failure we cannot ship — a
    # frail, weak, winded or semi-conscious person who has fallen says "help
    # me" QUIETLY. Requiring audible panic optimises for a demo and fails the
    # patient.
    #
    # The discriminator is not TONE, it is whether they are SUMMONING someone
    # versus CONVERSING. "Help me" called out to the room is a summons; "can
    # you help me with this" said into a laptop is conversation. Tone is now
    # only corroboration, never a requirement.
    "calling_for_help_detected: boolean. TRUE if the person on camera ASKS "
    "FOR HELP, in the literal sense — words that request help or rescue for "
    "themselves, in ANY language:\n"
    "  - 'help', 'help me', 'somebody help', 'I need help';\n"
    "  - 'I've fallen', 'I can't get up', 'I'm hurt', 'I can't breathe';\n"
    "  - the same request in another language — judge the MEANING of what "
    "was transcribed, never the language it is in.\n"
    "\n"
    "Delivery does NOT matter. Calm, weak, breathless, quiet or matter-of-"
    "fact all count exactly as much as a shout — a person who has fallen and "
    "is exhausted often cannot raise their voice. Never require distress.\n"
    "\n"
    "Do NOT extend this to attention-seeking in general. Calling a person's "
    "name, calling out to the room, or asking whether anyone is there is NOT "
    "a request for help on its own — people call to each other constantly.\n"

    "\n"
    "Set FALSE if any of:\n"
    "  - the word appears inside ORDINARY CONVERSATION rather than as an "
    "appeal — discussing a task, helping someone else, a work or social call "
    "('can you help me with this file', 'I helped her yesterday'). The test "
    "is whether they are calling someone TO them, here, now;\n"
    "  - `heard` contains no request for help at all — the words are about "
    "something else, or are someone helping with a task;\n"
    "  - nothing intelligible was transcribed. If `heard` is UNINTELLIGIBLE "
    "or SILENCE you have no words to judge, so this is FALSE — do not infer "
    "an appeal from tone, noise, or the person's posture alone.\n"
    "\n"
    "\n"
    "`audio_source` CORROBORATES, it never vetoes. If the words are an "
    "explicit request for help, report TRUE even when you judged the source "
    "to be a device or another room, and lower the confidence instead. "
    "Getting the source wrong is easy on a distant microphone: on "
    "2026-08-06 a real, correctly-transcribed 'Help me / Help me / Please "
    "help me' was thrown away as 'coming from a device'. A phone playing "
    "the word is a rare annoyance; a suppressed real call is the one "
    "failure this detector must never produce.\n"
    "calling_for_help_confidence: number 0–1. 0 = no appeal. 0.3–0.6 = words "
    "that MIGHT be an appeal but could be conversation. 0.7+ = a clear "
    "appeal to be come to or helped, whatever the tone.\n"
    "calling_for_help_reasoning: 1 short sentence quoting the words from "
    "`heard` that decided it (e.g. 'said \"help me\" twice while on the "
    "floor', 'said \"can you help me\" while working at a laptop — "
    "conversation, not a summons', 'audio_source=device')."
)

_CHOKING_PROMPT_FRAGMENT = (
    "choking_detected: boolean. Reason HOLISTICALLY about the entire clip — the "
    "person's posture, hand and arm movements, apparent breathing, facial "
    "expression or distress, and the context (e.g. whether they are eating) — and "
    "set TRUE if, taking everything together, there is a good chance the person "
    "is choking. There is no fixed checklist of required signs; use your overall "
    "judgment about whether this looks like a choking emergency.\n"
    "\n"
    "Camera note: this camera is mounted high and looks down at a steep angle, so "
    "hands raised toward the throat can appear to be at the mouth or chest. Take "
    "that into account rather than dismissing a possible throat grab.\n"
    "\n"
    "Set FALSE if the person is clearly just eating, drinking, talking, laughing, "
    "or sitting calmly with no sign of distress; or if no person is visible or "
    "the scene is too far / unclear to judge.\n"
    "\n"
    "choking_confidence: number 0–1 for how likely the person is choking (0 = "
    "clearly not, ~0.5 = plausible but uncertain, 0.9+ = very likely a choking "
    "emergency).\n"
    "choking_reasoning: 1 short sentence giving your overall judgment (e.g. "
    "'hands at the throat, hunched and appears unable to breathe — likely "
    "choking', 'eating calmly with no distress — not choking', 'no person "
    "visible')."
)

_MEAL_EATEN_PROMPT_FRAGMENT = (
    "meal_eaten_detected: boolean. TRUE only if the person is clearly eating a "
    "meal — seated at a table or with a plate / bowl of food, repeatedly "
    "bringing food to their mouth with a utensil or hand over the clip. The "
    "eating action must be visible.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the person only drinks a beverage, or handles food without eating it;\n"
    "  - someone is merely setting the table, cooking, or cleaning up — not "
    "eating;\n"
    "  - a single ambiguous hand-to-mouth motion with no visible food;\n"
    "  - no person is visible.\n"
    "\n"
    "meal_eaten_confidence: number 0–1. 0 = not eating. 0.3–0.6 = possibly "
    "eating but food / utensil not clearly seen. 0.7+ = clearly eating a "
    "meal.\n"
    "meal_eaten_reasoning: 1 short sentence (e.g. 'person seated at the table "
    "eating from a plate with a fork', 'person drank coffee but did not eat', "
    "'no person visible')."
)

_STOVE_LEFT_ON_PROMPT_FRAGMENT = (
    "stove_left_on_detected: boolean. TRUE only if a stove / cooktop is ON and "
    "UNATTENDED with nothing cooking — i.e. ALL of: (a) a burner is clearly on "
    "(a visible gas flame, or a glowing / red-hot electric element), (b) there "
    "is NO pot, pan, kettle, or food on the lit burner, and (c) no person is at "
    "or tending the stove anywhere in the clip. A lit but empty, unattended "
    "burner is the hazard — a stove left on by accident.\n"
    "\n"
    "Because each clip is only ~11 seconds, judge ONLY what is visible in THIS "
    "clip — do not try to infer how long it has been on. The combination "
    "'burner on + nothing on it + nobody there' is the in-clip signal that "
    "distinguishes a dangerous left-on stove from normal cooking.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - there is a pot, pan, kettle, or food on the lit burner (normal "
    "cooking);\n"
    "  - a person is at or near the stove tending it;\n"
    "  - the stove / burner is OFF (no flame, no glow);\n"
    "  - you cannot clearly tell whether a burner is on, or whether something is "
    "on it;\n"
    "  - no stove is visible.\n"
    "\n"
    "stove_left_on_confidence: number 0–1. 0 = no lit unattended empty burner. "
    "0.3–0.6 = a burner might be on but the cookware / a person / the on-off "
    "state is unclear. 0.7+ = clearly a lit burner with nothing on it and no one "
    "tending it.\n"
    "stove_left_on_reasoning: 1 short sentence (e.g. 'front gas burner lit with "
    "an empty grate and no one in the kitchen', 'burner on but a pot is on it "
    "and someone is cooking — not flagged', 'stove off', 'no stove visible')."
)

_EMERGENCY_PROMPT_FRAGMENT = (
    "emergency_detected: boolean. TRUE only on a SEVERE, unmistakable medical "
    "or safety emergency involving a person: a person collapsed and "
    "UNRESPONSIVE / not moving for the whole clip after going down, visible "
    "SERIOUS BLEEDING, seizure-like convulsing, or a person clearly "
    "unconscious. This is a catch-all for grave emergencies the more specific "
    "detectors might not name.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - ordinary activity, resting, or sleeping (a calmly sleeping person is "
    "NOT an emergency);\n"
    "  - a minor stumble, a person who falls but gets back up, or normal lying "
    "down;\n"
    "  - distress or agitation without a grave medical emergency;\n"
    "  - no person is visible, or the scene is too unclear to judge.\n"
    "\n"
    "Prioritize PRECISION: only fire when the emergency is obvious and severe. "
    "When unsure, set FALSE.\n"
    "\n"
    "emergency_confidence: number 0–1. 0 = no emergency. 0.3–0.6 = concerning "
    "but not clearly a grave emergency. 0.7+ = an obvious, severe emergency "
    "(unresponsive collapse, serious bleeding, seizure, unconsciousness).\n"
    "emergency_reasoning: 1 short sentence (e.g. 'person collapsed at 3s and "
    "lay motionless and unresponsive', 'visible bleeding from the head', "
    "'person sleeping calmly — not an emergency', 'no person visible')."
)

_POSSIBLE_INTRUDER_PROMPT_FRAGMENT = (
    "possible_intruder_detected: boolean. TRUE only if a person in the scene "
    "has their FACE DELIBERATELY CONCEALED — wearing a ski mask / balaclava, a "
    "covering pulled over the face, or otherwise obscuring their face in a way "
    "consistent with hiding their identity. The masked / concealed face must be "
    "visible.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - a plain medical / surgical face mask worn normally (that is ordinary, "
    "not an intruder signal);\n"
    "  - sunglasses, a hat, or a hood worn without the face being concealed;\n"
    "  - the face is simply turned away or partially out of frame (not "
    "deliberately covered);\n"
    "  - no person is visible.\n"
    "\n"
    "possible_intruder_confidence: number 0–1. 0 = no concealed face. 0.3–0.6 = "
    "the face is partly obscured but possibly ordinary. 0.7+ = a face clearly "
    "and deliberately concealed (ski mask / balaclava / pulled-up covering).\n"
    "possible_intruder_reasoning: 1 short sentence (e.g. 'person entered "
    "wearing a ski mask over the face at 2s', 'person wearing an ordinary "
    "surgical mask — not flagged', 'no person visible')."
)

_ENV_HAZARD_GLASS_PROMPT_FRAGMENT = (
    "env_hazard_glass_detected: boolean. TRUE only if there is BROKEN GLASS in "
    "the scene — a shattered glass, bottle, dish, mirror, or window, with "
    "visible shards / fragments scattered on the floor or a surface, or the act "
    "of glass shattering during the clip. The broken glass / shards must be "
    "visible.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - intact glassware, windows, or mirrors (whole, unbroken glass is "
    "normal);\n"
    "  - a spill of liquid with no broken glass;\n"
    "  - reflections or glare that are not actually broken glass;\n"
    "  - nothing resembling glass shards is visible.\n"
    "\n"
    "env_hazard_glass_confidence: number 0–1. 0 = no broken glass. 0.3–0.6 = "
    "something on the floor that might be shards but is unclear. 0.7+ = clearly "
    "broken glass / scattered shards.\n"
    "env_hazard_glass_reasoning: 1 short sentence (e.g. 'a glass shattered on "
    "the floor at 5s, shards scattered', 'intact glasses on the table — nothing "
    "broken', 'no broken glass visible')."
)

_ENV_HAZARD_WATER_PROMPT_FRAGMENT = (
    "env_hazard_water_detected: boolean. TRUE only if WATER (or another clear "
    "liquid) is SPILLED on the floor — a visible puddle or wet patch on the "
    "floor, or a spill happening during the clip — creating a slip hazard. The "
    "spilled liquid on the floor must be visible.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - liquid contained in a glass, cup, bottle, sink, or tub (not spilled on "
    "the floor);\n"
    "  - a normal wet / shiny floor finish, rug, or mat that is not an actual "
    "spill;\n"
    "  - broken glass with no liquid (that is the glass hazard, not this "
    "one);\n"
    "  - no spilled liquid on the floor is visible.\n"
    "\n"
    "env_hazard_water_confidence: number 0–1. 0 = no spill. 0.3–0.6 = a patch "
    "on the floor that might be a spill but is unclear. 0.7+ = a clear puddle / "
    "spilled liquid on the floor.\n"
    "env_hazard_water_reasoning: 1 short sentence (e.g. 'a glass of water "
    "spilled, leaving a puddle on the kitchen floor at 6s', 'shiny floor but no "
    "actual spill', 'no spill visible')."
)

_FIRE_SMOKE_PROMPT_FRAGMENT = (
    "fire_smoke_detected: boolean. The PURPOSE of this detector is to catch a "
    "DANGEROUS, uncontrolled fire — flames or smoke somewhere they should NOT "
    "be, signalling an emergency (e.g. furniture, curtains, bedding, clothing, "
    "an appliance, or a room catching fire; a fire that is spreading; or a space "
    "filling with smoke). Reason about whether what you see is a genuine hazard "
    "— not merely the presence of any flame.\n"
    "\n"
    "Set TRUE only for fire that is clearly UNCONTROLLED or out of place: flames "
    "on something that should not be burning, a fire that is spreading, or a "
    "haze / plume of smoke filling the room.\n"
    "\n"
    "Set FALSE for NORMAL, CONTROLLED flame and cooking — this is the key "
    "distinction. NONE of these count, even when a flame or some cooking smoke "
    "is visible:\n"
    "  - cooking over a flame: a wok, frying pan, or pot on a lit gas burner; a "
    "barbecue or grill; a stovetop or oven in normal use;\n"
    "  - a small controlled flame: a lit candle, a fireplace, a gas fire, a "
    "lighter, or a match;\n"
    "  - steam from cooking, a kettle, or a shower (steam is not smoke);\n"
    "  - light cooking smoke or haze coming from a pan that is clearly just "
    "cooking;\n"
    "  - dust, fog, glare, or camera artifacts that are not smoke;\n"
    "  - no flames or smoke visible.\n"
    "\n"
    "When a flame IS present, reason it through: is this controlled cooking / a "
    "candle / a fireplace (FALSE), or is something burning that shouldn't be, or "
    "a fire spreading (TRUE)? If it is ordinary cooking or a controlled flame, "
    "set FALSE.\n"
    "\n"
    "fire_smoke_confidence: number 0–1. 0 = no fire, or only controlled flame / "
    "cooking. 0.3–0.6 = smoke or flame that might be a hazard but could be "
    "cooking / controlled. 0.7+ = a clearly dangerous, uncontrolled fire, or a "
    "space filling with smoke.\n"
    "fire_smoke_reasoning: 1 short sentence (e.g. 'flames spreading up the "
    "curtains by the window', 'a wok flaming on the stove while cooking — normal, "
    "not flagged', 'lit candle on the table — controlled', 'no fire or smoke "
    "visible')."
)

# Internal one-time setup test. NOT a safety/routine detector — its only job is
# to let a family verify, during setup, that the whole alert path (detection →
# push → email) works, by holding three fingers up to a camera. Seeded hidden +
# fire-once (see CatalogEntry.hidden / .fire_once); disabled the instant it fires
# so it can never fire twice. Deliberately a HUMAN-HAND gesture, not a printed
# digit, so an ambient "3" on a clock/TV/book can't trip it.
_SETUP_TEST_PROMPT_FRAGMENT = (
    "setup_test_detected: boolean. TRUE only if a PERSON deliberately holds up "
    "ONE HAND toward the camera showing exactly THREE extended fingers — the "
    "'number 3' hand gesture — clearly and intentionally, as if presenting it to "
    "the camera. The hand should be reasonably close to / facing the lens with "
    "three fingers plainly extended and the others folded.\n"
    "\n"
    "Set FALSE for everything else, including: a hand at rest, waving, a thumbs-up, "
    "a peace sign (two fingers), an open hand (four or five fingers), a single "
    "finger, fingers that are ambiguous or partially hidden, or a printed/written "
    "number '3' on a clock, phone, TV, book, or sign (it must be a live human "
    "hand, not a drawn or displayed digit).\n"
    "\n"
    "setup_test_confidence: number 0–1. 0 = no three-finger gesture. 0.7+ = a "
    "person is clearly and deliberately holding up three fingers to the camera.\n"
    "setup_test_reasoning: 1 short sentence describing the hand you saw "
    "(e.g. 'a person held three fingers up to the lens', 'an open five-finger "
    "hand — not three', 'no raised-hand gesture')."
)


# ══════════════════════════════════════════════════════════════════
# Catalog entry
# ══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CatalogEntry:
    """One built-in detector definition. `key` is the runtime `identifier`."""

    key: str
    name: str
    icon_name: str
    description: str
    # Semantic label only — what KIND of thing this is, for wording,
    # colour and filtering. How urgently it is treated is `severity`
    # (catalog_v2.DIMS), which gates the scheduler's exemptions and the
    # compile bias. See schemas/detector.py:DetectorCategory.
    category: str       # safety | routine | behaviour | care
    cohort: str         # "child" | "adult" | "both"
    recommended: bool   # the library's "Recommended" UI group — only `fall`
    prompt_fragment: str
    # The STANDARD-sensitivity firing threshold. `event_promotion` adds the
    # per-detector sensitivity offset on top of this (low +, high −), so this is
    # the base, not the final bar. Uniform 0.70 today; per-detector for future
    # tuning.
    promotion_threshold: float = 0.70
    where_scope: str = "ALL ROOMS"
    when_scope: str = "ALWAYS"
    # Alerting spine (2026-07-26): which moment fires + internal temporal
    # nature. Most built-ins are leading; agitation is sustained; stove is a
    # duration ("on & unattended > duration_seconds").
    # record_only (2026-07-28): the detector's fields are recorded on every
    # tick for summarizers (sleep/activity) but NEVER promote an event or
    # notify anyone — skipped by reactive promotion AND the scheduled runner.
    alert_trigger: str = "leading"   # leading | trailing | duration | schedule | record_only
    is_sustained: bool = False
    duration_seconds: int | None = None
    # Most detectors use `<key>_detected`. Medication is the exception
    # (`medication_taken`), matching its prompt + the legacy schema.
    detected_field: str | None = None
    # Two INDEPENDENT flags:
    #   `recommended` — the special "Recommended" group in the library UI. Only
    #                   `fall` is recommended.
    #   `seeded`      — auto-created on patient create AND pre-checked at
    #                   onboarding: `fall` + the cohort presets (adult:
    #                   agitation/wandering/medication; child: climbing/distress).
    # A seeded-but-not-recommended detector is pre-selected when making a
    # patient yet still listed under "More detectors".
    seeded: bool = False
    # Internal detectors the caregiver should never see or manage. Excluded
    # from `for_cohort` (so it's not in the library picker) and filtered out of
    # `detector_repo.list_for_patient` (so it's not in the detectors list). It
    # still SEEDS (via `seeded_for_cohort`, which reads CATALOG directly) and
    # still RUNS (the worker's registry query is independent of the UI list).
    hidden: bool = False
    # Fires exactly once per patient, then disables itself. `event_promotion`
    # flips the (single, per-patient) detector row's `is_on=false` the instant
    # it promotes an event, so it can't fire again on this or any other camera.
    fire_once: bool = False
    # Extra per-tick response fields BEYOND the standard detected/confidence/
    # reasoning triple (2026-07-28, sleep/activity): merged into schema_json so
    # they reach the LLM RESPONSE SCHEMA — a field named only in the prompt is
    # never returned by structured output. {"field": {"type": "string"}, ...}
    extra_schema: "dict[str, dict[str, str]] | None" = None

    @property
    def detected_key(self) -> str:
        return self.detected_field or f"{self.key}_detected"

    @property
    def confidence_key(self) -> str:
        return f"{self.key}_confidence"

    @property
    def reasoning_key(self) -> str:
        return f"{self.key}_reasoning"

    @property
    def schema_json(self) -> dict[str, dict[str, str]]:
        """The `<detected> / <confidence> / <reasoning>` JSON-schema map the
        pipeline merges into the LLM response schema."""
        return {
            self.detected_key:   {"type": "boolean"},
            self.confidence_key: {"type": "number"},
            self.reasoning_key:  {"type": "string"},
            **(self.extra_schema or {}),
        }


# Order matters: entries appear in this order within each cohort (the iOS
# library list + the seed order). The SEEDED presets come first so onboarding
# seeds them in today's order — adult (fall, agitation, wandering, medication),
# child (fall, climbing, distress) — byte-identical to the previous preset
# tuples. Of those, only `fall` is `recommended` (the special "Recommended"
# group); the rest are seeded-but-not-recommended, so they're pre-checked at
# onboarding yet listed under "More detectors". `fall` is `cohort="both"`.
CATALOG: tuple[CatalogEntry, ...] = (
    # ── recommended presets (seeded by default) ──
    CatalogEntry(
        key="fall", name="Falls", icon_name="figure.fall",
        description="Sudden fall events across all rooms.",
        category="safety", cohort="both",
        recommended=True, seeded=True, prompt_fragment=_FALL_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="agitation", name="Agitation", icon_name="figure.walk.motion",
        description="Pacing or restless body language sustained over time.",
        category="behaviour", is_sustained=True, cohort="adult",
        recommended=False, seeded=True, prompt_fragment=_AGITATION_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    # Convulsion REMOVED from the catalog (Aryaman 2026-07-28): the
    # benchmark showed VLM video sampling sits below tremor frequency
    # (3-6 Hz) — a temporal-sampling ceiling, not resolution — so the
    # detector's FP/FN rate is unacceptable as a built-in. The rapid
    # LANE stays for custom detectors, which now carry a
    # "rapid_frequency" degradation warning at compile (compiler_v2).
    # Successor: computed-motion FFT multi-criterion detector,
    # post-pilot (docs/plans/2026-07-28-pilot-master-worklist.md).
    CatalogEntry(
        key="wandering", name="Wandering", icon_name="door.left.hand.open",
        description="Main door opening, front entry only.",
        category="safety", cohort="adult",
        recommended=False, seeded=True, prompt_fragment=_WANDERING_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="sleep", name="Sleep", icon_name="bed.double",
        description="Tracks bedtime, sleep duration, disturbances, and "
                    "out-of-bed events overnight.",
        category="routine", cohort="both",
        recommended=False, seeded=False, prompt_fragment=_SLEEP_PROMPT_FRAGMENT,
        promotion_threshold=0.60,
        alert_trigger="record_only",
        extra_schema={"sleep_activity": {"type": "string"}},
    ),
    CatalogEntry(
        key="activity", name="Activity", icon_name="figure.walk",
        description="Tracks how much of the day is spent lying, sitting, "
                    "standing, or physically active.",
        category="routine", cohort="both",
        recommended=False, seeded=False,
        prompt_fragment=_ACTIVITY_PROMPT_FRAGMENT,
        promotion_threshold=0.60,
        alert_trigger="record_only",
        extra_schema={"activity_state": {"type": "string"}},
    ),
    CatalogEntry(
        key="medication", name="Medication adherence", icon_name="pills",
        description="Confirms the morning and evening pill routine at the kitchen counter.",
        category="routine", cohort="adult",
        recommended=False, seeded=True, prompt_fragment=_MEDICATION_PROMPT_FRAGMENT,
        # ALL DAY (Aryaman 2026-07-28): presets never ship hardcoded
        # schedules — a time window is a CAREGIVER choice made through
        # the editor's picker (one representation, consistent on every
        # surface). The old "8 AM, 8 PM ±1H" was an authored guess in a
        # format no editor could edit.
        promotion_threshold=0.70,
        detected_field="medication_taken",
    ),
    # ── child library (not recommended) ──
    # ── adult library (not recommended) ──
    CatalogEntry(
        key="calling_for_help", name="Calling for help",
        icon_name="megaphone",
        description="The person calling out for help in distress.",
        category="safety", cohort="adult",
        recommended=False, prompt_fragment=_CALLING_FOR_HELP_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="choking", name="Choking", icon_name="lungs",
        description="Signs of choking: hands to the throat, unable to breathe.",
        category="safety", cohort="adult",
        recommended=False, prompt_fragment=_CHOKING_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="meal_eaten", name="Meal eaten", icon_name="fork.knife",
        description="Confirms the person ate a meal.",
        category="routine", cohort="adult",
        recommended=False, prompt_fragment=_MEAL_EATEN_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="stove_left_on", name="Stove left on", icon_name="flame.fill",
        description="A lit, unattended stove with nothing cooking on it.",
        category="safety", cohort="adult",
        recommended=False, prompt_fragment=_STOVE_LEFT_ON_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
        # "Left on" is a duration threshold, not an instant — fire once a lit
        # unattended stove has persisted past this bar (10 min).
        alert_trigger="duration", is_sustained=True, duration_seconds=600,
    ),
    # ── both cohorts (not recommended) ──
    CatalogEntry(
        key="emergency", name="General Emergency", icon_name="cross.case",
        description="Severe emergencies: unconsciousness, serious bleeding, seizure.",
        category="safety", cohort="both",
        recommended=False, prompt_fragment=_EMERGENCY_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="possible_intruder", name="Possible Intruder", icon_name="person.fill.questionmark",
        description="A person with a deliberately concealed (masked) face.",
        category="safety", cohort="both",
        recommended=False, prompt_fragment=_POSSIBLE_INTRUDER_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="env_hazard_glass", name="Broken glass",
        icon_name="wineglass",
        description="Broken glass or scattered shards on the floor.",
        category="safety", cohort="both",
        recommended=False, prompt_fragment=_ENV_HAZARD_GLASS_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="env_hazard_water", name="Spilled water",
        icon_name="drop",
        description="Water or liquid spilled on the floor, a slip hazard.",
        category="safety", cohort="both",
        recommended=False, prompt_fragment=_ENV_HAZARD_WATER_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="fire_smoke", name="Fire Hazard", icon_name="flame",
        description="Visible flames or smoke.",
        category="safety", cohort="both",
        recommended=False, prompt_fragment=_FIRE_SMOKE_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    # ── internal: one-time setup self-test (hidden, fire-once) ──
    # Seeded for every new patient and assigned to every camera (source=preset),
    # but never shown in the app. Fires once when someone holds three fingers up
    # to a camera, proving the alert path end-to-end, then disables itself.
    CatalogEntry(
        key="setup_test", name="Setup test", icon_name="hand.raised.fingers.spread",
        description="Internal one-time alert self-test.",
        category="routine", cohort="both",
        recommended=False, seeded=True, hidden=True, fire_once=True,
        prompt_fragment=_SETUP_TEST_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
)


# Fail loud on a malformed catalog at import time rather than producing a
# subtly-broken detector at runtime. Duplicate keys would silently shadow each
# other; duplicate schema fields would blow up `prompt.build_prompt_and_schema`
# mid-tick.
def _validate_catalog() -> None:
    seen_keys: set[str] = set()
    seen_fields: set[str] = set()
    for e in CATALOG:
        if e.key in seen_keys:
            raise ValueError(f"[catalog] duplicate detector key: {e.key!r}")
        seen_keys.add(e.key)
        if e.cohort not in ("child", "adult", "both"):
            raise ValueError(f"[catalog] {e.key}: bad cohort {e.cohort!r}")
        if e.category not in ("safety", "routine", "behaviour", "care"):
            raise ValueError(f"[catalog] {e.key}: bad category {e.category!r}")
        if e.alert_trigger not in ("leading", "trailing", "duration", "absence", "record_only"):
            raise ValueError(f"[catalog] {e.key}: bad alert_trigger {e.alert_trigger!r}")
        if e.alert_trigger == "duration" and not e.duration_seconds:
            raise ValueError(f"[catalog] {e.key}: duration trigger needs duration_seconds")
        for field in e.schema_json:
            if field in seen_fields:
                raise ValueError(
                    f"[catalog] duplicate schema field {field!r} "
                    f"(introduced by {e.key!r})"
                )
            seen_fields.add(field)


_validate_catalog()


# ══════════════════════════════════════════════════════════════════
# Helpers (read-only)
# ══════════════════════════════════════════════════════════════════


def is_minor(date_of_birth: date | None) -> bool:
    """True if the patient is under 18 as of today (UTC). An absent birthday
    is treated as NOT a minor — preserving the prior adult-default behaviour."""
    if date_of_birth is None:
        return False
    today = datetime.now(UTC).date()
    age = today.year - date_of_birth.year - (
        (today.month, today.day) < (date_of_birth.month, date_of_birth.day)
    )
    return age < 18


def for_cohort(minor: bool) -> list[CatalogEntry]:
    """Catalog entries shown to this cohort in the library UI: the cohort-
    specific ones (`child` or `adult`) plus the `both` entries, in CATALOG order
    (recommended first). `hidden` entries are excluded — they seed + run but are
    never offered in the picker."""
    want = "child" if minor else "adult"
    return [e for e in CATALOG if e.cohort in (want, "both") and not e.hidden]


def seeded_for_cohort(minor: bool) -> list[CatalogEntry]:
    """The subset auto-seeded on patient create AND pre-checked at onboarding —
    fall + the cohort presets (+ any hidden internal detectors like the setup
    self-test). Reads CATALOG directly rather than via `for_cohort` so a `hidden`
    detector still seeds even though it's excluded from the library UI."""
    want = "child" if minor else "adult"
    return [e for e in CATALOG if e.cohort in (want, "both") and e.seeded]


def hidden_keys() -> list[str]:
    """Identifiers of internal detectors that must never appear in any caregiver-
    facing list. `detector_repo.list_for_patient` filters these out."""
    return [e.key for e in CATALOG if e.hidden]


def fire_once_keys() -> frozenset[str]:
    """Identifiers that fire exactly once then disable themselves.
    `event_promotion` flips `is_on=false` on the row right after promoting."""
    return frozenset(e.key for e in CATALOG if e.fire_once)


def by_key(key: str) -> CatalogEntry | None:
    for e in CATALOG:
        if e.key == key:
            return e
    return None


def promotion_rules() -> dict[str, tuple[str, str, float]]:
    """`{identifier: (detected_field, confidence_field, min_confidence)}` for
    every reactive catalog detector. `event_promotion` builds its gate from
    this so every built-in detector is confidence-gated (custom detectors,
    which aren't in the catalog, keep their own bare-`_detected` fallback).
    Scheduled detectors are excluded — they fire via the scheduler, not the
    reactive path."""
    return {
        e.key: (e.detected_key, e.confidence_key, e.promotion_threshold)
        for e in CATALOG
        if e.alert_trigger != "absence"
    }


def all_keys() -> Sequence[str]:
    return [e.key for e in CATALOG]


# Within-tick precedence. When several detectors cross threshold on the SAME
# clip, event_promotion fires the single HIGHEST-severity one and suppresses the
# rest — so e.g. Choking wins over Agitation (if someone is choking, the caregiver
# doesn't also need an agitation alert). Higher number = higher severity. Specific
# life-threatening detectors outrank the generic catch-alls (emergency/distress),
# which outrank routine. Custom/unknown detectors get _DEFAULT_SEVERITY.
_DEFAULT_SEVERITY = 50
_SEVERITY: dict[str, int] = {
    "setup_test": 200,               # self-test — always wins during setup
    # ── life-threatening, specific ──────────────────────────────────
    "choking": 100,
    "fire_smoke": 98,
    "child_cord_entanglement": 97,
    # "seizure" removed 2026-07-28 with its catalog entry.
    "fall": 95,
    "child_head_impact": 93,
    "calling_for_help": 88,
    # ── generic catch-alls (below the specific critical ones) ───────
    "distress": 82,
    "emergency": 80,
    "possible_intruder": 78,
    # ── urgent safety ───────────────────────────────────────────────
    "stove_left_on": 74,
    "child_stairs": 73,
    "child_hazard_reach": 70,
    "climbing": 66,
    "child_crib_exit": 65,
    "env_hazard_glass": 62,
    "agitation": 60,
    "env_hazard_water": 56,
    "wandering": 55,
    "child_wandering": 55,
    # ── routine ─────────────────────────────────────────────────────
    "child_device_usage": 25,
    "meal_eaten": 22,
    "medication": 20,
    "sleep": 15,     # overnight labeling — below every safety + medication
    "activity": 14,  # daytime state labeling — record_only, lowest priority
}


def severity_for(identifier: str | None) -> int:
    """Within-tick precedence weight for a detector (higher wins). Custom /
    unknown detectors fall back to _DEFAULT_SEVERITY."""
    return _SEVERITY.get(identifier or "", _DEFAULT_SEVERITY)
