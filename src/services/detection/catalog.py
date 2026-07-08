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
`camera_detector_repo.assign_preset_detectors_of_patient_to_camera`, "Possible …"
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
from datetime import date, datetime, timezone

# ══════════════════════════════════════════════════════════════════
# Prompt fragments
# ══════════════════════════════════════════════════════════════════
# The first six are ported VERBATIM from the previous
# `onboarding.py` so seeding output is byte-identical. Do not reword
# them here without intending to change every patient's detector.

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


def build_wandering_fragment(door: dict | None) -> str:
    """Return the wandering `prompt_fragment` specialized to the caregiver's
    chosen exit door. `door` is the `door_config` payload (or None).

    With a selected door, the per-tick VLM is told WHICH doorway is the exterior
    exit so it only fires when the person leaves THROUGH that specific door —
    leaving via an interior doorway (bathroom, bedroom) or a frame edge no longer
    counts. With no door (None), falls back to the generic any-doorway fragment.
    """
    if not door or not door.get("box") or len(door.get("box") or []) != 4:
        return _WANDERING_PROMPT_FRAGMENT
    region = _door_region_phrase(door["box"])
    label = str(door.get("label") or "").strip()
    where = f"the {region} of the camera view"
    if label:
        where += f" (the {label})"
    targeting = (
        "THIS ROOM HAS ONE EXTERIOR EXIT DOOR — the ONLY doorway that counts as "
        f"wandering. It is located in {where}. A person counts as wandering ONLY "
        "if they walk COMPLETELY out of view THROUGH THAT specific exit doorway. "
        "Leaving through any OTHER doorway (to a bathroom, bedroom, or closet), or "
        "walking off a different edge of the frame, is NOT wandering — set FALSE "
        "for those. Everything below still applies to that one exit door.\n\n"
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
    "medication_taken: boolean. Set TRUE only with CLEAR, OBVIOUS proof "
    "that the person took medication, by one of these two patterns. Do "
    "not speculate about other things they might ingest or other "
    "containers they might handle.\n"
    "\n"
    "VISUAL GATE (oral). Set FALSE unless at least ONE of these is "
    "clearly present in view: (a) a typical pill container — a pill "
    "bottle, blister / foil pack, or weekly pill organizer; OR (b) an "
    "unmistakable pill, tablet, or capsule. A generic cup, mug, glass, "
    "food package, or other non-medicine container does NOT count. If "
    "neither a typical pill container nor a clearly visible pill is in "
    "view, set FALSE — never infer medication from a drink or a "
    "hand-to-mouth motion alone.\n"
    "\n"
    "ORAL — all three steps visible, with the drink either just before "
    "or just after the pill:\n"
    "  - they handle an obvious medicine container — a pill bottle, a "
    "blister / foil pack, or a weekly pill organizer;\n"
    "  - they take medicine out and put it in their mouth;\n"
    "  - they drink a liquid right before or right after (pill then "
    "drink, or drink then pill).\n"
    "\n"
    "INJECTION — they clearly give themselves an injection with an "
    "obvious device (insulin pen or syringe) against the body.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - neither a typical pill container nor a clearly visible pill is "
    "in view (a drink or a hand-to-mouth motion alone is not enough);\n"
    "  - you cannot clearly see the pill placed in the mouth (or, for "
    "injection, the device against the body);\n"
    "  - only part of the sequence is visible — e.g. a drink alone, or "
    "holding a container without taking anything out;\n"
    "  - the person eats food or drinks a beverage with no medicine "
    "container and no pill visible;\n"
    "  - they handle some other, non-medicine container (food, a mug, a "
    "drink bottle);\n"
    "  - no person is visible, or the action is too far or unclear to "
    "confirm.\n"
    "\n"
    "medication_confidence: number 0–1. 0 = no medication behavior / "
    "unclear. 0.3–0.6 = partial or ambiguous (e.g. handles a bottle but "
    "the pill-to-mouth is not clearly seen). 0.7+ = the full pattern "
    "above is clearly visible.\n"
    "medication_reasoning: 1 short sentence stating the proof you saw "
    "(e.g. 'opened a weekly pill organizer, placed a pill in mouth, "
    "drank water', 'injected with an insulin pen in the abdomen', "
    "'drank from a glass but no pills or medicine container visible', "
    "'no person visible')."
)

_CLIMBING_PROMPT_FRAGMENT = (
    "climbing_detected: boolean. TRUE if a child is climbing onto something "
    "they shouldn't — pulling up onto furniture (sofa back, table, kitchen "
    "counter, shelf, windowsill, railing), scaling a crib rail or safety gate, "
    "or already standing / perched on an elevated surface in a way that risks "
    "a fall. The climbing motion OR the child clearly perched on the elevated "
    "surface must be visible in the clip.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child is sitting, standing, or playing normally on the floor;\n"
    "  - an adult is holding or lifting the child;\n"
    "  - the child is seated in a high chair, crib, or stroller in normal use;\n"
    "  - no child is visible;\n"
    "  - you only hear sounds without seeing the climbing.\n"
    "\n"
    "climbing_confidence: number 0–1. 0 = no climbing. 0.3–0.6 = ambiguous "
    "(child near furniture but not clearly climbing). 0.7+ = clearly climbing "
    "onto or perched on an elevated surface.\n"
    "climbing_reasoning: 1 short sentence describing what you saw (e.g. "
    "'toddler pulled up onto the coffee table at 5s', 'child sitting on the "
    "floor playing — no climbing', 'no child visible')."
)

_DISTRESS_PROMPT_FRAGMENT = (
    "distress_detected: boolean. TRUE if a child shows sustained distress "
    "across the clip — crying, wailing, or visibly upset (scrunched / red "
    "face, flailing, reaching out while crying) for multiple seconds. Here, "
    "'child' means anyone who appears to be roughly 16 years old or younger — "
    "if a person could plausibly be 16 or under, treat them as a child, and "
    "only rule them out when they clearly look like an adult. Audio "
    "(crying / wailing) is SUPPORTING evidence that can corroborate a visual "
    "judgment, but audio alone is NEVER sufficient — if the child is visually "
    "calm, off-camera, or not visible, set FALSE.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child is calm, playing, laughing, or sleeping;\n"
    "  - the upset is a single brief moment rather than sustained;\n"
    "  - no child is visible.\n"
    "\n"
    "distress_confidence: number 0–1. 0 = calm / no child. 0.3–0.6 = some "
    "fussing but ambiguous. 0.7+ = clear sustained distress; raise within the "
    "band when crying audio corroborates the visual signal.\n"
    "distress_reasoning: 1 short sentence (e.g. 'child crying and reaching out "
    "for ~8s, audio wailing', 'child playing happily — calm', 'audible fussing "
    "but no child visible — not distress', 'no child visible')."
)

# ── New library prompts (14) — starting points, field-tune like any prompt ──

_CHILD_CRIB_EXIT_PROMPT_FRAGMENT = (
    "child_crib_exit_detected: boolean. TRUE only if a child is climbing OUT "
    "OF A CRIB specifically — pulling up and over the crib rail, throwing a leg "
    "over, or already straddling / perched on top of the crib rail in a way "
    "that risks falling out. The crib (a barred or solid-sided baby bed with "
    "high rails) must be identifiable, and the over-the-rail motion or the "
    "child perched on the rail must be visible in the clip.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the bed is a regular bed, toddler bed, sofa, or any non-crib surface — "
    "this detector is ONLY for cribs;\n"
    "  - the child is lying, sitting, or standing inside the crib normally;\n"
    "  - an adult is lifting the child in or out;\n"
    "  - no crib or no child is visible.\n"
    "\n"
    "child_crib_exit_confidence: number 0–1. 0 = no crib exit. 0.3–0.6 = child "
    "active at the crib rail but not clearly going over. 0.7+ = clearly "
    "climbing over / perched on the crib rail.\n"
    "child_crib_exit_reasoning: 1 short sentence (e.g. 'toddler threw a leg "
    "over the crib rail at 6s', 'child standing in crib but not climbing out', "
    "'no crib visible')."
)

_CHILD_CORD_ENTANGLEMENT_PROMPT_FRAGMENT = (
    "child_cord_entanglement_detected: boolean. TRUE only if a cord-like "
    "object — a blind / curtain pull cord, a window-shade string, an electrical "
    "or charger cord, a drawstring, or similar — is wrapped around, looped "
    "over, or pulled taut against a child's NECK. The cord and its contact "
    "with the neck must be visible.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child merely holds, plays with, or stands near a cord with no neck "
    "contact;\n"
    "  - the cord is around a hand, wrist, or torso but not the neck;\n"
    "  - no child is visible, or the scene is too far / unclear to see a cord "
    "at the neck.\n"
    "\n"
    "This is a high-stakes but visually subtle event — when a cord is plausibly "
    "at the neck but you cannot be certain, use a middle confidence rather than "
    "forcing TRUE or FALSE.\n"
    "\n"
    "child_cord_entanglement_confidence: number 0–1. 0 = no cord at neck. "
    "0.3–0.6 = a cord near the neck but contact unclear. 0.7+ = a cord clearly "
    "around / taut against the neck.\n"
    "child_cord_entanglement_reasoning: 1 short sentence (e.g. 'blind cord "
    "looped around the toddler's neck at 4s', 'child holding a charger cord, "
    "not near the neck', 'no child visible')."
)

_CHILD_STAIRS_PROMPT_FRAGMENT = (
    "child_stairs_detected: boolean. TRUE only if a young child is at a "
    "staircase unsupervised — climbing up or down stairs, or standing / "
    "crawling at the very top or edge of a staircase where a fall down the "
    "stairs is the risk. A staircase (a run of steps) and the child on or at "
    "the edge of it must be visible.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - an adult is with the child on or at the stairs;\n"
    "  - the child is near a single step, threshold, or low ledge — not a "
    "staircase;\n"
    "  - the child is elsewhere in the room, not at the stairs;\n"
    "  - no staircase or no child is visible.\n"
    "\n"
    "child_stairs_confidence: number 0–1. 0 = not at stairs. 0.3–0.6 = near a "
    "staircase but not clearly on / at the edge. 0.7+ = clearly on the stairs "
    "or at the top edge unsupervised.\n"
    "child_stairs_reasoning: 1 short sentence (e.g. 'toddler crawling up the "
    "stairs alone at 7s', 'child at the top of the stairs with no adult', "
    "'adult carrying child on stairs — not flagged', 'no stairs visible')."
)

_CHILD_HAZARD_REACH_PROMPT_FRAGMENT = (
    "child_hazard_reach_detected: boolean. This detector covers EXACTLY THREE "
    "hazards and nothing else: (1) a KNIFE or other obvious sharp blade, (2) an "
    "electrical OUTLET / wall socket, (3) a STOVE / cooktop / oven. Set TRUE "
    "only if a young child is reaching for, grabbing, or touching one of these "
    "three specific hazards. Ignore every other object — this detector does NOT "
    "cover any other hazard.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child is reaching for anything that is NOT a knife, an electrical "
    "outlet, or a stove;\n"
    "  - an adult is supervising and within arm's reach;\n"
    "  - the child is near one of the three hazards but not reaching for or "
    "touching it;\n"
    "  - no child is visible, or you cannot clearly identify the object as a "
    "knife, outlet, or stove.\n"
    "\n"
    "child_hazard_reach_confidence: number 0–1. 0 = no reach for any of the "
    "three hazards. 0.3–0.6 = reaching toward that area but the hazard or "
    "intent is ambiguous. 0.7+ = clearly reaching for / touching a knife, an "
    "outlet, or a stove.\n"
    "child_hazard_reach_reasoning: 1 short sentence naming WHICH of the three "
    "hazards (e.g. 'toddler reached for a knife on the counter at 5s', 'child "
    "poking fingers at a wall outlet', 'child near the table but not reaching "
    "for a knife / outlet / stove', 'no child visible')."
)

_CHILD_HEAD_IMPACT_PROMPT_FRAGMENT = (
    "child_head_impact_detected: boolean. TRUE only if a child's HEAD "
    "specifically takes an impact — the head strikes the floor, furniture, a "
    "wall, or another hard surface, or an object strikes the child's head. The "
    "impact must be to the head, and the strike must be visible in the clip.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the impact is to the body, limbs, or bottom but NOT the head;\n"
    "  - the child bumps something gently with no real impact;\n"
    "  - the child falls but the head clearly does not hit anything;\n"
    "  - no child is visible, or the contact is off-camera / too unclear to "
    "confirm a head impact.\n"
    "\n"
    "child_head_impact_confidence: number 0–1. 0 = no head impact. 0.3–0.6 = a "
    "knock that might have involved the head but it is unclear. 0.7+ = the head "
    "clearly struck a surface or was struck.\n"
    "child_head_impact_reasoning: 1 short sentence (e.g. 'child's head hit the "
    "table edge at 3s', 'toddler fell but landed on hands, head did not hit', "
    "'no child visible')."
)

_CHILD_WANDERING_PROMPT_FRAGMENT = (
    "child_wandering_detected: boolean. TRUE only if, within this clip, you SEE "
    "a child walk through a doorway and LEAVE the room UNSUPERVISED — the child "
    "was inside the room earlier in the clip, and by the end they have gone out "
    "through a doorway and stayed out, with no adult accompanying them. Only "
    "OUTBOUND exits count. Here, 'child' means anyone who appears to be roughly "
    "16 years old or younger — if a person could plausibly be 16 or under, "
    "treat them as a child, and only rule them out when they clearly look like "
    "an adult.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child only approaches, touches, opens, or looks toward the door but "
    "does not actually walk through;\n"
    "  - the child was outside and walked IN (entering doesn't count, only "
    "leaving does);\n"
    "  - an adult goes with the child or is leading them out;\n"
    "  - the child leaves and returns within the same clip;\n"
    "  - the child walks out of frame somewhere that isn't a clearly visible "
    "doorway;\n"
    "  - no child is visible.\n"
    "\n"
    "child_wandering_confidence: number 0–1. 0 = no exit seen. 0.3–0.6 = "
    "ambiguous (child near the door, exit unclear). 0.7+ = clearly walked out "
    "through a doorway, unaccompanied, and did not come back.\n"
    "child_wandering_reasoning: 1 short sentence (e.g. 'toddler walked out the "
    "doorway alone at 9s and did not return', 'child approached the door but did "
    "not go through', 'child left with an adult — not flagged', 'no child "
    "visible')."
)

_CHILD_DEVICE_USAGE_PROMPT_FRAGMENT = (
    "child_device_usage_detected: boolean. TRUE only if a child is using a phone "
    "or tablet for a SUSTAINED stretch of this clip — holding and looking at / "
    "interacting with a handheld smartphone or tablet for MOST of the clip "
    "(roughly 8 of its ~11 seconds), not just a brief glance. Here, 'child' "
    "means anyone who appears to be roughly 16 years old or younger — if a "
    "person could plausibly be 16 or under, treat them as a child, and only "
    "rule them out when they clearly look like an adult.\n"
    "\n"
    "Because each clip is only ~11 seconds, 'prolonged' here means sustained "
    "across most of THIS clip: the child is on the device for the majority of "
    "the seconds you can see, rather than momentarily picking it up.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - the child only briefly glances at, picks up, or hands over a device;\n"
    "  - the device is a TV, laptop, game controller, or a toy phone — this is "
    "specifically a handheld phone or tablet;\n"
    "  - an adult is holding the device, or it is sitting unused;\n"
    "  - the child is mostly doing something else during the clip;\n"
    "  - no child or no device is visible.\n"
    "\n"
    "child_device_usage_confidence: number 0–1. 0 = no device use. 0.3–0.6 = a "
    "child with a phone/tablet but only briefly or unclear. 0.7+ = a child "
    "clearly engaged with a phone/tablet for most of the clip.\n"
    "child_device_usage_reasoning: 1 short sentence (e.g. 'child watched a tablet "
    "held in both hands for ~9s', 'child glanced at a phone then put it down', "
    "'adult was holding the phone, not the child', 'no device visible')."
)

_CALLING_FOR_HELP_PROMPT_FRAGMENT = (
    "calling_for_help_detected: boolean. TRUE only if the person on camera is "
    "clearly calling for help — shouting 'help', 'somebody help', calling out a "
    "caregiver's name in distress, or crying out that they have fallen, are "
    "hurt, or cannot get up. BOTH a distressed delivery AND help-seeking "
    "content should be evident.\n"
    "\n"
    "Set FALSE if any of:\n"
    "  - ordinary conversation, a normal phone call, or calling someone for a "
    "routine reason (no distress);\n"
    "  - the voice likely comes from a TV / radio / phone / device, or cannot "
    "be attributed to the person on camera;\n"
    "  - sounds of distress with no discernible call for help;\n"
    "  - no person is visible, or the audio is too unclear to judge.\n"
    "\n"
    "calling_for_help_confidence: number 0–1. 0 = no call for help. 0.3–0.6 = a "
    "raised or distressed voice but the help-seeking content is unclear. 0.7+ = "
    "an unmistakable call for help.\n"
    "calling_for_help_reasoning: 1 short sentence (e.g. 'person on the floor "
    "shouted \"help, I've fallen\" at 5s', 'loud but ordinary phone "
    "conversation', 'no person visible')."
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
    category: str       # "safety" | "routine"
    shape: str          # "point_in_time" | "state" | "scheduled"
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
        category="safety", shape="point_in_time", cohort="both",
        recommended=True, seeded=True, prompt_fragment=_FALL_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="agitation", name="Agitation", icon_name="figure.walk.motion",
        description="Pacing or restless body language sustained over time.",
        category="safety", shape="state", cohort="adult",
        recommended=False, seeded=True, prompt_fragment=_AGITATION_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="wandering", name="Wandering", icon_name="door.left.hand.open",
        description="Main door opening — front entry only.",
        category="safety", shape="point_in_time", cohort="adult",
        recommended=False, seeded=True, prompt_fragment=_WANDERING_PROMPT_FRAGMENT,
        promotion_threshold=0.70, where_scope="HALLWAY",
    ),
    CatalogEntry(
        key="medication", name="Medication adherence", icon_name="pills",
        description="Confirms the morning and evening pill routine at the kitchen counter.",
        category="routine", shape="point_in_time", cohort="adult",
        recommended=False, seeded=True, prompt_fragment=_MEDICATION_PROMPT_FRAGMENT,
        promotion_threshold=0.70, where_scope="KITCHEN", when_scope="8 AM, 8 PM ±1H",
        detected_field="medication_taken",
    ),
    CatalogEntry(
        key="climbing", name="Climbing", icon_name="figure.climbing",
        description="Climbing onto furniture or elevated surfaces.",
        category="safety", shape="state", cohort="child",
        recommended=False, seeded=True, prompt_fragment=_CLIMBING_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="distress", name="Distress", icon_name="exclamationmark.bubble",
        description="Sustained crying or visible distress.",
        category="safety", shape="state", cohort="child",
        recommended=False, seeded=True, prompt_fragment=_DISTRESS_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    # ── child library (not recommended) ──
    CatalogEntry(
        key="child_crib_exit", name="Crib exit", icon_name="bed.double",
        description="A child climbing out of a crib over the rail.",
        category="safety", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_CRIB_EXIT_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="child_cord_entanglement", name="Cord / blind risk", icon_name="scribble",
        description="A cord or blind string around a child's neck.",
        category="safety", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_CORD_ENTANGLEMENT_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="child_stairs", name="Near stairs", icon_name="figure.stairs",
        description="A child at a staircase unsupervised.",
        category="safety", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_STAIRS_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="child_hazard_reach", name="Reaching for a hazard",
        icon_name="exclamationmark.triangle",
        description="A child reaching for a knife, an outlet, or a stove (only these three).",
        category="safety", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_HAZARD_REACH_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="child_head_impact", name="Head impact", icon_name="brain.head.profile",
        description="An impact specifically to a child's head.",
        category="safety", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_HEAD_IMPACT_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="child_wandering", name="Wandering", icon_name="door.left.hand.open",
        description="A child leaving the room through a doorway unsupervised.",
        category="safety", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_WANDERING_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="child_device_usage", name="Electronic device usage", icon_name="ipad",
        description="A child using a phone or tablet for a sustained period.",
        category="routine", shape="point_in_time", cohort="child",
        recommended=False, prompt_fragment=_CHILD_DEVICE_USAGE_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    # ── adult library (not recommended) ──
    CatalogEntry(
        key="calling_for_help", name="Calling for help",
        icon_name="megaphone",
        description="The person calling out for help in distress.",
        category="safety", shape="point_in_time", cohort="adult",
        recommended=False, prompt_fragment=_CALLING_FOR_HELP_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="choking", name="Choking", icon_name="lungs",
        description="Signs of choking — hands to the throat, unable to breathe.",
        category="safety", shape="point_in_time", cohort="adult",
        recommended=False, prompt_fragment=_CHOKING_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="meal_eaten", name="Meal eaten", icon_name="fork.knife",
        description="Confirms the person ate a meal.",
        category="routine", shape="point_in_time", cohort="adult",
        recommended=False, prompt_fragment=_MEAL_EATEN_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="stove_left_on", name="Stove left on", icon_name="flame.fill",
        description="A lit, unattended stove with nothing cooking on it.",
        category="safety", shape="point_in_time", cohort="adult",
        recommended=False, prompt_fragment=_STOVE_LEFT_ON_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    # ── both cohorts (not recommended) ──
    CatalogEntry(
        key="emergency", name="General Emergency", icon_name="cross.case",
        description="Severe emergencies — unconsciousness, serious bleeding, seizure.",
        category="safety", shape="point_in_time", cohort="both",
        recommended=False, prompt_fragment=_EMERGENCY_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="possible_intruder", name="Possible Intruder", icon_name="person.fill.questionmark",
        description="A person with a deliberately concealed (masked) face.",
        category="safety", shape="point_in_time", cohort="both",
        recommended=False, prompt_fragment=_POSSIBLE_INTRUDER_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="env_hazard_glass", name="Environmental hazard — Glass",
        icon_name="wineglass",
        description="Broken glass / scattered shards.",
        category="safety", shape="point_in_time", cohort="both",
        recommended=False, prompt_fragment=_ENV_HAZARD_GLASS_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="env_hazard_water", name="Environmental hazard — Spilled water",
        icon_name="drop",
        description="Water or liquid spilled on the floor — a slip hazard.",
        category="safety", shape="point_in_time", cohort="both",
        recommended=False, prompt_fragment=_ENV_HAZARD_WATER_PROMPT_FRAGMENT,
        promotion_threshold=0.70,
    ),
    CatalogEntry(
        key="fire_smoke", name="Fire / smoke", icon_name="flame",
        description="Visible flames or smoke.",
        category="safety", shape="point_in_time", cohort="both",
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
        category="routine", shape="point_in_time", cohort="both",
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
        if e.category not in ("safety", "routine"):
            raise ValueError(f"[catalog] {e.key}: bad category {e.category!r}")
        if e.shape not in ("point_in_time", "state", "scheduled"):
            raise ValueError(f"[catalog] {e.key}: bad shape {e.shape!r}")
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
    today = datetime.now(timezone.utc).date()
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
        if e.shape != "scheduled"
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
}


def severity_for(identifier: str | None) -> int:
    """Within-tick precedence weight for a detector (higher wins). Custom /
    unknown detectors fall back to _DEFAULT_SEVERITY."""
    return _SEVERITY.get(identifier or "", _DEFAULT_SEVERITY)
