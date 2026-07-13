"""Two custom micro-slow detectors to generalize the crop-clean finding beyond
medication. Same design philosophy as the medication prompt: at camera distance
the object is a few pixels / invisible, so the prompt reads the MOTION / gesture,
with explicit FALSE cases to rule out confounders.

  collar_tug — micro, slow, body-part detail (hand→collar, grasp+pull)
  wearing_ring — micro, slow, SMALL OBJECT + fine two-hand fingertip manipulation

Importable by the recall harness (they follow the <id>_detected / _confidence /
_reasoning contract, so they drop into DetectorSpec / build_prompt_and_schema).
"""

from src.services.detection.prompt import DetectorSpec

COLLAR_TUG = DetectorSpec(
    identifier="collar_tug", name="Collar tug", always_on=False,
    prompt_fragment=(
        "collar_tug_detected: boolean. TRUE when the person deliberately TUGS, "
        "PULLS, or ADJUSTS the collar or neckline of their shirt with a hand — "
        "fingers grasping the collar/neckline and pulling it out, down, or to the "
        "side, or repeatedly adjusting it. The KEY signal is a hand rising to the "
        "neck/collar area and grasping and pulling the FABRIC there.\n"
        "\n"
        "This camera is mounted far away, so read the MOTION and hand position, not "
        "fine fabric detail: a hand travels up to the collar line, the fingers close "
        "on the shirt collar, and pull or adjust it (often more than once).\n"
        "\n"
        "Set FALSE if any of:\n"
        "  - a hand scratches the bare neck, jaw, or face with no fabric grasp;\n"
        "  - the person adjusts their hair, or touches a necklace/chain;\n"
        "  - a hand simply rests on the chest or upper body;\n"
        "  - any hand-to-neck motion where the collar is not clearly grasped and "
        "pulled;\n"
        "  - no person is visible, or the scene is too far / unclear to judge.\n"
        "\n"
        "collar_tug_confidence: number 0-1. 0 = no collar tug. 0.3-0.6 = a "
        "hand-to-neck motion but you cannot tell the collar is grasped/pulled. "
        "0.7+ = a clear grasp-and-pull or repeated adjust of the shirt collar.\n"
        "collar_tug_reasoning: 1 short sentence (e.g. 'right hand rose to the collar "
        "and tugged it down twice', 'hand scratched the neck, no collar grasp', "
        "'no person visible')."),
    schema_fields={"collar_tug_detected": {"type": "boolean"},
                   "collar_tug_confidence": {"type": "number"},
                   "collar_tug_reasoning": {"type": "string"}})

WEARING_RING = DetectorSpec(
    identifier="wearing_ring", name="Wearing a ring", always_on=False,
    prompt_fragment=(
        "wearing_ring_detected: boolean. TRUE only when the person slides a RING "
        "onto a FINGER. The ring itself is a few pixels and effectively invisible "
        "at this distance, so judge by WHERE the placing motion happens and where "
        "it ENDS:\n"
        "  - RING (TRUE): contact begins at the FINGERTIPS of the receiving hand "
        "and the slide travels DOWN A SINGLE FINGER toward the knuckle, with the "
        "receiving hand's fingers extended; often ends with spreading or flexing "
        "the fingers to inspect the hand.\n"
        "  - WRISTWEAR (FALSE): the motion WRAPS AROUND THE WRIST above the hand — "
        "putting on a watch, fitness band, or bracelet. The hands work AT THE "
        "WRIST JOINT with a wrap / clasp-press / strap-threading action, and the "
        "slide crosses the wrist, not along a finger.\n"
        "\n"
        "FIRST decide where the object ended up — on a FINGER or around the WRIST "
        "— and NAME that location in your reasoning. If you cannot confidently "
        "tell finger from wrist, set FALSE with confidence <= 0.4 — do NOT guess "
        "TRUE from a generic hands-together placing motion.\n"
        "\n"
        "Set FALSE if any of:\n"
        "  - the object goes around the WRIST (watch / fitness band / bracelet) — "
        "the most easily confused action;\n"
        "  - the hands clasp, wring, rub together, interlace, or crack knuckles;\n"
        "  - applying lotion / massaging the hands;\n"
        "  - any two-hand contact where you cannot see the slide END on a single "
        "finger;\n"
        "  - no person is visible, or the action is too far / unclear to judge.\n"
        "\n"
        "wearing_ring_confidence: number 0-1. 0 = nothing / wristwear. 0.3-0.6 = "
        "a placing motion but finger-vs-wrist uncertain (must stay FALSE). 0.7+ = "
        "the slide clearly ended on a single finger (fingertip-to-knuckle).\n"
        "wearing_ring_reasoning: 1 short sentence that NAMES the location (finger "
        "or wrist) — e.g. 'slid an object down the left ring finger, then spread "
        "the fingers', 'wrapped a band around the wrist and pressed a clasp — "
        "wristwear, not a ring', 'hands together but cannot tell finger from "
        "wrist — not confident'."),
    schema_fields={"wearing_ring_detected": {"type": "boolean"},
                   "wearing_ring_confidence": {"type": "number"},
                   "wearing_ring_reasoning": {"type": "string"}})

# ── RAPID-MOTION detectors (dual-channel: clean + overlay in ONE call) ──────
# Design (2026-07-12): convulsion and body_shaking are MUTUAL confounders and
# always ride the same prompt. Amplitude decides between them — never the
# absence of purposeful action (ill/elderly people shiver while otherwise
# motionless). MOTION_DESCRIPTION is always_on so it precedes the booleans
# (describe-then-decide) without entering primary_detection's firing list.

MOTION_DESCRIPTION = DetectorSpec(
    identifier="motion_description", name="Motion description", always_on=True,
    prompt_fragment=(
        "BEFORE the detector fields below, DESCRIBE THE MOTION using both "
        "halves of the split-screen.\n"
        "From the RIGHT half (the motion-tint copy) fill:\n"
        "motion_region: string. The body part(s) or frame area where motion "
        "concentrates ('none' if nothing moves).\n"
        "motion_spread: string, one of: none | localized (one small area, e.g. "
        "one hand) | regional (one limb or the torso) | whole_body.\n"
        "motion_rhythm: string, one of: none | rhythmic (regular repeating "
        "beat) | continuous (steady, no distinct beats) | sporadic (isolated "
        "movements).\n"
        "motion_amplitude: string, one of: none | flicker (tint pulses in "
        "place, body outline stays put — fine trembling) | displacement (body "
        "parts visibly TRAVEL between frames — gross movement).\n"
        "From the LEFT half (the clean copy) fill:\n"
        "motion_voluntary: string, one of: voluntary (a recognizable purposeful "
        "action — rubbing, wiping, adjusting, exercising) | involuntary "
        "(stereotyped, uncontrolled, serves no visible purpose) | unclear. "
        "Judge voluntariness ONLY from what the person is actually doing in "
        "the clean left half, never from the tint.\n"
        "Fill these five fields FIRST; every detector answer below MUST be "
        "consistent with them."),
    schema_fields={"motion_region":    {"type": "string"},
                   "motion_spread":    {"type": "string"},
                   "motion_rhythm":    {"type": "string"},
                   "motion_amplitude": {"type": "string"},
                   "motion_voluntary": {"type": "string"}})

# Channel-neutral variant for CLEAN-ONLY arms (no split-screen, no tint).
# Same identifier + schema fields — never pass both variants to one call.
MOTION_DESCRIPTION_CLEAN = DetectorSpec(
    identifier="motion_description", name="Motion description", always_on=True,
    prompt_fragment=(
        "BEFORE the detector fields below, DESCRIBE THE MOTION you observe. "
        "Watch the person carefully ACROSS frames — small, fast, repetitive "
        "movements matter and are easy to miss.\n"
        "motion_region: string. The body part(s) where motion concentrates "
        "('none' if nothing moves).\n"
        "motion_spread: string, one of: none | localized (one small area, e.g. "
        "one hand) | regional (one limb or the torso) | whole_body.\n"
        "motion_rhythm: string, one of: none | rhythmic (regular repeating "
        "beat) | continuous (steady, no distinct beats) | sporadic (isolated "
        "movements).\n"
        "motion_amplitude: string, one of: none | flicker (body outline stays "
        "put, surface trembles in place — fine trembling) | displacement "
        "(body parts visibly TRAVEL between frames — gross movement).\n"
        "motion_voluntary: string, one of: voluntary (a recognizable purposeful "
        "action — rubbing, wiping, adjusting, exercising) | involuntary "
        "(stereotyped, uncontrolled, serves no visible purpose) | unclear.\n"
        "Fill these five fields FIRST; every detector answer below MUST be "
        "consistent with them."),
    schema_fields={"motion_region":    {"type": "string"},
                   "motion_spread":    {"type": "string"},
                   "motion_rhythm":    {"type": "string"},
                   "motion_amplitude": {"type": "string"},
                   "motion_voluntary": {"type": "string"}})

CONVULSION = DetectorSpec(
    identifier="convulsion", name="Convulsion", always_on=False,
    prompt_fragment=(
        "convulsion_detected: boolean. TRUE when the person shows a "
        "seizure-like CONVULSION: GROSS, rhythmic, involuntary jerking of the "
        "limbs, torso, or head — body parts visibly DISPLACE between frames in "
        "a repeating, stereotyped pattern the person does not control. A "
        "convulsion can happen in ANY position — lying down, sitting in a "
        "chair, slumped over, or standing. Do NOT require lying down.\n"
        "\n"
        "Convulsion vs shivering — the REQUIRED distinction: a convulsion is "
        "LARGE-amplitude (arms/legs/head travel visibly between frames); "
        "shivering/trembling is fine, SMALL-amplitude surface shaking where "
        "the body outline stays in place (report that under body_shaking, NOT "
        "here). The two are mutually exclusive — decide by AMPLITUDE and NAME "
        "the amplitude (gross displacement vs fine tremble) in your reasoning. "
        "A person who is otherwise still and doing nothing is NOT thereby "
        "convulsing — stillness plus fine trembling is shivering.\n"
        "\n"
        "The red motion tint alone is NEVER sufficient evidence: red marks any "
        "pixel change, including ordinary movement. Confirm the actual jerking "
        "in the clean view before setting TRUE.\n"
        "\n"
        "Set FALSE if any of:\n"
        "  - fine trembling/shivering with no gross limb displacement "
        "(body_shaking instead);\n"
        "  - vigorous but VOLUNTARY activity: exercising, stretching, "
        "scratching hard, rolling over, adjusting position or blankets, "
        "rocking, dancing;\n"
        "  - repetitive purposeful motion: rubbing arms, wiping, shaking an "
        "object, typing, waving, bouncing a leg;\n"
        "  - a person simply moving quickly or restlessly;\n"
        "  - no person is visible, or too unclear to judge.\n"
        "\n"
        "convulsion_confidence: number 0-1. 0 = no convulsion. 0.3-0.6 = "
        "rhythmic motion but amplitude or voluntariness uncertain (must stay "
        "FALSE). 0.7+ = clear gross rhythmic involuntary jerking.\n"
        "convulsion_reasoning: 1 short sentence NAMING the amplitude and "
        "voluntariness (e.g. 'whole-body rhythmic jerking with gross arm "
        "displacement, no purpose — convulsion', 'fine in-place trembling "
        "only — shivering, not convulsion', 'vigorous but voluntary "
        "stretching')."),
    schema_fields={"convulsion_detected":   {"type": "boolean"},
                   "convulsion_confidence": {"type": "number"},
                   "convulsion_reasoning":  {"type": "string"}})

BODY_SHAKING = DetectorSpec(
    identifier="body_shaking", name="Body shaking / shivering", always_on=False,
    prompt_fragment=(
        "body_shaking_detected: boolean. TRUE when the person shows sustained "
        "fine SHAKING / SHIVERING / TREMBLING: high-frequency, small-amplitude "
        "involuntary trembling of the body, limbs, or hands — the body outline "
        "stays essentially in place while the surface trembles continuously "
        "(shivering from cold, fever rigors, or a tremor). Often with a "
        "hunched or curled posture. IMPORTANT: the person may be otherwise "
        "completely still — in ill or elderly people shivering frequently "
        "occurs with NO other activity, so the absence of purposeful movement "
        "is NOT evidence against shivering and NOT evidence for a convulsion. "
        "Amplitude decides.\n"
        "\n"
        "Mutually exclusive with convulsion_detected: gross rhythmic jerking "
        "with visible limb displacement is a convulsion, not body_shaking.\n"
        "\n"
        "Set FALSE if any of:\n"
        "  - gross rhythmic jerking with visible limb displacement "
        "(convulsion instead);\n"
        "  - voluntary rubbing of arms/hands, rocking, bouncing a leg, "
        "fidgeting, or adjusting clothing/blankets;\n"
        "  - shaking an OBJECT (bottle, blanket, cushion) rather than the "
        "body itself trembling;\n"
        "  - apparent flicker in the tint with no visible trembling in the "
        "clean view (compression/noise);\n"
        "  - no person is visible, or too unclear to judge.\n"
        "\n"
        "body_shaking_detected is a STATE: judge whether trembling is present "
        "and sustained within this clip.\n"
        "body_shaking_confidence: number 0-1. 0 = no trembling. 0.3-0.6 = "
        "possible fine trembling but not clearly sustained or not clearly "
        "involuntary (must stay FALSE). 0.7+ = clear sustained fine trembling "
        "of the body/limbs.\n"
        "body_shaking_reasoning: 1 short sentence NAMING the amplitude (e.g. "
        "'continuous fine in-place trembling of torso and arms, hunched — "
        "shivering', 'briskly rubbing arms — voluntary warming, not "
        "shivering')."),
    schema_fields={"body_shaking_detected":   {"type": "boolean"},
                   "body_shaking_confidence": {"type": "number"},
                   "body_shaking_reasoning":  {"type": "string"}})

GLASSES_ON = DetectorSpec(
    identifier="glasses_on", name="Putting on eyeglasses", always_on=False,
    prompt_fragment=(
        "glasses_on_detected: boolean. TRUE when the person PUTS ON eyeglasses — "
        "picks up or raises a pair of glasses with one or both hands, brings them "
        "up to the FACE, and seats them on the nose/ears (often a small two-handed "
        "adjust at the temples afterward). The KEY signal is the hand(s) carrying "
        "something small up to EYE level and the settling motion at the face.\n"
        "\n"
        "This camera is far away, so the glasses themselves may be barely visible — "
        "read the MOTION: hand rises to eye level (NOT the mouth, NOT the chin), a "
        "brief precise placing motion at the eyes/ears, then the hands drop away.\n"
        "\n"
        "Set FALSE if any of:\n"
        "  - the person rubs their eyes, wipes their face, or shields their eyes;\n"
        "  - adjusts hair, a hat, or earphones;\n"
        "  - TAKES OFF glasses (removal does not count — only putting on);\n"
        "  - a hand passes the face without a settling motion at the eyes;\n"
        "  - no person is visible, or too far / unclear to judge.\n"
        "\n"
        "glasses_on_confidence: number 0-1. 0 = nothing glasses-like. 0.3-0.6 = a "
        "hand at eye level but no clear seating motion. 0.7+ = a clear "
        "raise-to-eyes + seat-on-face motion.\n"
        "glasses_on_reasoning: 1 short sentence (e.g. 'raised both hands to eye "
        "level and seated glasses, adjusted temples', 'rubbed eyes, no object "
        "placed', 'no person visible')."),
    schema_fields={"glasses_on_detected": {"type": "boolean"},
                   "glasses_on_confidence": {"type": "number"},
                   "glasses_on_reasoning": {"type": "string"}})

PHONE_USE = DetectorSpec(
    identifier="phone_use", name="Using a phone (scrolling)", always_on=False,
    prompt_fragment=(
        "phone_use_detected: boolean. TRUE when the person is actively USING a "
        "handheld phone — holding a small flat object in one hand at chest/lap "
        "height, face angled DOWN toward it, with the other hand's thumb/finger "
        "making small repeated swipe/tap motions over its surface, sustained for "
        "several seconds.\n"
        "\n"
        "The phone itself may be tiny at this distance — read the POSTURE + MOTION: "
        "head tilted down at a hand held still in front of the body, plus small "
        "repetitive fingertip motions on the held object. This is DIFFERENT from "
        "bringing a hand to the FACE or EAR.\n"
        "\n"
        "Set FALSE if any of:\n"
        "  - the person holds the object to their EAR (a call is not scrolling);\n"
        "  - reading a book/paper (larger object, page-turn motion, two-handed "
        "grip at wider spread);\n"
        "  - hands are idle in the lap with the head down (dozing, resting);\n"
        "  - a brief glance at the object without sustained interaction;\n"
        "  - no person is visible, or too far / unclear to judge.\n"
        "\n"
        "phone_use_confidence: number 0-1. 0 = no phone use. 0.3-0.6 = head-down "
        "posture but no visible fingertip interaction. 0.7+ = clear sustained "
        "hold + look-down + repeated swipe/tap motions.\n"
        "phone_use_reasoning: 1 short sentence (e.g. 'held a small object at chest "
        "height, head down, thumb swiping repeatedly for ~8s', 'object held to "
        "ear — a call, not scrolling', 'no person visible')."),
    schema_fields={"phone_use_detected": {"type": "boolean"},
                   "phone_use_confidence": {"type": "number"},
                   "phone_use_reasoning": {"type": "string"}})
