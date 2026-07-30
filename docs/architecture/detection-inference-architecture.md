# Detection architecture — taxonomy, perception, eventing

> Status: **SUPERSEDED by `detection-architecture-v2.md`** (2026-07-13), which
> generalizes this material into the hierarchical detector model. This file is kept
> as the **experimental-evidence record** — §4's measured results remain the proof
> base cited by v2. Do not extend this file; extend v2.
>
> **Naming drift vs shipped code (reconciled 2026-07-27)** — this doc's design
> vocabulary predates the phase C–H2 refactors. Where it says:
> - `persistence: point / state` and "'Scheduled' is not a shape" → shipped as the
>   **alerting spine**: `alert_trigger` (`leading`/`trailing`/`duration`/`absence`)
>   + `is_sustained` on the `detectors` row; the `shape` column/enum was dropped.
> - `requires_movement` (§7) → shipped as the `requires_motion` boolean column
>   (the intermediate `evidence_branch` enum was later collapsed into it).
> - k-of-N / windowed-count M (§6) → the within-clip confirm exists in
>   `decision.py`; the **entry M-of-N shipped as an INTERNAL confidence gate
>   derived from detection difficulty** (`compiler_v2.py`), never user-facing.
> - Multi-person fan-out (§5) → shipped with a **global home-wide crowd
>   fallback**: above `scheduler_crowd_fallback_tracks` tracks, crop buckets
>   fall back to full-scene.
> - Layer-1/Layer-2 tracking-vs-attribution (§5) → shipped as the SAM3+EdgeTAM
>   continuous perception pod (`gpu/perception/`) + the `gpu/identity/` sidecar,
>   not the in-process YOLO+BoT-SORT stack described here.
> The measurements in §4 are unaffected.

> Original status: **design spec** (2026-07-11, re-drafted). The north star is **detection
> accuracy** across the whole size×speed space — large/small, slow/fast — plus an
> **eventing layer that is consistent and not inundatory**. Cost is a set of knobs
> (§7), not a goal. Nothing here has shipped; prod today is a single full-scene VLM
> call with every detector in one prompt.

---

## 1. What we're fixing

Prod makes **one VLM call per tick**: whole scene, every detector in one prompt, one
global motion overlay. Three accuracy failures follow:

1. **Cross-talk false positives** — one model checking N things at once does each
   worse; a single ambiguous action satisfies several detectors (a phone → a pill; a
   moving hand → a seizure). Measured: a benign clip fired `emergency`, `seizure`
   *and* `medication` across runs of the *same* footage.
2. **The overlay corrupts micro-object reasoning** — great for rapid tremor, but on a
   zoomed hand it *drives* false positives (§4).
3. **Micro and rapid signals are lost** — a pill or a fine tremor is a few pixels in a
   1080p wide frame; the model can't see what isn't resolved.

The fix is to **route each detector to the perception its signal actually needs**, run
focused prompts per pixel-set, and make eventing deliberate.

---

## 2. The detector taxonomy — 6 core properties

A detector is fully specified by six properties the custom-detector **agent infers**
from natural language. Everything else *derives* from them.

| # | Property | Values | What it is |
|---|---|---|---|
| 1 | **criteria** | free text | the observable definition (→ prompt fragment) |
| 2 | **temporality** | presence / action | one frame answers it, vs a sequence is the signal (action may be video and/or audio) |
| 3 | **size** | macro / micro | whole body/scene vs a small object/part |
| 4 | **speed** | slow / rapid | *(actions only)* deliberate vs fast-rhythmic |
| 5 | **category** | safety / routine | alert-anyone-now vs the patient's record |
| 6 | **persistence** | point / state | a discrete event vs a condition that holds over time |

Plus **scope** (`where` / `when` / `sensitivity` — *user-given*, not inferred) and one
agent **judgment**: **FP-propensity** (how confusable the signal is).

Everything else is **derived**:

```
temporality ─────────────► modality (still · video · audio)
temporality + size + speed ► BUCKET (still · wide · person-crop · rapid)
size ────────────────────► crop? · media_res
speed ───────────────────► fps · overlay (rapid only)
category ────────────────► attribution · urgency · severity
category + sensitivity ──► confidence threshold
FP-propensity + urgency ─► describe-then-decide · confirmation k
persistence ─────────────► edge-trigger + debounce (state) vs fire (point)
temporality + modality ──► requires_movement (runtime cost gate, §7)
```

Notes that keep the six clean:
- **State ≠ no-motion.** State is *persistence* (yoga is a state full of motion). It's
  orthogonal to movement and drives only how we *event* it, not how we perceive it.
- **"Scheduled" is not a shape** — it's a `when`-scope time filter.
- **Absence** ("X has *not* happened by T") is not a perception property — it's an
  **eventing rule** composed over a base detector (§6).

There is **no motion gate** in the taxonomy. (Motion may return purely as a runtime
cost optimization — §7 — invisible to detector definitions.)

---

## 3. Perception — buckets (the unit of a VLM call)

`temporality + size + speed` collapse each detector onto one **bucket** = one distinct
pixel-set = one focused VLM call. Grouping by pixel-set is what removes the cross-talk
FPs *and* the wasted cost of paying for the same pixels per-detector.

| Bucket | Frames | Overlay | Serves | Why (accuracy) |
|---|---|---|---|---|
| **STILL** | 1–3 sharp stills | no | presence / objects | max spatial detail, no motion needed |
| **WIDE** | full-scene video | no | macro actions | fall, wandering, choking — VLM sees it whole |
| **PERSON-CROP** | tracked person crop | **no** | micro-**slow** actions | zoom makes a small gesture legible; overlay banned (§4) |
| **RAPID-MOTION** | tracked crop, **overlay-only**, 8fps, **full-flash tier** | **yes** (sole input) | **all rapid** actions (macro + micro) | fast rhythm *is* the signal; flash reads the tint as motion evidence AND judges purpose from the scene beneath it |
| **AUDIO** | audio (+ wide context) | n/a | vocal signals | rides an action call's audio |

**The action 2×2** (size × speed), with the worked example and the tool per quadrant:

| | **slow** | **rapid** |
|---|---|---|
| **macro** | fall, wandering — *WIDE, VLM alone* | convulsion — *RAPID (crop overlay-only 8fps, flash)* |
| **micro** | medication, eating, collar-tug — *PERSON-CROP, no overlay* | seizure tremor — *RAPID*; shivering — **below sensor floor** (§4) |

Presence detectors sit outside this grid (no speed) — classified by size alone, served
by STILL. **Crop: size decides for slow actions; ALL rapid actions crop (no rapid
detector needs the environment). Overlay and fps are decided by speed.**

**STILL criteria (locked 2026-07-13).** A detector is STILL iff ALL of:
1. **The photograph test, strict form: one frame SETTLES it, not suggests it.** A
   stranger shown ONE photo answers correctly with no inference — the evidence is
   fully contained and unambiguous in the frame's pixels: door open, person in bed,
   plate on table, walker present, dressed vs pajamas. If the verdict is an
   INFERENCE assembled from multiple weak cues (gaze + audio + duration → "watching
   TV"), it is NOT a still, even though a photo often hints at it — it routes to a
   video(+audio) call. And never what anyone is DOING: any verb of motion → video
   bucket (measured: stills score 47% on actions). STILL is only for the obvious.
2. **It's a state, not an instant:** the condition lingers, so it stays true for
   whichever motion-gated tick's frame we sample and is re-checkable later.
   Edge-triggered eventing applies.
3. **Above the object-size floor:** person-scale and macro objects qualify; micro
   objects (pill bottle: 0/18 full-frame) do not until open-vocab localization
   exists. Floor location TBD from the vanilla-object test.
No dedicated trigger: STILL frames come from ticks the motion gate already produces
(scene transitions make motion). Every STILL detector must pass FP confounders that
defeat naive presence reads (people on TV screens, photos on walls, clothes on chairs).

**RAPID collapses to ONE bucket (decided + validated 2026-07-12).** Macro-rapid and
micro-rapid share RAPID-MOTION: the tracked person crop with the motion overlay,
**overlay-only, 8fps, on the full-flash model tier**, with a describe-then-decide
schema (`{region, spread, rhythm, amplitude, voluntary}` filled before any boolean).
The dual-channel idea (clean + overlay in one call) was tested and LOST — see §4:
two video parts silently collapse in the API; side-by-side halves resolution; a
temporal-concat raw segment re-injects the "person resting" prior and *lowers* recall.
**Model tier is a hard requirement**: flash-lite is blind to rapid motion under every
packaging (it pattern-matches "red = seizure" instead of reading motion — the root
cause of prod's overfiring seizure detector); full flash reads the tint as motion
evidence and judges voluntariness from the scene beneath it. Confounder law for the
flagship pair: convulsion vs shivering is decided by **amplitude** (gross limb
displacement vs fine in-place flicker), never by absence of purposeful action —
ill/elderly people shiver while otherwise motionless, and convulsions happen in any
position, not just lying down.

**A detector is three orthogonal choices (framing 2026-07-12): perception shape ×
evidence unit × eventing shape.** Perception shapes are the buckets above (plus
FACE-CROP below). Evidence units — what ONE tick emits:
  - **verdict** (boolean + confidence + reasoning) — one tick can constitute the event
    (fall, convulsion, medication);
  - **evidence instance** — observations logged, NEVER per-tick alerts (word-groping,
    task restarts, isolated jerks; confusion is an instance extractor across visual +
    audio in the same call);
  - **state sample** (motionless=true this tick — often free from the gate lattice).
Eventing shapes compose ticks (§6): point + k-of-N, windowed-count states,
schedules/absence, day-level consolidation. Jerks = instances + sparse windowed count;
confusion = instances + consolidation. Neither needed a new bucket.

**FACE-CROP (planned geometry variant, untested).** Micro signals split by body
region: hand/limb scale rides the person crop; FACE-scale signals (lip-smacking
automatisms, facial twitching, eye state) get a tighter crop from the InsightFace box —
same overlay/fps/model-tier rules as their speed class (lip smacking ≈ 1–3 Hz, safely
under the 8fps Nyquist limit — the open question is resolution, not sampling). Next
recording experiment.

**Bucket sizing — how many detectors per call (measured 2026-07-12):** dilution curve
on identical crop-clean inputs (micro target, 4 clips × N=10): **1 det = 92%, 3 det =
90% (tie, within the ±5% run-to-run noise), 5 det = 72%.** The cliff is between 3 and
5, and the hardest clip sagged already at 3 (9→6→5) — dilution eats weak evidence
first. Rule: **≤3 detectors per VLM call; 1–2 for safety-critical or known-hard
contexts; a bucket with more actives splits into multiple calls.** (Caveat: measured
on a micro target; macro-target dilution is untested — measurable on the 18 labeled
agitation positives without new recording.)

---

## 4. What we've validated (accuracy evidence)

Measured on real footage, `gemini-3.1-flash-lite`, N=10.

**Precision** — benign clip, every fire is a false positive (lower better):

| Method | Medication FP |
|---|---|
| crop, **no overlay** | **0/10** |
| high-res stills | 0/10 |
| native + overlay (prod) | 0–1/10 |
| crop **+ overlay** | **4/10** (reproduced) |

**Recall** — 3 real medication events, event caught if any tick fires (higher better):

| Method | Event recall |
|---|---|
| **crop-clean (D)** | **100%** |
| crop-clean + crowded prompt (F) | **100%** |
| crop + overlay (B) | 93% |
| native + overlay, med-only (A) | 87% |
| native + overlay, **crowded prompt (E)** | **53%** |
| stills (C) | 47% |

Conclusions, all reproduced/mechanistic:
- **Crop-clean wins recall *and* precision** for micro-slow actions → the PERSON-CROP branch.
- **Overlay is toxic on a crop** (B: 4/10 FP, 93% recall < D) → overlay only on RAPID-MOTION.
- **A crowded prompt halves recall on native (87→53%); crop rescues it (F=100%)** → per-bucket focused prompts, and crop makes shared prompts survivable.
- **Stills can't do actions** (47%) → STILL is for presence only.
- **At deployment distance, medication is gesture-recognition, not object-recognition** — a container was invisible to *both* stills and video (0/18); the model reads the *motion*, which is why crop-video wins and stills lose.

**The perceptual floor (measured 2026-07-12, wearing_ring):** a micro-slow gesture is
viable ONLY if its discriminating detail has **arm-scale correlates**. `wearing_ring`
(pure fingertip manipulation) vs Whoop-band-donning confusers: v1 prompt = 83% recall /
**87% FP** (fires on the generic hands-together pose — zero discrimination); v2 prompt
(nearest-confuser named, finger-vs-wrist decision forced, uncertain→FALSE) = **0% FP /
7% recall** — the model correctly names wristwear 30/30 but cannot affirmatively see a
finger-slide. No operating point exists between: **prompts control where the errors go,
not whether the information exists.** Consequences: (a) prompt-craft law — every micro
prompt names its nearest confounder + forces the discriminating question + defaults
uncertain→FALSE; (b) the **compiler agent needs a feasibility gate** ("does the detail
have arm-scale correlates?") and must refuse below-floor requests (eye-twitch,
finger-only motions) rather than ship coin-flips; (c) wristwear-donning IS viable
(30/30 recognized) — the floor is motion amplitude, not object size.

**RAPID validated (measured 2026-07-12, convulsion + shivering, 2 conv TP events, 2
shiv TP events, 6 confounder takes, all 8fps captures, tracked crop):**

- **Model tier is the gate.** `gemini-3.1-flash-lite` (prod) is blind to rapid motion
  under EVERY packaging tried (clean, side-by-side, temporal concat, overlay-only;
  4 and 8 fps): 0–2/10 recall, reads acted convulsions as "completely still" or
  "voluntary postural adjustments", and its only convulsion fire in a 200-call 2×2
  was a hallucination on a confounder. `gemini-3-flash-preview` on identical inputs:
  see below. This also explains prod's seizure detector: lite can't read motion, so
  overlay-mode degenerates to "red = seizure" (overfires) and clean-mode to "person
  lying down" (underfires).
- **Packaging findings.** (1) TWO video parts in one Gemini request silently collapse
  — the model reports "one video" and attends to one channel; never ship dual-part
  requests. (2) Side-by-side split-screen works but halves per-channel resolution.
  (3) Temporal concat (overlay segment then raw segment, loop-back timestamp in
  prompt) *reduced* recall vs overlay alone (7/10 vs 10/10 solo) — the raw "resting
  on sofa" footage re-injects the pose prior. (4) **Overlay-only wins.**
- **Decision run** (overlay-only crop 8fps, flash-preview, BOTH detectors in one
  call, N=5 × 10 events): convulsion **recall 80% (8/10), FP 2% (1/30 confounder
  samples), 0/10 cross-fire on shivering — discrimination +78%.** Misses were the
  amplitude/voluntary coin landing wrong once per clip; a real convulsion persists
  over many consecutive ticks, so per-tick 80% compounds to ~certain at event level;
  the lone stochastic confounder FP is exactly what k-of-N confirm suppresses.
- **Shivering is below the SENSOR floor at 1080p/8fps.** body_shaking recall 0/10 —
  and the frame-diff overlay itself shows only intermittent wisps of tint on the
  shivering takes (vs solid whole-body red on convulsions): ~10Hz sub-inch tremor
  exceeds Nyquist at 8fps and is near sub-pixel at crop scale. The model's "no fine
  trembling visible" is factually correct about the pixels. Feasibility gate:
  fine-tremor detectors (shivering, Parkinsonian rest tremor, eye-twitch) are
  below-floor on current hardware; do not ship. The convulsion/shivering AMPLITUDE
  boundary held perfectly in the other direction (no conv fire on shiv takes).

**STILL object floor (measured 2026-07-13; 1080p full-frame sharpest still, lite,
N=10, ~1,100 image tokens @high — the max image budget, so not input-starved):**
- `phone_on_couch`: **10/10 on the real phone AND 10/10 on a black TV remote** —
  slab-scale (~15 cm) object *presence* is visible but *identity* is below the
  discrimination floor. The prompt forced phone-vs-remote naming with
  uncertain→FALSE; the model confidently named "smartphone" both times (the
  wearing_ring law: prompts can't add information the pixels lack).
- `meal_plate` (plate + apple, dining table, background distance): **0/10** —
  plainly visible to a human in the same still; the model described the nearer
  coffee table instead. Distance/salience kills mid-object presence.
- **Tier does NOT fix stills** (measured: flash-preview identical — 10/10 FP on the
  remote, 0/10 on the plate). Stills fail at the ENCODER: both tiers compress the
  frame to the same ~1.1k-token budget; a background object doesn't survive it no
  matter who reasons about it. (Contrast RAPID, where the failure was motion
  *reasoning* and tier fixed it.)
- **REGION-CROP RESCUES STILLS, ON LITE** (measured, same frames/prompts): couch
  crop → phone 10/10 AND remote 0/10 ("identifiable by the grid of buttons") —
  perfect discrimination; table crop → plate+apple 9/10. The crop law generalizes:
  it was never model capability, it's token-budget allocation.
- **STILL recipe (final, 2026-07-13): zone-crop stills on lite, reserved for the
  obvious.** Measured on lite: zone crops took phone TP 10/10 / remote FP 10/10→
  **0/10** ("identifiable by the grid of buttons") / distant plate 0/10→**9/10**.
  Default size rule, no machinery: **objects smaller than a PLATE are assumed
  unreliable at distance — they require a zone crop; pill-bottle scale reroutes
  to the INTERACTION ("taking medication", not "pill bottle present" — the
  measured medication law) or is declined.** Large-or-near obvious states (person
  in bed, door open, walker-scale) work full-frame. Action/state feasibility
  stays BUCKET-level, certified once globally by the harness experiments — no
  per-home probing of anything.

**Every bucket now has a measurement.** WIDE + AUDIO: prod track record. PERSON-CROP:
medication/collar-tug (+ wearing_ring floor). RAPID: convulsion (+ fine-tremor floor).
STILL: object floor above. Remaining gaps: person-presence FP (TV-people confounder),
FACE-CROP, fidgeting micro, night/IR — see §9.

---

## 5. The architecture — compile → run

**Compile (agent, once per detector):** infer the 6 properties → emit the derived
config + prompt fragment. Because misreading *temporality* or *category* is
safety-critical, axis assignment runs on the reasoning model, not a cheap one.

**Run (per tick) — the full clip flow:**

```
tick (every 10s) → 11s window: frames + audio
        │
 ┌──────▼── GATE (exists today) ──────────────────────┐
 │ enabled & not paused?             no → SKIP        │
 │ person present (open) OR motion?  no → SKIP        │
 └──────┬─────────────────────────────────────────────┘
        │ pass
 ┌──────▼── TIER 0  (classical, local, cheap) ────────┐
 │ build MP4 · YOLO person boxes + BoT-SORT tracks    │
 │ re-ID → patient (attribution) · motion score       │
 │ [FUTURE] object detection → object boxes           │
 └──────┬─────────────────────────────────────────────┘
        │ route active detectors by (temporality + size + speed)
        │ [FUTURE] per-bucket motion gate: skip a bucket whose subject is still
 ┌──────▼── BRANCHES — one focused VLM call per active bucket (parallel) ──────────┐
 │  STILL          WIDE            PERSON-CROP       RAPID-MOTION      AUDIO        │
 │  stills         full-scene vid  person crop       crop + overlay    audio       │
 │  (+obj-crop     4fps · no ovl   high·4fps·no ovl  high·8fps         (+wide ctx) │
 │   [future])     macro actions   micro-SLOW acts   +describe         vocal       │
 │  presence/      (fall,choking)  (medication…)     micro-RAPID       (help…)     │
 │  objects                                          (seizure…)                    │
 └──────┬─────────────────────────────────────────────────────────────────────────┘
        │ per detector: {detected, confidence, reasoning}
 ┌──────▼── CONFIRM (k-of-N, same clip) ──────────────┐
 │ fired & k>1? → re-run SAME clip N×, require k      │
 │ (0 latency; urgent always k=1)                     │
 └──────┬─────────────────────────────────────────────┘
        │
 ┌──────▼── EVENTING (consistent · non-inundatory) ───┐
 │ threshold (conf ≥ bar × sensitivity)               │
 │ attribution (routine→patient · safety→anyone)      │
 │ intra-clip dedup (primary_detection → ONE event)   │
 │ persistence (state = fire-once-on-enter + debounce)│
 │ absence rules (scheduled: "not fired by T → alert")│
 │ severity + rate-limit (rank · cooldown · budget)   │
 └──────┬─────────────────────────────────────────────┘
        ▼  promote → notify caregiver
```

Multi-person: the micro branches need a subject — patient re-ID box → else the sole
person → else fan out per person (safety alerts on anyone; routine holds unattributed
pending re-ID). Never invent a subject.

**Tracking vs attribution — two layers, unbundle them (decision 2026-07-12):**

- **Layer 1 — tracking (always on, identity-free):** `body_detector` (YOLO) +
  `tracker` (BoT-SORT + OSNet appearance association) → stable anonymous track IDs,
  person count, crop targets. Verified dramatically more stable than naive
  ultralytics tracking (E4: 3 clean tracks vs 6 fragments/duplicates on the same
  clip). Needs no enrollment, no faces.
- **Layer 2 — attribution (optional, per-patient FaceID):** InsightFace face gallery
  (`embeddings`/`enrollment_processor`) + `harvester` (adaptive: appends
  confidently-matched wall-cam face crops to close the selfie→wall-cam domain gap)
  + `cross_clip_body_gallery` (body-appearance memory so an identified person stays
  attributed across clips when the face hides). Stamps names on Layer-1 tracks.
- The body gallery is dual-use: clusters work anonymously ("same second person as
  earlier") even without FaceID; FaceID only names them.
- **Today `reid_enabled` gates the whole bundle — FaceID off means NO tracking runs.**
  Unbundle: Layer 1 unconditional (feeds fan-out + subject resolution), Layer 2 per
  patient. Experiment harnesses must use the Layer-1 prod stack, not hand-rolled
  ultralytics tracking (measured: toy tracker produced the E4 wrong-person crops;
  prod stack tracked both people from frame 0).

**Deferred capabilities (both marked `[FUTURE]` above):**
- **Object detection (open-vocab).** Prerequisite for the STILL branch to crop small
  objects (stove, glass, walker); YOLO is person-only today. Without it, STILL only does
  macro presence (fire, masked face). See §9.
- **Motion-gating refinement (cost).** *Tighten the existing presence/motion gate* —
  which today fires on any person-present-or-motion — with **person-region** motion and
  **per-bucket** skipping (don't fire an action bucket whose subject is still). A pure
  runtime optimization, decoupled from the taxonomy. See §7.

---

## 6. Eventing — consistent and not inundatory

Perception decides *what happened*; eventing decides *whether the caregiver hears about
it*. This layer is what keeps alerts trustworthy and non-flooding.

- **Confirmation (k-of-N, same clip).** Re-run the firing clip N times, require k
  agreements. Attacks the **stochastic** FP (a detector fired 4/10 on the *same* benign
  clip). **Fire-then-confirm** (only re-run on a fire) → cheap; **zero added latency**
  (all N on the current tick) → safe for urgent detectors. `k=1` by default; `k=2–3`
  only for **relaxed + high-FP** detectors — **urgency always wins** (never delay a
  safety alert; buy FP suppression from describe-then-decide + threshold instead).
  *(k only fixes stochastic FPs; a systematic FP needs the prompt/threshold.)*
- **Promotion threshold.** Per-detector confidence bar × per-patient sensitivity
  (low/standard/high). Seizure sits lower (recall-biased); routine sits higher.
- **Intra-clip dedup.** When several detectors fire on one clip (a fall is also an
  emergency), `primary_detection` names the dominant one → **one event, not five.**
- **States are ONE primitive: edge-triggered windowed counting (unified 2026-07-12).**
  A state ENTERS when ≥M evidence-ticks land inside window T; it fires **once on
  enter**, stays silent while it persists, and EXITS edge-triggered (inverse
  predicate). Every duration variant is a **parameter choice, not a new type**:
    - **dense M** (≈ every tick in T): "motionless ≥5 min", "fidgeting ≥10 min" — the
      duration-threshold case, with dissent tolerance built in for free (set M
      slightly under T/tick so one flaky call doesn't reset the clock);
    - **sparse M**: accumulation patterns — "myoclonic activity present" (≥M
      jerk-instance ticks in T), stumble clusters, repeated-question counting.
  Perception stays per-tick (verdicts or logged evidence instances); the window does
  the composing. `point` events are the other primitive: fire on the discrete
  occurrence, k-of-N confirm applies there.
- **Absence / negation.** A scheduled rule over a base detector's history: "medication
  did **not** fire by 10am → alert." The SAME windowed-count primitive with count == 0
  and a deadline. Reuses the base detector's perception; adds a deadline + polarity.
  This is how "hasn't taken meds / hasn't moved in 12h / hasn't eaten" work — no new
  bucket.
- **Severity + rate-limit.** Rank by severity; per-detector cooldowns and a
  per-patient alert budget so a busy afternoon doesn't become 40 pushes.

---

## 7. Cost — knobs, not goals

Accuracy leads; these are the levers if a bill needs trimming. Token counts are
**measured** (`flash-lite`, 11s@4fps/high, 7 det): 13,236 in / 375 out. Components:
video **~221 tok/frame @ high**, **~59 @ medium** (3.7×, same coverage); still ~1,100;
prompt ~540 + ~390/detector.

- **Per-bucket media_res** — high→medium is 3.7× with no coverage loss (use where
  macro detection tolerates it).
- **Per-bucket fps** — video scales linearly; slow branches can drop to 1–2fps, rapid
  needs 8fps. (Tuned: RAPID 8fps, WIDE + PERSON-CROP high/4fps.)
- **Crop is token-neutral** — per-frame budget is set by media_res, not source
  resolution. Crop buys *accuracy*, not cost.
- **Fire-then-confirm** keeps k-of-N cheap (N× only on a fire).
- **`requires_movement` gate (decided 2026-07-12)** — a **derived** runtime attribute,
  not a taxonomy axis: `requires_movement = action AND video-modality` (audio actions
  gate on audio energy; presence detectors never gate). If a clip has no **sustained**
  person-movement, skipping a movement-requiring detector is recall-safe *by
  definition* — the action didn't happen.
  - **Sustained** = motion active ≥1s continuous OR ≥2s cumulative in the 11s window
    (parameterized); robust to 1–2 sporadic sub-second twitches/flinches. Tune the
    thresholds on the eval set (no labeled true event may fall below the gate).
  - **Per-person, via Layer-1 tracking** (always on per the decouple decision): track
    centroid displacement + in-box change per subject — immune to TV/pet false-wakes,
    and the crop branch gates on *its subject's* movement.
  - **The gates form a LATTICE that opens detector SETS (framing 2026-07-12).**
    `person_present` and `sustained_motion` are not detectors — they are the Layer-1
    substrate every detector is gated by:
      - no person → only room/absence states run;
      - person + NO motion → presence states + STILL checks run, and the no-motion
        ticks themselves are the evidence feed for "motionless"-style windowed-count
        states. **The gate is the trigger, never the verdict**: no-motion ticks are
        semantically blind (motionless-watching-TV ≠ motionless-unresponsive). When
        the windowed count crosses its threshold, ONE cheap context call (STILL or
        short VLM) classifies the inactivity — watching TV / reading / sleeping /
        staring blankly / slumped — and that single call routes ALL inactivity-family
        detectors at once (suppresses the unresponsive alert, feeds watching_tv,
        fires staring-spell). The gate decides WHEN to ask; the context call decides
        WHAT is true.
      - person + motion → action buckets open (WIDE / PERSON-CROP / RAPID by class).
  - **Scene states fire on motion-edge — no extra trigger machinery (decided
    2026-07-13).** Every scene transition makes motion (door opening, plate
    arriving), so the existing motion gate serves them; transitions missed during a
    blind window are a system-availability problem, not a detector-design problem.
    **STILL detectors are strictly limited to point-in-time location / object /
    scene description** — anything needing motion or behavior is a video bucket.
  - **Seizure exemption is MOOT (resolved 2026-07-12):** fine tremor is below the
    sensor floor regardless of gating, and a real convulsion lights whole-body
    frame-diff for seconds — it clears any reasonable threshold. The gate can be
    strict.
- **Not a lever:** cadence (only "saves" by going blind between clips).

Order-of-magnitude, 1 camera 24/7 with a person always present: single-digit
dollars/camera-day; dominated by tick frequency × active buckets. Real homes with
intermittent presence fall well below.

---

## 8. The 24 built-in detectors → buckets

`T` = **P**resence / **A**ction. Overlay only on micro-rapid. `safety` / `routine` by stripe intent.

| Detector | cat | persist | T | size | speed | Bucket | Overlay |
|---|---|---|---|---|---|---|---|
| fall | safety | point | A | macro | rapid | WIDE | no |
| agitation | safety | state | A | macro | rapid | WIDE + AUDIO | no |
| seizure | safety | point | A | micro | rapid | **RAPID-MOTION** | **yes** |
| wandering | safety | point | A | macro | slow | WIDE | no |
| medication | routine | point | A | micro | slow | **PERSON-CROP** | no |
| climbing | safety | state | A | macro | slow | WIDE | no |
| distress | safety | state | A | macro | slow | WIDE + AUDIO | no |
| child_crib_exit | safety | point | A | macro | slow | WIDE | no |
| child_cord_entanglement | safety | point | P | micro | — | STILL | no |
| child_stairs | safety | point | A | macro | slow | WIDE | no |
| child_hazard_reach | safety | point | A | micro | slow | PERSON-CROP | no |
| child_head_impact | safety | point | A | macro | rapid | WIDE | no |
| child_wandering | safety | point | A | macro | slow | WIDE | no |
| child_device_usage | routine | state | A | micro | slow | PERSON-CROP | no |
| calling_for_help | safety | point | A | — | — | AUDIO | n/a |
| choking | safety | point | A | macro | slow | **WIDE** | no |
| meal_eaten | routine | point | A | micro | slow | PERSON-CROP | no |
| stove_left_on | safety | state | P | micro | — | STILL | no |
| emergency | safety | point | A | macro | rapid | WIDE | no |
| possible_intruder | safety | point | P | micro | — | STILL | no |
| env_hazard_glass | safety | state | P | micro | — | STILL | no |
| env_hazard_water | safety | state | P | micro | — | STILL | no |
| fire_smoke | safety | point | P | macro | — | STILL / WIDE | no |
| setup_test | routine | point | P | macro | — | STILL / WIDE | no |

7 presence, 17 action. A custom detector ("collar-tug" → action/micro/slow →
PERSON-CROP; "walker present" → presence/micro → STILL) inherits routing with no new code.

---

## 9. Validation status & next experiments

**Proven:** WIDE (macro) and AUDIO — prod track record. **PERSON-CROP** on medication —
recall + precision (this doc's experiments).

**Owed — two tracks:**

*Classification (the meta-test — the taxonomy is only as good as its assignment):*
1. Audit the **24 existing** detectors' property/bucket assignments (§8).
2. Test the **compiler agent** end-to-end on custom detectors: does it infer the 6
   properties correctly AND emit sound reasoning + a gesture-reading prompt?

*Perception:*
3. ~~PERSON-CROP generalization~~ — **done** (collar-tug 98%; wearing_ring = floor).
4. ~~RAPID-MOTION~~ — **done** (convulsion +78% discrimination via overlay-only 8fps
   flash; shivering below sensor floor; §4).
5. **STILL-presence FP test** — the last unvalidated bucket. Empty rooms, pets,
   photos-on-walls, TV screens vs person-present; stills precision is suspect after
   the ring result. Existing footage, no recording.
6. **FACE-CROP** — lip-smacking automatisms + facial confounders (talking, chewing,
   yawning). One short recording session.
7. **STILL / object-presence — blocked.** Micro-objects need an object *crop*, which
   needs object *localization* we don't have (YOLO is **person-only**). The container
   test (0/18, full-frame) is the symptom. Prerequisite: open-vocab detection
   (YOLO-World / GroundingDINO) or a wide-VLM-returns-box pass. Until then only macro
   presence (fire, masked face) works.
   *(Speech-content probing DROPPED — decision 2026-07-12: the VLM handles audio in
   the same call; word-finding/confusion evidence rides the existing audio path.)*

**De-risk before building** (cheap, targeted): (a) object localization for STILL — can we
even crop an object; (b) does YOLO find a *fallen* person (else crop + attribution break
during a fall); (c) ~~does RAPID fix seizure~~ — yes; ship the RAPID recipe to prod's
seizure detector (flash tier + overlay-only + 8fps + describe-then-decide prompt).

**Dropped from the queue (2026-07-13):** macro-dilution rerun on agitation positives
(WIDE already carries a prod track record) and k-of-N efficacy testing (strictly
non-harmful for non-time-sensitive detectors — apply it, don't test it).

**Test thoroughness bar (2026-07-13):** remaining validations are NOT minimum-viable —
each detector gets ~6–10 TP events AND ~6–10 confounder/FP takes across ≥2 rooms and
positions before the taxonomy is considered confident.

**Gate to implementation (ordered, 2026-07-13):**
1. ~~FACE-CROP session~~ — **done (2026-07-13, 10 events × 2 arms × N=5, flash).**
   Results: `facial_grimacing` (PAINAD pain face) **60% recall / 9% FP (+51%) on
   CLEAN face crop — viable; overlay destroys it (0% recall)** → overlay is OFF
   for all face-scale detectors. `lip_smacking` is **distance-gated**: 0–1/5 at a
   200px face crop, 4–5/5 at 298px — viable only with face ≥ ~300px (near-camera;
   a placement/compiler constraint, not a prompt problem); talking/drinking/
   laughing confounders held at 0 FP, residual FPs were cross-fires between acted
   facial-repetition takes. `eye_twitch` **0/30 including exaggerated takes →
   REFUSE LIST, measured** (joins fine tremor / fingertip manipulation).
   **Product decision (Aryaman, 2026-07-13): the fine-motion / small-object
   detector CLASS carries a PERPETUAL degradation warning in the app** ("accuracy
   degrades with distance for this detector type"), and the compiler must attach
   a **clear-face + distance warning notice** to any custom detector it routes
   into a micro bucket — reasoning about the class at compile time, warning the
   caregiver permanently in the UI.
   **Tighter cropping is NOT a lever below the budget (measured 2026-07-13):**
   re-running lip_1 at an aggressive 1.2× face crop (120px vs 200px) stayed 0/5 —
   a crop already below the model's per-frame effective resolution represents
   every source pixel; shrinking it further adds nothing (the face is ~50 source
   px either way). Crop only pays when the region would otherwise be downsampled.
   **HARD-REJECT class at custom creation (Aryaman, 2026-07-13):** signals that
   are sub-Nyquist or sub-resolution get auto-rejected by the compiler with an
   explanation, no probing: fine high-frequency motion (eye flutter, shivering,
   rest tremor) and static objects at ring/pill scale. The interaction reroute
   applies ONLY when the interaction has ARM-SCALE motion (wristwear-donning:
   viable 30/30; ring-donning: fingertip-scale, measured below floor — motion
   alone doesn't clear the gate, amplitude does).
2. ~~Micro-body session~~ — **done (2026-07-13, general_fidgeting, 12 events ×
   2 arms × N=5).** Headline: P4 43%R/17%FP (+27%), P8 50%R/20%FP (+30%) — but
   the aggregate hides the real structure:
   - **Close range works** (wringing 4–5/5, tapping 4/5 at 8fps; typing/phone/
     gestures/adjust-once all correctly rejected 0–1/5).
   - **The "purpose" discriminator needs the OBJECT to be visible, so distance
     breaks it BOTH ways**: far wringing excused as "phone use" (0/5 miss); far
     cord-fiddling condemned as "purposeless rubbing" (10/10 — the only FP
     source). Purpose-judgment at distance is a guess.
   - **8fps is load-bearing for fine tapping** (close tap 1/5 @4fps → 4/5 @8fps)
     — fidgeting sits on the RAPID side; route to 8fps flash.
   - **Seated heel-tapping is below the clean-crop floor** (0/5 close, both fps).
   - **Overlay does NOT rescue fidgeting (O8 arm measured: 47%R/23%FP, +23% —
     WORSE than clean P8's +30%).** Heels stayed 0–1/5 tinted; the cord FP stayed
     5/5 (purpose was never a motion question — the tint can't show what the
     hands act on); typing gained FPs (tinted busy hands read as restless).
   Verdict: general_fidgeting ships CLEAN person-crop 8fps flash, distance-gated
   with the perpetual micro-class degradation warning; heel-scale foot motion
   joins the refuse list.
3. **Final validation pass** — one end-to-end review of the taxonomy, architecture,
   and compile→run flow against ALL measured results before any code.
Then implementation: build one reference detector per (perception shape × evidence
unit × eventing shape) combination that has demand, each certified by the harness
pattern (2–3 acted TP takes + nearest confounders, N=5, discrimination score), then
a **24-hour in-home soak** measuring real TP/FP per detector before any
family-facing rollout.

---

## 10. Build plan (local branch, 1–2 day soak)

1. Add the 6 core properties to `CatalogEntry`; derive the rest.
2. Bucket router in the tick worker — group active detectors, one VLM call per bucket.
3. STILL, PERSON-CROP (reuse the tracking crop), RAPID (overlay-first) passes.
4. describe-then-decide per micro/action bucket.
5. Eventing layer: k-of-N confirm, dedup, persistence debounce, absence rules, rate-limit.
6. Soak against live cameras; compare event stream to prod.

Deployed code → branch + PR, never straight to main. Work lives on
`feat/gateway-camera-ownership`.
