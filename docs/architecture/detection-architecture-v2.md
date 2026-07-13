# Detection Architecture v2 — the generalized detector model

Status: **design locked pending Aryaman's review** · 2026-07-13
Supersedes: `detection-inference-architecture.md` (kept as the experimental-evidence record; its §4 measurements are the proof base for every recipe below).

This document generalizes everything measured in the June–July 2026 validation program into one hierarchical detector model, maps all 24 built-ins plus the newly validated detectors into it, states exactly what is ported / changed / removed from today's code (with file:line anchors), and ends with the migration and test strategy.

---

## 0. The laws this design is built on (measured, not assumed)

1. **Token budget is scarce; cropping points it.** Video ≈ 221 tok/frame at `high` regardless of source resolution; a still image ≈ 1,100 tok. Cropping is token-neutral and buys effective resolution — but is useless once the subject is below the per-frame budget (tight-crop probe: no change).
2. **Model tier gates rapid-motion perception.** Lite is blind to 8 fps rapid rhythm under any packaging; flash + overlay-only + 8 fps reads it (convulsion 8/10 R, 1/30 FP). Tier does NOT fix stills — the image encoder budget is identical across tiers.
3. **Overlay helps exactly one thing:** body-scale rapid rhythm. It hurts faces (grimace 0/5), hurts purpose discrimination (fidget O8 < clean P8), and adds FPs (typing). Overlay is a per-bucket recipe property, never a global flag.
4. **≤ 3 detectors per call.** Detector dilution measured at 92/90/72% — packing is a hard cap, not a style choice.
5. **Prompts steer error direction, not information.** (ring: 83%R/87%FP → 0%FP/7%R on prompt change alone). Prompt bias is the precision/recall dial; perception recipe is the information dial. Never fix an information problem with a prompt.
6. **Physics refuses some detectors.** Sub-Nyquist (fine tremor/shivering at 8 fps), sub-resolution (eye flutter 0/30, ring-scale objects, fingertip manipulation), and seated heel-taps are measured zeros. These are compile-time hard rejects, not tunables.
7. **Distance degrades a known class.** Face-scale detectors need ≥ ~300 px faces; object stills below plate size fail past ~7 m; purpose discrimination needs the held object visible. This class ships with a perpetual in-app degradation notice.

---

## 1. What a detector IS — six dimension groups

A detector is a point in a six-dimensional space. Today's `Detector` row (src/db/detector.py:25) covers group 1 and fragments of 3, 5, 6; groups 2 and 4 do not exist in code at all.

```
Detector =
  1. SEMANTICS    what it means        (identifier, name, prompt_fragment, category, cohort)
  2. SUBJECT      who it is about      (patient | anyone | environment)
  3. ACTIVATION   when it runs         (gate expr × schedule × placement × duty_cycle)
  4. PERCEPTION   how it looks         (bucket → locked recipe: fps, channel, crop, tier)
  5. DECISION     how evidence→event   (evidence unit × eventing shape × threshold/severity)
  6. DELIVERY     what the user gets   (notify, clip, privacy, warnings)
```

The hierarchy matters: **subject and activation decide whether we look at all; perception decides how we look; decision decides when we speak; delivery decides how we say it.** Cost falls out of activation × perception. Accuracy falls out of perception × decision. Nothing else in the system gets to influence those outcomes.

**The organizing principle above all of it (Aryaman, 2026-07-13): detectors are NOT organized by instrument — they're organized by which EVIDENCE BRANCH of the scene feeds them.** A tick classifies the scene into three branches, per camera and per person:

```
ENVIRONMENT      evidence exists without any person: stove, door, fire,
                 hazards. Evaluated every tick regardless of occupancy —
                 driven by environment motion (change OUTSIDE person
                 boxes) + schedule.
PERSON-MOTION    a tracked person's in-box activity ≥ the floor: falls,
                 seizure, wandering, medication, eating, fidgeting —
                 anything evidenced by action. Per track, per subject.
PERSON-STILL     a tracked person below the floor: facial actions (a still
                 person can lip-smack), sleep/slump states, and the "still"
                 evidence that duration states count.
```

The six buckets are the *instruments* each branch points at the scene; `needs_audio` is a property of the call, and the decision shapes (point/state) consume whatever the branches produce. Everything in §1.3-§1.5 hangs off this tree.

**Intruder is NOT an identity inference (Aryaman, 2026-07-13):** only the patient is face-enrolled, so every visitor/relative/aide is "unknown" — an unknown track can never be the trigger. `possible_intruder` is video-evidenced concealment (mask/covered face — exactly what its catalog prompt already says, catalog.py:1026-1032), living on the person branches with subject=anyone; attribution state is at most CONTEXT passed to the model, never the verdict.

### 1.1 Semantics (ported unchanged)

`identifier`, `name`, `icon_name`, `description`, `category` (safety|routine), `cohort`, `prompt_fragment`, `seeded/hidden/fire_once/recommended`, `custom_prompt`. All exist today (`is_builtin` was dropped in P2 — redundant with `source='preset'`) (catalog.py:812-853, detector.py:25-86). The prompt_fragment remains the precision/recall dial (Law 5) — compile per stakes as today (detector_compiler.py:124-195).

### 1.2 Subject — NEW dimension (answers "visitor is present — what runs?")

```
subject_scope: patient | anyone | environment
```

While attribution is unresolved the rule is FIXED, not per-detector
configuration (unknown_policy column cut, Aryaman 2026-07-13): run —
suppress only on VERIFIED not-patient evidence, never on uncertainty.

Attribution comes from the existing per-track PATIENT/OTHER/UNKNOWN machine (body_pipeline.py:35-37, face-lock after 2 hits :364-377, OTHER after 5 verified failures :383-406, never demote PATIENT :391-397). What changes is **who consumes it**: today it feeds one binary whole-call gate (camera_ingest.py:572-620 — patient absent + only-verified-others ⇒ skip the entire LLM call). In v2 the same signal filters **per detector**:

| Scene attribution state | environment | anyone (safety) | patient (clinical/routine) |
|---|---|---|---|
| Patient present (locked) | ✅ | ✅ | ✅ |
| Unknown person, no enrollment | ✅ | ✅ | ✅ (identity-agnostic fallback — today's behavior) |
| Unknown person, enrollment exists, unresolved | ✅ | ✅ | ✅ (suppress only on positive evidence) |
| **Verified other(s) only** | ✅ | ✅ | ❌ |
| No person | ✅ (+ absence states keep counting) | ❌ | ❌ |

Scope assignment is deliberately simple (Aryaman, 2026-07-13): **almost everything is `patient`** — including fall, choking, seizure, emergency; the patient is who this product watches. `anyone` is reserved for detectors explicitly about other people (possible_intruder); `environment` for no-person detectors (fire/smoke, hazards, stove). Net behavioral delta vs today's whole-call skip: with a verified other alone in frame, intruder + environment detectors now keep running — nothing else changes. The suppression philosophy is unchanged from today's gate — suppress only on *verified* not-patient evidence, never on mere uncertainty.

**Required plumbing change:** `IdentityResult` is aggregate-only today (inference.py:70-181 — one patient bbox, peak count, only-others flag; no track list). v2 exposes the per-track list `[{track_id, bbox_series, label, confidence}]` to the ingest so the scheduler can target crops and scope subjects per track. Body tracking must also run **always-on**, decoupled from `reid_enabled` (today `reid_enabled=false` kills ALL tracking — inference.py:232-233; the decided direction: tracking always on, FaceID attribution optional on top).

### 1.3 Activation — gates, schedule, placement, duty cycle

**Gate lattice (Layer-1 signals, all cheap, no VLM):**

| Signal | Source | Today | v2 |
|---|---|---|---|
| `person_present` | tracker (+LLM people_present as confirmation) | camera state open/closed (state_machine.py) | per-track, backend-owned |
| `patient_present` | attribution machine | aggregate patient_in_clip | per-track label |
| `motion` — ONE signal, TWO readouts (amount + churn concentration) | one frame-diff pass over track boxes (the rapid_energy module already computes both) | **whole-frame phone SAD via data packet** (motion_cache.py:63-82, 15 s TTL) — gates only the `closed` state (camera_ingest.py:449) | the AMOUNT readout gates every action bucket including RAPID (RAPID just uses a person-scoped activity floor — general motion, nothing rapid-specific; Aryaman 2026-07-13, backed by the eval probe: churn concentration overlapped too much between convulsions and vigorous negatives to earn a default floor, and any extra floor risks blocking a real seizure). Churn stays computed + logged as a soak-tuning dial. SNAPSHOT checks simply ride motion ticks — too cheap (~$0.0001) to warrant their own gate |
| `env_motion` | change OUTSIDE all person boxes (same diff pass) | — (whole-frame SAD conflates person and environment) | drives the ENVIRONMENT branch: snapshot checks, new-track/entry attention — so a door opening or a burner lighting is never masked by a still person |
| `face_valid(min_px)` | InsightFace box size | — | gates FACE bucket; also drives the distance warning |
| `no_person` | tracker | close_threshold streak | absence states count on it |

**Per-track semantics (multi-person):** every signal above is evaluated PER TRACK where it can be — each person carries their own motion amount, attribution, and face size; action buckets open for the tracks that are moving and target their crops; the subject filter applies per track. Scene-level lanes (SCENE instrument, environment checks) open if ANY qualifying track or env_motion demands them.

The whole-frame phone SAD stays as a hint/fallback, but motion becomes **backend-owned and person-scoped** — this also fixes the current fragility that gateway (Tapo) cameras depend on a client-published motion packet, and that a TV or curtain holds the frame "in motion."

**Critical inversion vs today:** today the `open` state runs the full call **every tick regardless of motion** (camera_ingest.py — motion only matters in `closed`). v2 runs action buckets only under `sustained_motion`. A sleeping patient generates near-zero VLM traffic while no-motion evidence accrues free for any state detector counting it (none ships by default — catalog stays at 24) — and any nocturnal motion (seizure, wandering, fall) opens the gates instantly. Presence-without-motion runs only sampled states and FACE (a still person can lip-smack).

**Schedule:** structured windows replace the free-text `when_scope` string ("8 AM, 8 PM ±1H" → `{windows: [...], tz: caregiver.timezone}`; schedule_config JSONB already exists, detector.py:71). Absence deadlines ("no meal by 2 PM") are the same state primitive (§1.5), not a special shape.

**Placement:** `cameras_detectors` assignment + per-camera `door_config` are ported as-is (camera_detector.py:20-39, registry.py:51-107). `where_scope` becomes structured room tags used for auto-assignment only.

**Duty cycle (NEW, the cost dial):** `every_eligible_tick` (safety points) | `sampled_1_of_k` (states — windowed counting tolerates sparse evidence by design) | `edge_only` (stills). Cost class is derived, never hand-set: bucket tier × duty cycle.

### 1.4 Perception — six locked buckets (+ a needs_audio flag)

A detector picks a **bucket**; every knob inside the bucket (fps, channel, crop, model tier, media res) is fixed by measurement and not exposed — not to caregivers, not to the compiler, not per-call.

| Bucket | Frames | Channel | Crop | Tier | Gate default | Validated by |
|---|---|---|---|---|---|---|
| **SCENE** | 4 fps video | clean | full frame | lite | sustained_motion | prod + collar/medication program |
| **PERSON** | 4 fps video | clean | person crop | lite | sustained_motion (patient track) | medication 100%R/0FP, collar 98% |
| **FINE-MOTION** | 8 fps video | clean | person crop | flash | sustained_motion | fidgeting +30% (clean P8 beat overlay) |
| **RAPID** | 8 fps video | **overlay-only** | person crop (tint→crop order) | flash | person motion ≥ activity floor | convulsion 8/10R 1/30FP (+78%) |
| **FACE** | 8 fps video | clean (overlay destroys it) | face crop, pad 2.0 | flash | presence + face_valid(≥300 px) | grimacing +51%; lip smacking distance-gated |
| **SNAPSHOT** | 1 frame | clean | **zone crops** | lite | rides motion ticks / scheduled | phone 10/10, remote FP 10→0, plate 9/10 |

**Audio is NOT a bucket** (Aryaman, 2026-07-13): the MP4 already carries the audio track on the native-video path — sound inherently rides whatever video call runs. A detector whose evidence is primarily sound (calling_for_help, verbal confusion cues) simply carries **`needs_audio: true`** — a packing constraint, not a perception shape: the packer must place it in a native-video call, and a fallback that loses audio marks the tick degraded (frame-sampling providers receive no audio today, gemini.py:6-8; fallback stays allowed, today's chain semantics keep).

Packaging laws baked into the artifact builders: one video per request (two parts silently collapse), never side-by-side (halves per-channel resolution), never temporal concat (re-injects resting prior; T8 7/10 < O8 10/10), tint on native pixels **before** cropping.

SNAPSHOT admission criteria (strict): one frame *settles* it (not merely suggests), the state isn't instantaneous, subject above the plate-size floor at expected distance, and it survives ambiguity confounders. "Watching TV" fails (it's an inference from gaze+audio+duration → FINE-MOTION/SCENE with needs_audio). Stills are **point-in-time object detection** — nothing more.

### 1.5 Decision — evidence units × two eventing primitives

**Evidence units:** `verdict` (bool + confidence per call — today's shape) · `instance` (a logged occurrence — confusion cues, isolated jerks; never a per-tick alert) · `sample` (a state's per-tick observation).

**Eventing shapes — exactly two:**

- **`point`** — fires on a qualifying verdict. Optional `confirm_k_of_n` re-check for high-severity/low-prior detectors (spend tokens on fires, which are rare — nearly free FP suppression). Today there is **no** confirmation anywhere; re-fire suppression is three-tier by shape: point = 30 s dedup only (event_promotion.py:128 — a persistent condition the model keeps reporting can re-fire ~every 30 s), state = per-detector rising-edge hysteresis (`episode_gap_seconds`, else 60 s safety / 300 s routine — :193-244, one event per episode), scheduled = fires only at its deadline via the scheduler, never reactively (:303-304).
- **`state`** — one windowed-count primitive replacing today's threshold+rising-edge state logic: **enter when ≥ M evidence ticks within window T** (edge-triggered), exit via hysteresis gap (today's `episode_gap_seconds` ports directly). Dense M ≈ duration semantics with free dissent tolerance ("motionless > 5 min"); sparse M ≈ accumulation ("≥ 4 confusion instances this afternoon"); **absence = count == 0 at deadline** ("no meal by 2 PM"). Today's third shape `scheduled` disappears — it was always a schedule window (activation) plus absence semantics (decision). **Entry is a per-detector dial, independent of anti-spam:** states that are TRUE ON SIGHT (smoking, visitor present, collar off) get M=1–2 — they fire within a tick or two; a real window applies only when persistence is part of the claim's own definition (watching TV, agitation, motionless-too-long). Spam prevention never depends on entry speed — the episode gap alone guarantees one event per episode (prod agitation already behaves exactly this way: instant entry + rising-edge suppression). Both dials come from a small preset menu inferred from the description and shown in plain words on the review screen.

Naming note: the duration shape keeps its existing name **state** (today's DB enum value ports as-is). Stills are point-in-time object detection — a perception-axis concept, unrelated to the state shape.

Kept from today's promotion machinery: sensitivity → threshold offsets (event_promotion.py:93-113), per-detector base thresholds, one-event-per-tick primary selection by severity→hint→confidence (:156-190) — extended to arbitrate **across the multiple calls of one tick**, since v2 packs ≤3 detectors per call. Changed: `severity` moves from the in-code map (catalog.py:1179-1208) onto the row; per-detector `cooldown_seconds` replaces the single global 30 s `_DEDUP_WINDOW`.

### 1.6 Delivery, honesty, economics

Ported: push/email fan-out, clip archival with privacy re-check, `clip_suppressed`, Realtime updates (camera_ingest.py:867-921, clip_archival.py). New per-detector fields:

- `degradation_warning: none | distance_class` — perpetual in-app notice for FACE, FINE-MOTION-with-object, and small-object SNAPSHOT detectors (Law 7). Set by compiler or catalog, surfaced by apps. (The only delivery addition; a shadow/certification tier was considered and cut as overkill — validation happens via the soak process, not a schema column.)
- `cost_class` (derived) — surfaced to the household budget governor (§7).

---

## 2. Runtime — five layers per camera tick

```
L0  SIGNALS     always-on tracking + attribution + motion/rapid/face/edge signals   (no VLM)
L1  SCHEDULER   per detector: subject × gates × schedule × duty-cycle → due set     (pure function)
L2  PACKER      due detectors → calls: group by (bucket, target track), ≤3/call,
                + always-on context slot (people_present / motion_description),
                build artifacts per recipe, route tier + provider (audio/8fps-aware)
L3  DECISION    verdicts/instances/samples → point (k-of-N) or state counters →
                cross-call primary selection → event row
L4  DELIVERY    push/email/clip/privacy + cost ledger
```

Topology unchanged: one worker per patient room, N `CameraIngest` per worker (room_worker.py:141), 10 s tick / 11 s window / FrameBuffer fork (settings.py:147-154). The tick engine keeps its skeleton; `_run_tick_inner`'s single-call body is replaced by L1→L3. The binary identity gate and the closed-state motion check dissolve into L1 (they become per-detector subject filtering and per-bucket gating). `ai_paused`/`disabled`/privacy short-circuits stay exactly where they are.

**One shared context call** classifies the inactivity family when relevant (sleeping vs sitting vs watching TV — an inference, not a still) so N states don't each buy their own context.

**Eventing is PATIENT-level, never camera-level (multi-camera correctness, Aryaman 2026-07-13):** state counters and absence deadlines live on the patient and pool evidence from EVERY camera in the household — a meal seen by the kitchen camera satisfies "no meal by 2 PM" no matter which camera's tick is evaluating; "motionless 30 min" cannot fire because one room is empty while the patient moves in another (no-person ticks contribute nothing — only patient-still evidence counts); absence fires only when NO camera produced evidence by the deadline. Today's code already points this way — episode suppression and event dedup both query by `patient_id` across cameras (event_promotion.py:210-244, :432-453) — v2 makes it the explicit contract for the windowed-count primitive. Camera assignment (`cameras_detectors`) scopes which cameras may CONTRIBUTE evidence to a detector, never where its counter lives.

---

## 3. The catalog mapped into v2 (all 24 + newly validated)

| Detector | Subject | Bucket | Eventing | Notes |
|---|---|---|---|---|
| fall | patient | SCENE | point | k=1 — recall-critical |
| **seizure** | patient | **RAPID** | point | **P0 hotfix**: today lite+global-overlay+biased prompt (catalog.py:269-316) — the measured worst case |
| emergency | patient | SCENE | point | |
| choking | patient | PERSON | point | hands-to-throat is body-scale |
| calling_for_help | patient | SCENE (needs_audio) | point | |
| fire_smoke | environment | SCENE | point | smoke is large-scale + dynamic |
| possible_intruder | anyone | SCENE | point | evidence = visible concealment (mask), per its existing prompt — never "unenrolled person" (only the patient is enrolled); attribution is context at most |
| wandering / child_wandering | patient | SCENE (door zones) | point | door_config ports as-is |
| stove_left_on | environment | SNAPSHOT (zone) | state absence-style | small-flame degradation warning |
| env_hazard_glass / _water | environment | SNAPSHOT (zone) | point | rides motion ticks; size-floor warnings |
| medication | patient | PERSON | point (absence state later) | schedule window; validated 100%R/0FP |
| meal_eaten | patient | PERSON | point | |
| agitation | patient | SCENE (needs_audio) | state (sparse M-of-T) | replaces rising-edge episode logic |
| climbing / distress (child) | patient | SCENE (needs_audio) | state (dense) | the child IS the patient |
| child_crib_exit / _stairs / _hazard_reach / _head_impact / _cord_entanglement | patient | SCENE (cord: PERSON) | point | cord = thin-object warning |
| child_device_usage | patient | PERSON | state (sparse) | routine |
| setup_test | anyone | SCENE | point, fire_once | unchanged |
| **general_fidgeting** (new) | patient | FINE-MOTION | state (sparse) | +30% validated; object-visibility warning |
| **facial_grimacing** (new) | patient | FACE | state (sparse) | +51% validated |
| **lip_smacking** (new) | patient | FACE | state (sparse) | distance-gated ≥300 px face; warning |
| **confusion / word-finding** (new) | patient | SCENE instances (needs_audio) | state (sparse over instances) | never per-tick alerts |
| **myoclonic jerks** (new) | patient | RAPID instances | state (sparse) | |
| *refused forever* | — | — | — | eye flutter, fine tremor/shivering, ring-scale objects, fingertip manipulation, seated heel-taps (Law 6) |

---

## 4. Port / change / kill (against today's code)

**PORT unchanged:** worker/room/ingest topology + FrameBuffer fork (room_worker.py:897-951); provider chains + per-caregiver `vlm_provider` (vlm_routing.py:36-79) with new bucket-capability constraints; promotion threshold/sensitivity machinery (event_promotion.py:93-113); primary-selection ranking (:156-190); event schema + push/email/clip/privacy (camera_ingest.py:867-921); `cameras_detectors` + door_config; seeding/library; compiler skeleton incl. stakes-based bias (detector_compiler.py:124-195); the attribution state machine (body_pipeline.py) and harvesting; data-collection recorder + privacy paths.

**CHANGE:**
- `detectors` table: + `subject_scope`, `evidence_branch` (environment|person_motion|person_still — the top-level taxonomy), `bucket`, `needs_audio`, `evidence_unit`, `eventing` JSONB (`{shape, k, n, M, T, absence_deadline}`), `duty_cycle`, `severity` (persisted), `cooldown_seconds`, `degradation_warning`. `shape` enum collapses `point_in_time|state|scheduled` → `point|state`. Idempotent delta SQL per convention.
- Identity: expose per-track list from ReID to ingest; run tracking always-on regardless of `reid_enabled` (inference.py:232-233 early-out becomes attribution-only).
- Motion: backend person-scoped signals from tracker; phone SAD demoted to hint.
- Call assembly: `build_prompt_and_schema` gains a packer in front (≤3/call, per-bucket artifact builders); `motion_overlay_enabled` global flag dies — overlay becomes a RAPID recipe property (camera_ingest.py:641-656 moves into the RAPID artifact builder).
- Seizure prompt rewritten neutral + amplitude-forced ("red tint alone is NEVER sufficient"; stillness+fine-tremble = shivering, not seizure).

**KILL:** all-detectors-one-call (prompt.py:101-183 as sole assembly); binary whole-call identity gate (camera_ingest.py:572-620); `scheduled` as a shape; free-text where/when scopes as runtime semantics; global overlay flag; open-state-runs-every-tick economics.

---

## 5. Compiler v2 (customs)

Pipeline: classify → **feasibility gate** → prompt gen → certify.

1. **Classify** (extends `infer_detector_config`): assign subject, bucket, evidence unit, eventing shape + params, gates/schedule — from the caregiver's plain English.
2. **Feasibility gate (new, hard):**
   - HARD REJECT (Law 6 list) with a plain-language reason — no probing, no maybes.
   - SNAPSHOT admission criteria; below plate-size default rule ("objects smaller than a plate can be hard to detect from some distance away" is baked into the classifier prior).
   - Interaction reroute: object too small for a still frame but donning/usage is arm-scale → recompile as motion detector (wristwear 30/30 vs ring-donning 7% — amplitude decides, and sub-arm-scale interaction is rejected too).
   - Distance class ⇒ attach `degradation_warning` + compiler notice at creation time.
3. **Prompt gen** under the bucket's recipe constraints, stakes-biased as today, OpenAI-first/Gemini-fallback unchanged.

---

## 6. Cost model (corrected 2026-07-13)

Per-call: SCENE/PERSON ≈ 13.2k tok lite ≈ $0.0014 · FINE-MOTION/RAPID/FACE ≈ 20.9k tok flash ≈ $0.0070 · SNAPSHOT ≈ 1.1k/zone lite ≈ $0.0001. Profile: single occupant in their 70s, 2 cameras, ~1,300 person-scoped motion ticks/day, ~4,000 presence ticks/day.

| Scenario | $/day | $/month |
|---|---|---|
| Today's system (presence-driven: open state bills every tick incl. sleep; one lite call, all detectors) | ~$9 | **~$275** — and it's the broken one |
| v2 naive (buckets ungated: RAPID+FINE-MOTION+FACE on every eligible tick, no sampling) | ~$33 | **~$1,000** — not shippable |
| **v2 with the controls designed in** (motion-gated SCENE ×2-3 packed calls, rapid_energy pre-gate ~15%, states sampled 1/3–1/6, k-of-N only on fires) | ~$8–10 | **~$250–310** |

The headline: **v2 costs the same as today while actually working** — the gates pay for the tier upgrades. The three load-bearing controls (rapid pre-gate, state sampling, motion-gated SCENE incl. night) are worth ~$700/mo/household; they are architecture, not tuning. Budget governor: per-household daily token ledger; degrade order = increase sampling interval on routine states → FACE → FINE-MOTION; **never** degrade safety points.

---

## 7. Migration & test strategy (branch + PR per phase)

Test infrastructure first (identified gaps): a **central fake VLM provider** (scripted `detector_fields` per call — today's tests inject dicts below `run_vlm`; nothing drives the tick end-to-end), a shared LiveKit fake module (today three files each define their own), and a full-tick integration test harness. These land in P1 and every phase adds to them.

- **P0 — Seizure hotfix** (immediate, small blast radius): seizure → RAPID recipe as a dedicated solo call behind a crude motion pre-check; neutral amplitude-forced prompt. Fixes the worst live FP/FN today. Tests: prompt snapshot, promotion path, FP regression fixtures from the eval set.
- **P1 — Schema v2 + registry**: new columns + delta SQL; backfill all 24 built-ins per §3 table; severity onto rows; promotion reads row severity. Tests: migration idempotency, catalog validation, promotion parity.
- **P2 — L0 signals**: always-on tracking decoupled from `reid_enabled` (existing decided branch direction); per-track surface into ingest; the unified motion signal per-track (amount + churn concentration) + face_valid; phone SAD demoted. Tests: signal unit tests on recorded fixtures.
- **P3 — Scheduler + packer**: activation resolution (subject table §1.2 as a pure function — heavily unit-tested truth table); ≤3 packing; per-bucket artifact builders (tint→crop order asserted); tier/provider routing with audio/8fps constraints. Tests: packer property tests, golden artifacts.
- **P4 — Decision layer**: state windowed-count primitive over `analyses` (replaces rising-edge for states; `episode_gap_seconds` ports as exit hysteresis); absence deadlines; k-of-N confirm; per-detector cooldown; cross-call primary selection. Remove/replace: `test_state_episode_dedup`, `test_detector_inference_gap` (become state-primitive tests).
- **P5 — Compiler v2**: feasibility gate + hard-reject list + warnings. Tests: reject-list table tests, classification goldens (compiler currently has zero tests).
- **P6 — Reference detectors + 24 h in-home soak** with the cost ledger running; soak measures TP/FP per bucket and $/day against §6.

Stale-test watchlist (will intentionally flip): `test_camera_schema_source_type` XOR guards, `test_parse_validate` people_present-not-authoritative guards, any test assuming one-call-per-tick.
