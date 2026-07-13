# Detection v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Full technical spec: `docs/architecture/detection-architecture-v2.md`. Evidence base: `docs/architecture/detection-inference-architecture.md` §4.

**Goal:** Replace the single-call, everything-every-tick detection pipeline with the validated v2 architecture — per-detector subject scoping, gated activation, seven locked perception buckets, two eventing primitives — at roughly today's cost, starting with the live seizure-detector fix.

**Architecture:** Six-dimension detector model (semantics / subject / activation / perception / decision / delivery) executed by a five-layer tick runtime (signals → scheduler → packer → decision → delivery). Shipped in seven phases, each an independently deployable branch + PR with its own tests.

**Tech Stack:** Python/FastAPI, SQLAlchemy + Supabase Postgres, LiveKit worker, Gemini (flash-lite + flash tiers) with OpenAI/OpenRouter fallback, pytest (`asyncio_mode=auto`), ffmpeg/OpenCV.

---

# Part 1 — The architecture, in plain English

*(This section is the human-readable overview. Skip to Part 2 for the build plan.)*

## Every detector answers six questions

Today a "detector" is a prompt fragment plus a threshold, and every enabled detector rides one big model call every 10 seconds. In v2, a detector is a complete declaration:

1. **What is it looking for?** — name, description, the prompt that steers precision vs recall. (Unchanged from today.)
2. **Who is it about?** — `patient`, `anyone`, or `environment`. A fall matters no matter who falls. Medication tracking is about the patient only. A stove doesn't need a person at all.
3. **When does it look?** — which cheap signals must be true before we spend a model call: someone present (and who), ONE motion signal (its amount gates every action bucket — RAPID included, just with its own person-scoped activity floor; the churn readout is computed alongside and logged as a tuning dial, not a gate), a big-enough face, a schedule window. All arithmetic on data the tick already has (tracker boxes, the InsightFace face box, one frame-diff pass) — the new part is only that detectors consult them before spending a call. Plus how often (every eligible tick, or sampled; snapshots simply ride motion ticks — they're too cheap to gate further).
4. **How does it look?** — one of six fixed "perception buckets": **SCENE** (whole room), **PERSON** (body-scale actions), **FINE-MOTION** (small repetitive movement), **RAPID** (convulsive shaking), **FACE** (facial actions), **SNAPSHOT** (one-frame object checks). Audio isn't a bucket — sound already rides every native-video call; a detector whose evidence is sound just carries `needs_audio`, which forces it into a call that actually has audio. Each bucket's frame rate / cropping / model tier was locked by measurement. Nobody — not caregivers, not the compiler — tunes these knobs per detector.
5. **When does it speak?** — instantly (`point`, optionally double-checked before alerting), or as a `state` with two independent dials: **entry** (on-sight for states true in one look — smoking, visitor present; a window of "M positive observations within T minutes" only when persistence is part of the claim's definition — watching TV, agitation) and **re-alert gap** (the existing episode hysteresis — one event per episode, never spam, regardless of entry speed). Absence is the same primitive — "no meal by 2 PM" is a count of zero at a deadline.
6. **How does it reach you?** — the delivery machinery that already exists today (push, per-patient email gating, clip archival, privacy suppression), unchanged; plus exactly one new marker: the permanent distance-degradation notice for the fine-motion/small-object detector class.

## A day in the life

- **Patient asleep, 2 AM.** Nobody is moving, so no video calls happen at all — any state detector that counts no-motion evidence (if the caregiver has one enabled; none ships by default) accrues for free from the absence of motion. The moment there IS motion (a fall out of bed, a convulsion, wandering), the gates open instantly and the right buckets run. Today, by contrast, the bedroom camera bills a full model call every 10 seconds all night.
- **A visitor is in the room, patient napping elsewhere.** Identity says "verified other person." Only detectors explicitly about others or the environment (intruder, fire/smoke, hazards) stay armed; everything else — including fall and choking — is about the patient and closes, exactly like today's whole-call skip but scoped per detector. When identity is merely *unsure* (face hidden, no enrollment), patient detectors keep running — suppression requires verified evidence, never uncertainty.
- **Possible seizure.** Rapid-motion energy on the person's body trips a pre-gate; a dedicated 8 fps motion-overlay crop goes to the stronger model with an amplitude-forced prompt. Measured: 8/10 recall, 1/30 false positives — versus today's prompt that literally tells a motion-blind model to lean TRUE on any red patch.
- **A caregiver writes a custom detector.** The compiler classifies it into the six dimensions, hard-rejects the physically impossible (eye flutter, fine tremor, ring-sized objects — measured zeros), and attaches a distance warning if it's in the degradation class.

## Why cost stays flat while accuracy jumps

The stronger model tiers and extra calls are paid for by not looking when there's nothing to see: person-scoped motion gating (sleep and empty rooms become ~free), a cheap rapid-energy pre-gate in front of the expensive seizure bucket, and sampling for state detectors (a 10-minute window doesn't need all 60 ticks). Estimated ~$250–310/mo per 2-camera household vs ~$275 today — same money, working detections. A per-household token ledger enforces this, degrading routine sampling first and never touching safety detectors.

---

# Part 2 — Build plan

**Standing rules for every phase:** work on a feature branch, PR to main, never straight to main. DB delta SQL applied to prod BEFORE backend deploy. Run the full test suite before each commit; commit at every green step; remove stale tests in the same PR that obsoletes them. New behavior ships behind an env flag defaulting OFF until soak-verified. **After every task commit: an adjudicated review checkpoint** — independent finder briefed on intent + a full-context pass, over five lenses: correctness, efficiency (hot-path cost), memory, security, and goal adherence to the v2 design + Aryaman's locked decisions; findings are fixed, refuted with a stated reason, or escalated. At phase end: full-diff review + entire suite + exit criteria before the PR.

**Phase granularity note:** P0 is specified to executable task detail below. P1–P2 are specified to task level. P3–P6 are milestone-scoped — each gets its own detailed TDD plan document at phase start (their internals depend on what P0–P2 actually look like when landed).

## Phase 0 — Seizure hotfix (RAPID recipe, solo call)  *(~2–3 days)*

The single worst live accuracy problem, fixed with the validated recipe, minimal blast radius. Branch: `feat/detection-v2-p0-seizure-rapid`.

**Design:** when the seizure detector is assigned and a crude rapid-energy pre-gate trips, run a **second, dedicated VLM call**: person-cropped, motion-overlay-only, 8 fps, `gemini-3-flash-preview`, with the neutral amplitude-forced prompt (ported from the validated experiment spec: any body position, "red tint alone is NEVER sufficient", stillness + fine trembling = shivering, not seizure). Seizure is removed from the main lite call while the flag is on. Everything downstream (promotion at 0.65, severity, push, clips) is unchanged.

**Key mechanical prerequisite:** `FrameBuffer` currently downsamples to 4 fps at push (`frame_buffer.py:77-101`). Change: buffer retains 8 fps; the main-call encode path decimates to 4 fps by dropping alternate frames (overlay for RAPID is computed on native 8 fps frames BEFORE cropping — tint→crop order is load-bearing). SCENE token cost stays identical.

**Tasks (TDD, red→green→commit each):**

1. **`FrameBuffer` at 8 fps + decimation helper.**
   - Test (`test/unit/test_frame_buffer.py`): buffer accepts 8 fps; new `decimate_to(frames, target_fps)` returns exact alternate-frame subset; window snapshot at both rates.
   - Implement: `livekit_worker_buffer_fps=8` setting; `frame_buffer.py` min-gap from buffer fps; decimation in the main-call encode path (`camera_ingest.py:494-500`).
2. **Rapid-energy pre-gate (embryonic Layer-1 signal).**
   - Test (new `test/unit/test_rapid_energy.py`): synthetic frame sequences — large rhythmic in-box deltas score high; slow drift and static scenes score ~0; TV-like small-region flicker scores low.
   - Implement: `src/services/detection/signals/rapid_energy.py` — reuse `motion_overlay.py`'s diff machinery (blur → absdiff → threshold) reduced to a scalar energy score over the person region (whole-frame fallback when no boxes); threshold in settings.
3. **RAPID artifact builder.**
   - Test (new `test/unit/test_rapid_artifact.py`, skipif no ffmpeg): given 8 fps frames + person boxes → output MP4 is overlay-tinted on NATIVE frames then cropped (assert via probe pixels: tinted region survives crop at correct offset); one artifact per tracked person.
   - Implement: `src/services/detection/rapid_artifact.py` — person boxes from the Re-ID tick result when available (`IdentityResult.per_frame_bboxes`), else a one-shot `PersonDetector`+`PersonTracker` pass over the window (both already in `src/services/reid/`).
4. **RAPID call path with per-call model override.**
   - Test (`test/unit/test_prompt_builder.py` extension + new integration test): the RAPID call contains exactly [motion-description context + seizure detector] and targets `gemini-3-flash-preview`; the main call's detector list excludes seizure when the flag is on.
   - Implement: `model_override` parameter through `run_vlm` → `gemini_provider`; `NEMARA_RAPID_SEIZURE_ENABLED` (default False) branching in `_run_tick_inner`; new seizure prompt constant beside the old one in `catalog.py` (old one deleted when flag defaults True).
5. **End-to-end through a fake VLM.**
   - Build the missing seam FIRST: `test/fakes/fake_vlm.py` — a scripted provider registered into `vlm_routing` returning canned `detector_fields` per call; this is permanent infrastructure every later phase reuses.
   - Test (new `test/integration/test_rapid_seizure_path.py`): tick with pre-gate tripped → two calls issued (main without seizure, RAPID with it) → seizure verdict promotes through existing path → event row; pre-gate quiet → RAPID call never issued.
6. **Soak prep:** wire per-call token usage logging (input/output counts per bucket) into the analyses row — the cost ledger's first data.

**Exit criteria:** all tests green; flag-off behavior byte-identical to today (regression suite proves it); flag-on validated against the labeled eval-set clips (replay harness, expect ≥8/10 convulsion recall, ≤1/30 FP); deployed to the home for a 24 h flag-on observation before defaulting on.

## Phase 1 — Detector schema v2 + registry  *(~2 days)*

Branch: `feat/detection-v2-p1-schema`. Adds the six-dimension columns; nothing reads them yet except severity (pure-parity change), so risk is near-zero.

1. Delta SQL `scripts/apply_detector_v2_delta.sql` (idempotent, per convention): `subject_scope`, `evidence_branch` (environment|person_motion|person_still), `bucket`, `needs_audio`, `evidence_unit`, `eventing` JSONB, `duty_cycle`, `severity`, `cooldown_seconds`, `degradation_warning`; `shape` gains `point` value (backfill `point_in_time`→`point` deferred to P4 when `scheduled` dissolves).
2. ORM + Pydantic mirrors; `CatalogEntry` gains the same fields; backfill all 24 built-ins per the §3 mapping table in the v2 doc; catalog validation extended (every entry must declare bucket/subject/eventing).
3. Severity moves from `catalog._SEVERITY` onto rows; `event_promotion._select_primary` reads the row. Test: promotion parity fixture — identical inputs produce identical primary selection before/after.
4. Onboarding seeding writes the new fields; `from-library` path too.

## Phase 2 — Layer-0/1 signals  *(~4–5 days)*

**Status (2026-07-13): BUILT** — landed on `feat/gateway-camera-ownership` (continuing the active branch per Aryaman, not the planned p2 branch name). Commits: d285dfe (`is_builtin` dropped — Aryaman-directed cleanup; its drop delta runs AFTER backend deploy, inverse of DB-first, see script header) · e48abc5 + c4685e9 (item 1 + review fixes) · b84d61d (item 2) · 2385f5b (item 3 — built as `signals/tick_signals.py`, one full-frame diff map at 320 px; the P0 rapid gate keeps its own eval-tuned crop-first measurement until the soak proves the map can replace it) · 942e5fa (item 4 — plus the two prescribed behavior changes: closed-state backend wake and presence union; new dial `signals_motion_min_active_fraction=0.02`). Deviation from item 3's letter: signal unit tests use synthetic windows, not recorded fixtures. **Wake backoff REJECTED (Aryaman 2026-07-13):** a closed camera that wakes on motion must never wake *less often* because past wakes found nobody — that is a deliberate blind window, and we cannot assume nothing happens during it. Sustained-false-wake cost (TV/fan in a closed room, ~1 LLM call per tick until P3) is handled ONLY by non-blinding levers: the amplitude floor dial (soak-tuned) and the P3 scheduler, under which a no-person wake resolves tracker-only with zero VLM spend (env branch runs only if armed, at snapshot cost). Do not re-propose backoff.

Branch: `feat/detection-v2-p2-signals`. Depends on the decided re-ID direction (tracking always-on, FaceID attribution optional — see memory/branch notes).

1. Decouple body tracking from `reid_enabled`: the early-out at `inference.py:232-243` becomes attribution-only; YOLO+BoT-SORT runs every presence tick.
2. `IdentityResult` gains the per-track list `[{track_id, label, bbox_series, confidence}]`; aggregate fields kept for compatibility.
3. Backend signals module: `person_present`, `patient_present`, the unified `motion` signal per-track (amount + churn concentration; the P0 rapid_energy module graduates to per-track scoping), `face_valid(min_px)` — all from tracker output + one frame-diff pass, unit-tested on recorded fixture windows. Phone/gateway SAD packets demoted to a hint (both camera types publish them today).
4. Signals logged per tick into analyses for observability; no behavior change yet (scheduler still P3). Camera open/close state machine now fed by backend presence.

## Phase 3 — Scheduler + packer  *(milestone; detailed plan at start)*

Activation resolution as a pure function (the §1.2 subject truth table + gates + schedule + duty cycle — exhaustively table-tested), ≤3-detectors-per-call packing with the shared context slot, per-bucket artifact builders (SCENE/PERSON/FINE-MOTION/FACE/SNAPSHOT zones), per-bucket tier routing, provider-capability constraints (audio and 8 fps require native-video providers). The old single-call assembly and the binary identity gate are deleted here. This is the largest phase and the point where the §1.2 visitor behavior goes live.

## Phase 4 — Decision layer  *(milestone; detailed plan at start)*

The windowed-count state primitive (≥M evidence ticks in window T over `analyses`; `episode_gap_seconds` becomes exit hysteresis), absence deadlines (dissolving the `scheduled` shape into schedule-window + absence-state), k-of-N confirm for configured point detectors, per-detector `cooldown_seconds` (fixes point detectors re-firing every 30 s during persistent conditions), cross-call primary selection per tick. Replaces `test_state_episode_dedup` / `test_detector_inference_gap` with state-primitive tests.

## Phase 5 — Compiler v2  *(milestone; detailed plan at start)*

Dimension classification, the feasibility hard-reject list with plain-language reasons, SNAPSHOT admission criteria + plate-size prior, interaction reroute (arm-scale amplitude rule), degradation warnings at creation. The compiler currently has zero tests — this phase builds its golden-suite from scratch (reject-list table tests + classification goldens). Also (P2 review carry-over): a **reserved-field-prefix check** at compile time — custom detectors must not declare schema fields named `sig_*`, `motion_src`, `people_present_src`, `identity_*`, `people_count`, `woke_by_backend`, `wake_env_motion`, `vlm_*_tokens`, `rapid_*` (backend-stamped keys silently win today; make the collision a loud compile reject instead).

## Phase 6 — Reference detectors + soak  *(milestone)*

One reference detector per bucket certified via the harness pattern (acted TP takes + nearest confounders, N=5, discrimination score), the cost ledger + budget governor live, then the **24-hour in-home soak** measuring real TP/FP per detector and $/day against the §6 model. Exit = go/no-go per detector + the degradation-warning copy for the apps.

---

# Part 3 — Test infrastructure (built in P0–P1, grown every phase)

- **`test/fakes/fake_vlm.py`** — scripted provider (the seam that doesn't exist today; current tests inject dicts below `run_vlm` and nothing drives a tick end-to-end).
- **`test/fakes/fake_livekit.py`** — consolidates the three per-file `_FakeRoom*` copies (`test_room_reconciler`, `test_livekit_privacy`, `test_room_worker_token_rotation`).
- **Full-tick harness** — drive `CameraIngest._run_tick` with fixture frames + fake VLM + real DB; the backbone of every phase's integration tests.
- **Eval-set replay** — the labeled home clips (`~/Desktop/eval-set/`) replayed through artifact builders + real model for recipe regression (manual/CI-optional job, not per-commit).
- **Stale-test watchlist** (flip intentionally, remove in the phase that obsoletes them): one-call-per-tick assumptions, `people_present`-authoritative guards in `test_parse_validate`, XOR guards in `test_camera_schema_source_type`.

# Part 4 — Risks & dependencies

- **P2 CPU budget:** always-on YOLO+BoT-SORT per presence tick on the Railway worker — profile early; mitigation is decimated tracking fps, which the signals tolerate.
- **8 fps buffer memory:** doubles FrameBuffer frames (~17 s × 8 fps × N cameras) — measure; I420 frames are the dominant term.
- **Provider fallback is allowed everywhere** — but a fallback that loses the signal (audio detectors on frame-sampling providers get no audio; RAPID on a frame-sampler loses the validated recipe) must mark the tick as degraded in the analyses row, so a silent detector isn't mistaken for a healthy one. One log line + one field, not machinery.
- **Motion moves backend-side in P2** for person-scoping only (a TV must not hold the gate open). Both camera types already publish motion packets today (wall-phone app and gateway publisher.py alike); nothing about gateway behavior changes.
- **Uncommitted working tree** (`feat/gateway-camera-ownership` carries experiment + doc changes, incl. a capture hook marked "remove before PR") must be untangled before P0 branches.
