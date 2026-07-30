# Detection — as-built pipeline & detectors

What actually runs today (written 2026-07-23; reconciled to the phase C–H2 detector refactors 2026-07-27). Onboarding for detector accuracy work. Measurement history that justifies the recipes is in `detection-inference-architecture.md`.

## Status: detection is PAUSED (DB state as of 2026-07-23 — re-check before trusting)

- All detectors `is_on=false`; `NEMARA_REID_SOAK_ONLY=true` → the executor is skipped, **zero VLM calls**. Last real inference was 2026-07-20 (soak end).
- **Resume:** set `.env NEMARA_REID_SOAK_ONLY=false` + recreate the container (env is read at boot), then `UPDATE detectors SET is_on=true WHERE identifier <> 'setup_test';` (next tick). Leave `setup_test` off — fire-once self-test.

## The pipeline (one camera, every 10 s tick)

`RoomWorker` (per room) spawns a `CameraIngest` (per camera) that runs a fire-and-forget tick over an ~11 s clip. Each stage reads specific detector attributes (glossary below):

1. **L0 signals** (`signals/`, `services/identity/`) — always-on continuous tracking (SAM3 + EdgeTAM on the GPU pod, `gpu/perception/`) + the identity sidecar's verdicts consumed over `/state` (`identity_client.py`; the in-process YOLO+BoT-SORT brain is deleted — the sidecar is the only identity provider) → per-track `{bbox, label ∈ PATIENT|UNKNOWN}`; per-track motion + whole-frame `env_motion`. Detector-agnostic: the raw evidence every detector draws on.
2. **L1 schedule** (`scheduler.py:resolve_due`) — decides *which detectors run this tick*. Skips anything with `is_on=false`, outside its `when_scope` window, or off-tick for its `duty_cycle` (`thinned` = every k-th tick). `subject_scope` gates presence: `environment` detectors need no person; `patient` detectors suppress only when re-ID *positively* places the patient elsewhere. A per-track motion floor then drops still people for `requires_motion` detectors — **unless `severity ≥ 80`** (emergencies are never gated on motion). Output: the due set + which track each detector targets.
3. **L2 pack** (`packer.py` → `executor.py`) — turns the due set into VLM calls. `bucket` picks the fixed clip recipe (**SCENE** = full-frame 4 fps lite; **PERSON** = person crop; **RAPID** = tinted 8 fps flash crop, seizure only); `needs_audio` forces a native-video provider. Each detector contributes its `prompt_fragment` + `schema_json` (what the model is asked + the JSON it returns back). Grouped ≤3 per call, one call per person per bucket, dispatched **sequentially**. **Crowd fallback is a GLOBAL home-wide rule**: when the home-wide track count exceeds `scheduler_crowd_fallback_tracks` (default 4), crop-bucket detectors fall back to full-scene calls for that tick (no per-person fan-out storm).
4. **L3 decide** (`event_promotion.py` → `decision.py`) — turns verdicts into events. A detector clears the bar when `<id>_detected=true` AND `confidence ≥` its base threshold shifted by `sensitivity` (low +0.15 stricter / high −0.30 looser). The **alerting spine** decides the rest — `alert_trigger` (which moment fires) + `is_sustained` (temporal nature), which replaced the old `shape` column: `leading` fires on that clip's rising edge (sustained detectors re-arm after `episode_gap_seconds`); `trailing` fires on the falling edge when a sustained episode ends; `duration` fires when the condition has held longer than `duration_seconds`; `absence` fires when the signal never appeared by a scheduled deadline. `eventing` (`entry:{m,t}` windowed count + `confirm:{k,n}` re-check) adds confirmation; it is **NULL on every built-in ⇒ k=1**, and for custom detectors the entry M-of-N is now an **internal confidence gate derived from detection difficulty** (bucket + degradation class + safety/routine stakes, `compiler_v2.py`) — never a caregiver-stated duration. One event per clip (`severity` breaks ties), then a `cooldown_seconds` re-fire guard (default 30 s).
5. **L4 deliver** (`camera_ingest.py`, `clip_archival.py`) — APNs push + email (if `email_alerts_enabled`) + a ±15 s clip assembled in-process (skipped under privacy mode).

Provider: Gemini ladder (`vlm_routing.py`) — primary `gemini_model` (default `gemini-3.6-flash`), a same-path Gemini model fallback (`gemini_fallback_model`), then DashScope (Alibaba-direct Qwen3-VL, native video), then OpenRouter (frame-based Qwen) as the last-resort net. The per-caregiver `vlm_provider` column can re-root the chain. RAPID lane uses `rapid_model` (default `gemini-3-flash-preview`).

### Detector attributes (columns on `detectors`)

Grouped by the stage that reads them:

| attribute | what it does | stage |
|---|---|---|
| `is_on` | master on/off switch | L1 |
| `identifier` | the detector key; must be non-null to run at all | L1 |
| `subject_scope` | `patient` / `anyone` / `environment` — who it's about, and how attribution gates it | L1 |
| `requires_motion` | whether a still person is dropped by the motion floor (boolean; the old `evidence_branch` enum was collapsed into it — DONE 2026-07-26) | L1 |
| `when_scope` | time-of-day window; outside it the detector is skipped | L1 |
| `where_rooms` | canonical room-set scope (JSONB; null / `["ALL ROOMS"]` = everywhere) driving camera auto-assignment; the stored `where_scope` scalar was dropped — it's derived on serialize. Runtime "where" is still the `cameras_detectors` join | assign |
| `duty_cycle` | `every_eligible_tick` or `thinned` (only every k-th tick) | L1 |
| `bucket` | the clip recipe — `scene` / `person` / `rapid` (fps, crop, tier, channel are all fixed by the bucket) | L2 |
| `needs_audio` | forces the call onto a native-video (audio-carrying) provider | L2 |
| `prompt_fragment` / `schema_json` | this detector's VLM instruction + the structured-output schema it returns | L2 |
| `door_config` (per-camera) | wandering's exit-door zones, stored on `cameras_detectors` | L2 |
| `severity` | 0–100; **≥80 = emergency** (exempt from motion floor + cross-camera suppression); also breaks the one-event-per-clip tie | L1 & L3 |
| `sensitivity` | shifts the confidence bar: `low` +0.15, `standard` 0, `high` −0.30 | L3 |
| `alert_trigger` | which moment fires: `leading` (rising edge) / `trailing` (falling edge) / `duration` (held too long) / `absence` (never seen by deadline). Replaced `shape` (column + enum dropped 2026-07-26) | L3 |
| `is_sustained` | instant vs sustained temporal nature (the other half of the old `shape`) | L3 |
| `duration_seconds` | the "gone on too long" bar for `alert_trigger='duration'` (e.g. stove 600 s) | L3 |
| `eventing` | optional `entry:{m,t}` windowed count + `confirm:{k,n}` re-check; NULL ⇒ single-verdict (k=1). Entry M-of-N is INTERNAL — derived from difficulty at compile time, never user-set | L3 |
| `episode_gap_seconds` | for sustained detectors: how long an episode stays "ongoing" before it can re-fire | L3 |
| `cooldown_seconds` | minimum spacing between repeat events (default 30 s) | L3 |
| `category` | `safety` / `routine` — sets the default state gap (60 s vs 300 s) + event styling | L3 |

The base confidence bar (0.70; seizure 0.65) lives in `catalog.py`, not a row column — `sensitivity` shifts it.

## The detectors (9 built-in + hidden setup_test)

All `is_on=false` right now; all assigned to both cameras. "Fires ≥" is the standard-sensitivity bar.

| detector | trigger (alert_trigger / sustained) | subject | bucket → clip | when / where | severity | fires ≥ |
|---|---|---|---|---|---|---|
| choking | leading | patient | PERSON → clean crop | always / all | 100 | 0.70 |
| seizure ("Convulsion") | leading | patient | RAPID → tinted crop | always / all | 96 | **0.65** |
| fall | leading | patient | SCENE → clean | always / all | 95 | 0.70 |
| calling_for_help | leading | patient | SCENE (+audio) | always / all | 88 | 0.70 |
| emergency | leading | patient | SCENE | always / all | 80 | **0.40** (high sens) |
| agitation | leading, **sustained** | patient | SCENE (+audio) | always / all | 60 | 0.70 |
| wandering | leading | patient | SCENE | **10 PM–6 AM / Hallway** | 55 | 0.70 |
| medication | leading | patient | PERSON → clean crop | **8 AM/8 PM ±1h / Kitchen** | 20 | 0.70 |
| stove_left_on | **duration (600 s), sustained** | environment | SCENE | always / all | — | 0.70 |
| setup_test | leading | anyone | SCENE | always / all | 200 | fire-once |

Prompts + schema live per-detector in `catalog.py` (live seizure prompt in `rapid_call.py`). The catalog defines ~24 library detectors; only these are seeded.

## What's actually used (vs merely defined)

- **Buckets — 3 implemented, 2 defined-but-NOT-implemented, 1 retired.** Live & in use: **SCENE, PERSON, RAPID** (seizure). **FINE-MOTION** and **FACE** are recipe entries in `packer.py:_RECIPES` that **no detector uses and are not wired end-to-end** — FACE has no crop builder, so a face detector silently degrades to a full-scene call (`executor.py:261`); FINE-MOTION has a recipe but no detector. They are left in the code only as placeholders for validated-but-unshipped detectors (fidgeting → FINE-MOTION; grimacing / lip-smacking → FACE). **SNAPSHOT** is retired (normalized to scene).
- **k=1 for every built-in** — one VLM verdict = one alert. The confirmation layer (k-of-N + windowed-count, `decision.py`) is built; `eventing=NULL` on all built-ins ⇒ inert. Custom SUSTAINED detectors that are hard to see (distance-class or rapid/fine-motion buckets) get an internal entry M-of-N derived by `compiler_v2.py` — safety (2 of 180 s) vs routine (3 of 300 s); easy-to-see states still fire on the first confident detection.
- **`when_scope` is enforced** at runtime (medication 8am/8pm, wandering 10pm–6am); **rooms are seed-time** — `where_rooms` (the canonical set; stored `where_scope` is dropped) only drives which `cameras_detectors` rows get created, so a detector assigned to both cameras runs on both (medication fires in the Bedroom).
- **Overlay is RAPID-lane only** (fixed 2026-07-23; previously the global flag tinted every SCENE clip).
- Sensitivity shifts the bar: low +0.15, standard 0, high −0.30 (so high-sens emergency fires at 0.40).

## What's broken (2026-07 soak: 46 events, ~95% false positive)

Two failure families:
- **Bedroom, patient locked** — medication fires on phone/laptop-in-bed, wandering on ordinary walking, agitation on arm movement. Correct person, wrong verdict → **prompt problem**.
- **Living room, patient NOT identified** — choking/seizure/medication fire on visitors and are attributed to the patient (patient-scoped detectors run on UNKNOWN tracks; suppressed only on *positive* disconfirmation, never on "unidentified").

Root causes: k=1; catch-all prompts (`catalog.py`); confidence is a weak gate (soak averages wandering 1.00, medication 0.96); open-state runs the VLM every tick with no motion gate; the Qwen fallback produced the worst cluster. 11/46 events lost their clip (worker restart during the 15 s post-event capture).

## Files

```
src/services/detection/
  camera_ingest.py     the tick (gates → executor → promotion → clip)
  registry.py          which detectors run on a camera (is_on + assignment + wandering doors)
  catalog.py           built-in prompts + schema + severity + alerting spine  ← prompt work starts here
  catalog_v2.py        built-ins' v2 dimensions (requires_motion, subject_scope, bucket, duty_cycle)
  event_promotion.py   promotion gate (threshold, sensitivity, dedup, alert_trigger dispatch)  ← threshold work
  vlm_routing.py       per-account provider choice + fallback chain
  scheduling/scheduler.py   L1 due-set gating (incl. global crowd fallback)
  scheduling/dims.py        row → RuntimeDims resolution (catalog fallback)
  scheduling/compiler_v2.py custom-detector dimension classifier (incl. internal M-of-N)
  scheduling/packer.py      L2 bucket → calls
  scheduling/executor.py    build clips, run calls, merge, k-of-N confirm
  scheduling/decision.py    k-of-N + windowed-count + cooldown primitives  ← confirmation work
  signals/                  L0 motion/track signals; rapid_energy (seizure pre-gate)
  providers/                gemini / dashscope / openrouter(Qwen) / openai / _parse
  rapid_call.py             live seizure (RAPID) prompt
clip_archival.py       ±15 s clip assembly (in-process)
src/db/detector.py     the Detector row (all config columns)
src/services/scheduling/scheduler.py   absence + trailing runner (outside the tick path)
```

## Cleanup ledger (was "Known cleanup (TODO)")

- ~~**Collapse `evidence_branch` → `requires_motion`.**~~ **DONE 2026-07-26** (Phase H1, commit `fb7ad88`): the enum column is gone; `requires_motion` boolean is the stored contract (`scripts/apply_requires_motion_delta.sql`).
- Also landed since this doc was written (Phases C–H2, 2026-07-26/27): `shape` column + enum dropped in favor of `alert_trigger` + `is_sustained` (`1043491`, `d7a44f1`, `76fa214`); catalog remapped (`8e90d4f`); `duration` trigger + `trailing` firing + `schedule`→`absence` rename (`53b7a9e`); stored `where_scope` dropped for `where_rooms` (`146d693`); M-of-N made an internal difficulty-derived gate (`22c5d3e`); crowd fallback made a global home-wide rule (`06de5b9`).
