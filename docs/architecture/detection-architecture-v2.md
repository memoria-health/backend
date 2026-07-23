# Detection — as-built pipeline & detectors

What actually runs today (2026-07-23). Onboarding for detector accuracy work. Measurement history that justifies the recipes is in `detection-inference-architecture.md`.

## Status: detection is PAUSED

- All detectors `is_on=false`; `NEMARA_REID_SOAK_ONLY=true` → the executor is skipped, **zero VLM calls**. Last real inference was 2026-07-20 (soak end).
- **Resume:** set `.env NEMARA_REID_SOAK_ONLY=false` + recreate the container (env is read at boot), then `UPDATE detectors SET is_on=true WHERE identifier <> 'setup_test';` (next tick). Leave `setup_test` off — fire-once self-test.

## The pipeline (one camera, every 10 s tick)

`RoomWorker` (per room) spawns a `CameraIngest` (per camera) that runs a fire-and-forget tick over an ~11 s clip. Each stage reads specific detector attributes (glossary below):

1. **L0 signals** (`signals/`, `reid/`) — always-on YOLO+BoT-SORT tracking + ArcFace face attribution → per-track `{bbox, label ∈ PATIENT|OTHER|UNKNOWN}`; per-track motion + whole-frame `env_motion`. Detector-agnostic: the raw evidence every detector draws on.
2. **L1 schedule** (`scheduler.py:resolve_due`) — decides *which detectors run this tick*. Skips anything with `is_on=false`, outside its `when_scope` window, or off-tick for its `duty_cycle` (`thinned` = every k-th tick). `subject_scope` gates presence: `environment` detectors need no person; `patient` detectors suppress only when re-ID *positively* places the patient elsewhere. A per-track motion floor then drops still people for `requires_motion` detectors — **unless `severity ≥ 80`** (emergencies are never gated on motion). Output: the due set + which track each detector targets.
3. **L2 pack** (`packer.py` → `executor.py`) — turns the due set into VLM calls. `bucket` picks the fixed clip recipe (**SCENE** = full-frame 4 fps lite; **PERSON** = person crop; **RAPID** = tinted 8 fps flash crop, seizure only); `needs_audio` forces a native-video provider. Each detector contributes its `prompt_fragment` + `schema_json` (what the model is asked + the JSON it returns back). Grouped ≤3 per call, one call per person per bucket, dispatched **sequentially**.
4. **L3 decide** (`event_promotion.py` → `decision.py`) — turns verdicts into events. A detector clears the bar when `<id>_detected=true` AND `confidence ≥` its base threshold shifted by `sensitivity` (low +0.15 stricter / high −0.30 looser). `shape` decides the rest: `point_in_time` fires on that one clip; `state` fires only on a rising edge and re-arms after `episode_gap_seconds`; `scheduled` fires on *absence* at a deadline. `eventing` (`entry:{m,t}` windowed count + `confirm:{k,n}` re-check) would add confirmation but is **NULL on every built-in ⇒ k=1**. One event per clip (`severity` breaks ties), then a `cooldown_seconds` re-fire guard (default 30 s).
5. **L4 deliver** (`camera_ingest.py`, `clip_archival.py`) — APNs push + email (if `email_alerts_enabled`) + a ±15 s clip assembled in-process (skipped under privacy mode).

Provider: Gemini `gemini-3.1-flash-lite`, with a **silent fallback to Qwen** (`openrouter`) on any Gemini error.

### Detector attributes (columns on `detectors`)

Grouped by the stage that reads them:

| attribute | what it does | stage |
|---|---|---|
| `is_on` | master on/off switch | L1 |
| `identifier` | the detector key; must be non-null to run at all | L1 |
| `subject_scope` | `patient` / `anyone` / `environment` — who it's about, and how attribution gates it | L1 |
| `evidence_branch` → `requires_motion` | whether a still person is dropped by the motion floor (only the `person_motion` bit is read — see TODO) | L1 |
| `when_scope` | time-of-day window; outside it the detector is skipped | L1 |
| `where_scope` | room hint for initial auto-assignment only; runtime "where" is the `cameras_detectors` join | assign |
| `duty_cycle` | `every_eligible_tick` or `thinned` (only every k-th tick) | L1 |
| `bucket` | the clip recipe — `scene` / `person` / `rapid` (fps, crop, tier, channel are all fixed by the bucket) | L2 |
| `needs_audio` | forces the call onto a native-video (audio-carrying) provider | L2 |
| `prompt_fragment` / `schema_json` | this detector's VLM instruction + the structured-output schema it returns | L2 |
| `door_config` (per-camera) | wandering's exit-door zones, stored on `cameras_detectors` | L2 |
| `severity` | 0–100; **≥80 = emergency** (exempt from motion floor + cross-camera suppression); also breaks the one-event-per-clip tie | L1 & L3 |
| `sensitivity` | shifts the confidence bar: `low` +0.15, `standard` 0, `high` −0.30 | L3 |
| `shape` | `point_in_time` (fires per clip) / `state` (rising edge + hysteresis) / `scheduled` (fires on absence) | L3 |
| `eventing` | optional `entry:{m,t}` windowed count + `confirm:{k,n}` re-check; NULL ⇒ single-verdict (k=1) | L3 |
| `episode_gap_seconds` | for `state` detectors: how long an episode stays "ongoing" before it can re-fire | L3 |
| `cooldown_seconds` | minimum spacing between repeat events (default 30 s) | L3 |
| `category` | `safety` / `routine` — sets the default state gap (60 s vs 300 s) + event styling | L3 |

The base confidence bar (0.70; seizure 0.65) lives in `catalog.py`, not a row column — `sensitivity` shifts it.

## The detectors (9 built-in + hidden setup_test)

All `is_on=false` right now; all assigned to both cameras. "Fires ≥" is the standard-sensitivity bar.

| detector | shape | subject | bucket → clip | when / where | severity | fires ≥ |
|---|---|---|---|---|---|---|
| choking | point_in_time | patient | PERSON → clean crop | always / all | 100 | 0.70 |
| seizure | point_in_time | patient | RAPID → tinted crop | always / all | 96 | **0.65** |
| fall | point_in_time | patient | SCENE → clean | always / all | 95 | 0.70 |
| calling_for_help | point_in_time | patient | SCENE (+audio) | always / all | 88 | 0.70 |
| emergency | point_in_time | patient | SCENE | always / all | 80 | **0.40** (high sens) |
| agitation | **state** | patient | SCENE (+audio) | always / all | 60 | 0.70 |
| wandering | point_in_time | patient | SCENE | **10 PM–6 AM / Hallway** | 55 | 0.70 |
| medication | point_in_time | patient | PERSON → clean crop | **8 AM/8 PM ±1h / Kitchen** | 20 | 0.70 |
| stove_left_on | point_in_time | environment | SCENE | always / all | — | 0.70 |
| setup_test | point_in_time | anyone | SCENE | always / all | 200 | fire-once |

Prompts + schema live per-detector in `catalog.py` (live seizure prompt in `rapid_call.py`). The catalog defines ~24 library detectors; only these are seeded.

## What's actually used (vs merely defined)

- **Buckets — 3 implemented, 2 defined-but-NOT-implemented, 1 retired.** Live & in use: **SCENE, PERSON, RAPID** (seizure). **FINE-MOTION** and **FACE** are recipe entries in `packer.py:_RECIPES` that **no detector uses and are not wired end-to-end** — FACE has no crop builder, so a face detector silently degrades to a full-scene call (`executor.py:261`); FINE-MOTION has a recipe but no detector. They are left in the code only as placeholders for validated-but-unshipped detectors (fidgeting → FINE-MOTION; grimacing / lip-smacking → FACE). **SNAPSHOT** is retired (normalized to scene).
- **k=1 for every built-in** — one VLM verdict = one alert. The confirmation layer (k-of-N + windowed-count, `decision.py`) is built and used by custom detectors, but `eventing=NULL` on all built-ins ⇒ inert.
- **`when_scope` is enforced** at runtime (medication 8am/8pm, wandering 10pm–6am); **`where_scope` is seed-only** (both are assigned to both cameras, so medication fires in the Bedroom).
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
  catalog.py           built-in prompts + schema + severity  ← prompt work starts here
  event_promotion.py   promotion gate (threshold, sensitivity, dedup)  ← threshold work
  scheduling/scheduler.py   L1 due-set gating
  scheduling/packer.py      L2 bucket → calls
  scheduling/executor.py    build clips, run calls, merge, k-of-N confirm
  scheduling/decision.py    k-of-N + windowed-count + cooldown primitives  ← confirmation work
  signals/                  L0 motion/track signals; rapid_energy (seizure pre-gate)
  providers/                gemini / openrouter(Qwen) / _parse
  rapid_call.py             live seizure (RAPID) prompt
clip_archival.py       ±15 s clip assembly (in-process)
src/db/detector.py     the Detector row (all config columns)
```

## Known cleanup (TODO)

- **Collapse `evidence_branch` → `requires_motion`.** The `evidence_branch` column (`environment` / `person_motion` / `person_still`) is legacy dead weight: at runtime only its "is it `person_motion`" bit is read, to set the `requires_motion` boolean (`dims.py:120`). `person_still` is never consumed; `environment` duplicates `subject_scope=environment`. Fully removing it = a small migration (add `requires_motion`, backfill, drop `evidence_branch`) + `catalog_v2.py` / `compiler_v2.py` / `db/detector.py` + ~10 tests (~45–60 min). Deferred to its own commit to keep the schema change off the doc/overlay-fix handoff.
