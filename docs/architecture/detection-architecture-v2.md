# Detection — as-built pipeline & detectors

What actually runs today (2026-07-23). Onboarding for detector accuracy work. Measurement history that justifies the recipes is in `detection-inference-architecture.md`.

## Status: detection is PAUSED

- All detectors `is_on=false`; `NEMARA_REID_SOAK_ONLY=true` → the executor is skipped, **zero VLM calls**. Last real inference was 2026-07-20 (soak end).
- **Resume:** set `.env NEMARA_REID_SOAK_ONLY=false` + recreate the container (env is read at boot), then `UPDATE detectors SET is_on=true WHERE identifier <> 'setup_test';` (next tick). Leave `setup_test` off — fire-once self-test.

## The pipeline (one camera, every 10 s tick)

`RoomWorker` (per room) spawns a `CameraIngest` (per camera) that runs a fire-and-forget tick over an ~11 s clip:

1. **L0 signals** (`signals/`, `reid/`) — always-on YOLO+BoT-SORT tracking + ArcFace face attribution → per-track `{bbox, label ∈ PATIENT|OTHER|UNKNOWN}`; per-track + environment motion.
2. **L1 schedule** (`scheduling/scheduler.py:resolve_due`) — per detector: when-window, duty cycle, subject suppression, per-track motion floor (severity ≥ 80 is **exempt**) → the due set.
3. **L2 pack** (`scheduling/packer.py`) — group due detectors into VLM calls by (bucket, track): one uncapped **SCENE** call + **≤3-per-call PERSON crops** + the **RAPID** seizure crop. `executor.py` builds the clips and runs the calls **sequentially**.
4. **L3 decide** (`event_promotion.py`) — a detector fires if `detected=true` AND `confidence ≥ threshold`; `state` detectors also need a rising edge; one event per clip (highest severity wins); 30 s per-(patient,detector) dedup.
5. **L4 deliver** (`camera_ingest.py`, `clip_archival.py`) — APNs push + email + a ±15 s clip assembled in-process.

Provider: Gemini `gemini-3.1-flash-lite`, with a **silent fallback to Qwen** (`openrouter`) on any Gemini error.

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
