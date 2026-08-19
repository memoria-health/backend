# Shared cameras: primary + shared model

Status: PROPOSAL (2026-08-19). Supersedes the clone-per-patient behaviour
in `src/api/cameras_ingest.py` gateway provisioning.

## The problem, measured

One physical camera, three rows:

```
Warner Camera 1908 → cam_70fe8950 → Margaret O'Brien   (0 enrolled faces, 3 detectors)
                   → cam_434d178e → Aryaman            (25 enrolled faces, 4 detectors)
                   → cam_b5f841f7 → Soph               (0 enrolled faces, 6 detectors)
```

Gateway provisioning loops `for patient_id in patient_ids` and creates a
row per patient of that caregiver. Consequences, all live in prod today:

1. **N× cost for one scene** — the same stream is published into N rooms:
   N× uplink, N× ingest/decode, N× ticks, N× Gemini spend, N identity slots.
2. **No backfill** — a patient added after provisioning gets nothing
   (Jane Doe, 2026-08-19). Neither account-scoped nor patient-scoped.
3. **Duplicate events** — with `identity optional` (2026-08-19 gate fix),
   two unenrolled patients on the same physical camera both fire routine
   events for the same person. Before that fix they fired none. Neither
   is right; the missing condition is *how many patients this camera
   covers*.
4. **Capacity mispriced** — `NEMARA_NODE_CAPACITY` counts patients; the
   real ceiling is camera loops (~15 in SLA on one 4090, measured
   2026-08-16). A 3-patient room with 5 cameras costs 15 loops for 5
   scenes — a whole node.

## The model

A physical camera is ONE row with ONE primary patient and zero or more
SHARED patients. One stream, one tick, one set of model calls.
Attribution happens at event time.

```
cameras (unchanged)          camera_patients (new)
  id                           camera_id  → cameras.id
  patient_id  = PRIMARY        patient_id → patients.id
  ...                          added_at
                               PK (camera_id, patient_id)
```

`coverage(camera) = {cameras.patient_id} ∪ {camera_patients.patient_id}`

Primary stays in `cameras.patient_id` (no FK migration, no dual source of
truth for "who owns the stream"). The join table holds only the extra
patients. RLS policies extend to the join table so a caregiver sees a
shared camera under each of their patients.

### Streaming: unchanged

The camera keeps publishing into `patient_{primary}`. Shared patients do
NOT get their own stream — they get a share of the *results*. This
preserves the room-per-patient invariant the identity node is built on
and is why this design is small.

### Detection: one tick, union of detector sets

The tick loads detectors for every patient in coverage and groups them by
`(identifier, prompt_hash)`:

- same identifier, same prompt → ONE call, result fans out to each patient
- differing prompts (custom detectors, refinements) → one call per group

Per-patient *thresholds* apply at promotion, not at call time — the model
returns a confidence and each patient's `sensitivity` is applied to it.
So differing sensitivity never costs an extra call.

Cost: one physical camera costs the UNION of detectors, not N× the full
set. For the Warner case: {fall, sleep, drinking_water, …} once, instead
of 13 detector-runs across three clones.

### Identity: binary → multi-class

Today `_load_face_matrix(patient_id)` loads one patient's gallery and the
question is "is this the patient or not". For a shared camera the process
must load galleries for ALL covered patients and answer "WHICH of these
patients, or none". The matching code (top-k mean cosine) already
supports labelled classes; the change is the gallery load and the return
type, not the algorithm.

### Attribution rules (the heart)

Evaluated per detector verdict, in order:

| Coverage | Identity says | Routine / behaviour / care | Safety |
|---|---|---|---|
| exactly 1 patient | nothing | fire for that patient | fire |
| exactly 1 patient | "not the patient" | **suppress** (visiting-aunt rule, unchanged) | fire |
| >1 patient | "this is patient P" | fire for P only | fire for P |
| >1 patient | nothing / unknown person | **suppress** | fire for PRIMARY, narrative flags "person not identified" |

Two consequences worth stating plainly:

- **Identity stays optional for single-patient cameras** — the common
  case, and the 2026-08-19 contract. The gate condition becomes
  *"does this camera cover more than one patient"*, which is strictly
  better than the enrollment probe it replaces (that probe was a proxy
  for exactly this question).
- **Multi-patient rooms need enrollment for routine tracking.** You
  cannot honestly write "Margaret ate lunch" from a room where Margaret
  and Soph both live and nobody was identified. Safety still fires,
  because missing a fall is worse than attributing it to the wrong
  resident. This is a product truth to communicate in onboarding, not a
  bug to engineer around.

### Capacity: count cameras, not patients

`NEMARA_NODE_CAPACITY` becomes a CAMERA-LOOP budget (measured ceiling ~15
on a 4090; suggest 12 for headroom). The roster claims rooms until the
budget is spent, counting each room's online cameras.

Folded in from the roster starvation fix (same PR):
- **claim order**: homes with a live camera first, oldest-first among them
- **warm standby**: dormant homes may hold leftover slots (an idle room
  costs ~zero GPU) but are evicted the moment a live-camera home needs one
- **never evict** a home that has a live camera
- **watchdog**: count a home as blind only if it has a live camera, so the
  critical page means something

### Migration of existing clones

Clone rows carry history (`analyses`, `events` reference `camera_id`).
They must NOT be deleted.

1. Pick the primary: the clone with the most analyses (or oldest).
2. Insert `camera_patients` rows for the other clones' patients.
3. Mark the other clone rows `disabled = true` and stop the gateway
   publishing them (roster stops listing them).
4. History stays queryable against the old camera ids; new data lands on
   the primary.

### App

- Sensors tab (per patient) lists cameras where the patient is in
  coverage, with a "shared" badge when the primary is a different patient.
- Adding a patient prompts "which existing cameras cover this patient?" —
  solving the backfill gap explicitly instead of silently cloning.
- Privacy (`room_class`, `privacy_enabled`) is per physical camera and is
  therefore naturally shared — correct: a bathroom is a bathroom for
  everyone.

## Phasing

**Phase 1 — unblock (small, no schema):**
- roster: capacity in camera loops, live-first claim, evict-dormant-on-demand
- watchdog: blind = live-camera homes only
- gate: replace the enrollment probe with "coverage == 1" (coverage is
  today always 1 by row, so this is behaviour-preserving until phase 2)
- gateway provisioning: STOP cloning — create one row for the primary
  patient (first by created_at), so the problem stops growing

**Phase 2 — the model:**
- `camera_patients` table + RLS
- coverage-aware detector union and call grouping
- attribution rules above
- multi-patient gallery in the identity node

**Phase 3 — finish:**
- migrate existing clones
- app: shared badge, patient-add camera picker
- capacity/autoscale on the honest signal (live-camera homes unassigned)

## Open questions for Aryaman

1. **Safety attribution in a shared room with no identity** — fire on the
   primary (proposed), or fire on every covered patient so nobody misses
   it? Duplicates vs a possible wrong name on an urgent alert.
2. **Warm standby** — should dormant homes hold leftover slots at all, or
   keep slots free/reserved for live homes?
3. **Roster poll** — prod is on the 30s default; drop to 10s for faster
   camera-wake provisioning?
4. **RCFE shape** — is the target one camera covering many residents
   (shared model matters enormously), or one camera per resident's room
   (shared model is mostly about common areas)?
