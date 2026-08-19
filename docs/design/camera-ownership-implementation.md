# Camera ownership: implementation plan

Status: READY TO BUILD (2026-08-19). Design rationale lives in
`shared-cameras.md`; this is the build order.

Scope decisions already made: bridge untouched; iOS first (web is a TODO
that copies the iOS placement); `cameras.patient_id` dropped entirely;
identity compares against every patient a camera covers; safety in a
shared room fires for every covered patient; ops may assign on a
customer's behalf.

## 1. Schema (one delta, additive then destructive in two steps)

```
cameras
  id, caregiver_id → caregivers.id   ← NEW: the owning account
  room, icon_name, livekit_identity, device fields, privacy, state …
  (patient_id and livekit_room REMOVED — see below)

camera_patients                       ← NEW: the whole camera↔patient relation
  camera_id  → cameras.id  ON DELETE CASCADE
  patient_id → patients.id ON DELETE CASCADE
  is_primary boolean not null default false
  added_at   timestamptz  not null default now()
  PRIMARY KEY (camera_id, patient_id)
  partial unique index: one primary per camera
```

`coverage(camera)` = every row in `camera_patients`. Unassigned = no rows.
The LiveKit room is derived, not stored: `patient_<primary.patient_id>`.
`livekit_room` on `cameras` becomes a generated/derived value at query
time — the roster already returns it per poll, and the bridge reads it
from there, so removing the stored copy is safe.

Delta order (idempotent, additive first — prod safety):
1. add `cameras.caregiver_id`, backfill from `patients.caregiver_id`, set NOT NULL
2. create `camera_patients`; backfill one `is_primary=true` row per existing camera
3. RLS: caregiver sees cameras where `caregiver_id` is theirs; patient-scoped
   reads go through `camera_patients`
4. only after the code no longer reads them: drop `cameras.patient_id`,
   `cameras.livekit_room`

Step 4 ships in a SECOND deploy, after step 1-3 plus the code are live and
verified. Never in the same release — a rollback with the column already
dropped is unrecoverable.

## 2. Backend

**Call-site refactor (35 references to `cameras.patient_id`).** Each becomes
either a join through `camera_patients` (`is_primary` when one patient is
required) or a coverage query (when all covered patients are wanted).
Mechanical, but it is the bulk of the work and must be done before step 4.

**Provisioning** — `/v1/gateway/provision-camera` (what the bridge calls):
stop looping over patients. Resolve the account from the email, create ONE
camera with `caregiver_id` set and NO `camera_patients` rows. Return one
identity so the bridge's existing response contract holds. If the email
matches no account: 422, unchanged shape.

**Roster** — `/v1/gateway/roster` and `/v1/identity/roster` only return
cameras that have a primary. An unassigned camera therefore never
publishes and never occupies an identity slot.

**Assignment API** (new, caregiver-authenticated; ops may act for any
account they administer):
- `GET  /v1/cameras` — this account's cameras, each with coverage + a flag
  for unassigned ("ready to set up")
- `POST /v1/cameras/{id}/assign` — `{room, icon_name, primary_patient_id,
  also_patient_ids[]}`; creates coverage rows. Idempotent.
- `PATCH /v1/cameras/{id}/coverage` — add/remove covered patients, change
  primary. **Never changes `livekit_identity`** — the bridge pins it
  locally; only the derived room moves.
- `DELETE /v1/cameras/{id}/coverage` — unassign (all rows removed). Camera
  stays on the account, stops streaming at the next roster poll.

**Detection** — a tick resolves coverage for its camera and loads the union
of those patients' detectors, grouped by `(identifier, prompt)`: identical
detectors run ONCE and the verdict fans out; differing prompts run
separately. Per-patient `sensitivity` applies at promotion against the
returned confidence, so differing thresholds never cost an extra call.

**Attribution** (replaces this morning's enrollment probe):

| Coverage | Identity | Routine/behaviour/care | Safety |
|---|---|---|---|
| 1 | silent | fire | fire |
| 1 | "not the patient" | suppress | fire |
| >1 | "it is P" | fire for P | fire for P |
| >1 | silent | suppress | fire for EVERY covered patient |

**Identity node** — `_load_face_matrix(patient_id)` becomes a load for every
patient covered by any camera in the room, and matching returns WHICH
patient (labelled multi-class) instead of patient-vs-other. The matcher
(top-k mean cosine) already supports labels; the change is the gallery
load and the return type.

## 3. iOS (nemara-caregiver-app)

- Move "Add a sensor" OFF the per-patient sensors page and ONTO the roster
  page. Same sheet.
- The sheet's Memoria-sensors section lists cameras returned by
  `GET /v1/cameras` with no coverage — "ready to set up". This is NOT
  network discovery: the list comes from the backend, so the caregiver can
  be anywhere.
- Choosing one: name, icon, and assign to one or more patients (one marked
  primary).
- Per-patient sensors page keeps listing that patient's cameras and marks
  any also covering another resident.
- Camera detail: "Delete sensor" becomes **Unassign sensor**.
- Settings: outright **Delete sensor** (destructive — deleting a camera
  CASCADES its analyses; the UI must say so).
- Roster page: a single button beside "Add patient" for discover +
  reassign.

## 4. Web (nemara-web-app)

TODO only in this pass: mirror the iOS placement and flow once iOS is
settled.

## 5. Migration

Prod's three Warner clone rows were resolved by hand on 2026-08-19: the
Margaret and Soph rows were DELETED (cascading ~5k duplicate analyses of
the same footage, 0 events), leaving Aryaman's single Warner row. No
further clone migration is outstanding. New provisioning cannot create
clones once §2 ships.

## 6. Order of work

1. Delta steps 1-3 (additive) — safe alone, changes no behaviour
2. Backend call-site refactor + provisioning + roster + assignment API
3. iOS surfaces against the new API
4. Detection union + attribution + identity multi-class
5. Delta step 4 (drop the old columns) — separate deploy, after soak

Steps 1-2 and 4 are independent enough to run as parallel workstreams;
step 3 needs the API contract from step 2 but not its implementation.

## 7. Risks

- **Dropping `patient_id` is irreversible in a rollback.** Hence the
  two-deploy split.
- **A shared camera's tick makes more concurrent model calls** (union of
  two rosters). Bounded by the packer's existing caps; watch p95 on the
  first shared camera.
- **Attribution is the only NEW accuracy risk** and it is bounded by the
  table above: never a wrong routine record, never a missed safety alert.
- Single-patient cameras — the majority in the RCFE shape — are unaffected
  end to end.
