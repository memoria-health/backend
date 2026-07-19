# Plan: Patient Sharing (multi-caregiver access) + future per-patient roles

Branch: `feat/patient-sharing` (off `origin/main` @ c159356)

## Goal

Let multiple caregivers — each with their **own** Memoria login — access the
**same** patient. MVP = everyone shared has full access (view + operate), except
owner-only actions (share/unshare, delete patient). Structure it so **per-patient
differing permission levels** ("roles") drop in later with **no migration**.

Non-goals (MVP): an org/account layer; a capability matrix; a patient-facing
role. Those are additive later.

## Core idea

Ownership today is a single FK `patients.caregiver_id` + a manual
`.where(caregiver_id == …)` in every repo (RLS is only belt-and-braces; the app
enforces in Python). Sharing = introduce a **membership** table and change the
question everywhere from "is this MY patient?" → "am I a MEMBER of this patient?"

Two enforcement layers must both change:
1. **SQLAlchemy repos** (the load-bearing gate — service-role client bypasses RLS).
2. **Supabase RLS** (for direct client / Realtime access — web VideoGrid, etc.).

The `role` column exists from day one but is not enforced in the MVP.

---

## 1. Data model

New table (in `supabase/schema.sql`, near patients):

```
create table public.patient_members (
  id           uuid primary key default gen_random_uuid(),
  patient_id   uuid not null references public.patients(id)   on delete cascade,
  caregiver_id uuid not null references public.caregivers(id) on delete cascade,
  role         text not null default 'caregiver'
                 check (role in ('owner','caregiver','viewer')),  -- extend later
  invited_by   uuid references public.caregivers(id) on delete set null,
  created_at   timestamptz not null default now(),
  unique (patient_id, caregiver_id)
);
create index patient_members_caregiver_idx on public.patient_members(caregiver_id);
create index patient_members_patient_idx   on public.patient_members(patient_id);
```

- **No new columns on existing tables.** `patients.caregiver_id` stays = the
  original owner (also mirrored as a `role='owner'` membership row).
- Roles seeded: `owner` (the creator), `caregiver` (shared, full access in MVP).
  `viewer` is defined but unused until the roles layer.

### Backfill (idempotent)

```
insert into public.patient_members (patient_id, caregiver_id, role)
select id, caregiver_id, 'owner' from public.patients
on conflict (patient_id, caregiver_id) do nothing;
```

### Access helper (single source of truth)

```
create or replace function public.has_patient_access(pid uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.patient_members m
    where m.patient_id = pid and m.caregiver_id = auth.uid()
  );
$$;
```

---

## 2. RLS rewrite  (`supabase/schema.sql`, policies at ~lines 721–801)

Rewrite the **7 patient-scoped** policies to use membership; leave the
caregiver-self ones untouched.

| Policy | Currently | Change to |
|---|---|---|
| `patients_owner` | `caregiver_id = auth.uid()` | `has_patient_access(id)` (see note) |
| `cameras_owner` | `patient_id in (…caregiver_id=uid)` | `has_patient_access(patient_id)` |
| `detectors_owner` | same | `has_patient_access(patient_id)` |
| `cameras_detectors_owner` | join on patients.caregiver_id | `has_patient_access(<camera's patient>)` |
| `events_owner` | same subselect | `has_patient_access(patient_id)` |
| `patient_face_embeddings_owner` | same | `has_patient_access(patient_id)` |
| `analyses_owner` | `caregiver_id = auth.uid()` | `has_patient_access(patient_id)` |

- **`patients_owner` write note:** SELECT/UPDATE via `has_patient_access(id)`;
  INSERT check must stay `caregiver_id = auth.uid()` (a new patient's creator is
  its owner). Split into per-command policies if needed.
- **New policy on `patient_members`:** a caregiver may read rows for patients
  they can access; only owners may insert/delete (owner-guard also enforced in
  the API). Keep RLS here strict.
- Unchanged: `caregivers_self`, `caregiver_devices_owner`, `feedback_owner`,
  `ui_events_owner` (caregiver-self, not patient-scoped).

---

## 3. Backend Python — repos (the primary gate)

Add a shared helper (e.g. `src/repos/access.py`):
`accessible_patient_ids(caregiver_id)` → SQLAlchemy subquery selecting
`patient_members.patient_id where caregiver_id == :cid`. Then swap ownership
checks in the **~6 patient-scoped repos**:

- `patient_repo.py` — `list_for_caregiver` → patients whose id ∈ accessible ids;
  `find_by_id(…, caregiver_id)` → id == pid AND pid ∈ accessible ids (keeps the
  404-not-403 semantics); `update_partial`/`delete` likewise (delete = owner-only,
  guarded in service layer).
- `camera_repo.py`, `detector_repo.py`, `camera_detector_repo.py`,
  `event_repo.py`, `face_embedding_repo.py` — every `caregiver_id`-scoped query
  → membership scope via the helper.
- `analyses_store.py` — scope reads by membership (patient_id ∈ accessible).
- **Untouched:** `caregiver_repo`, `caregiver_device_repo`, `feedback_repo`
  (caregiver-self).

Owner-only actions (delete patient, share/unshare) check
`patient_members.role == 'owner'` for the caller.

## 4. Backend Python — endpoints

- **LiveKit token** (`src/api/livekit.py`): the room→patient ownership check that
  gates minting a `patient_<id>` viewer token becomes a **membership** check, or
  a shared caregiver can't get live video.
- **New sharing endpoints** (new router, e.g. `src/api/patient_members.py`):
  - `POST /v1/patients/{id}/members` — body `{email, role?}`. Owner-only. Look up
    caregiver by email; if found, insert membership; if not, create a pending
    invite (MVP option A: existing-users-only → 404 "no Memoria account"; option
    B: pending invites — deferred).
  - `GET /v1/patients/{id}/members` — list members (any member).
  - `DELETE /v1/patients/{id}/members/{caregiverId}` — owner-only; can't remove
    the last owner.
- Wire the router in the app; add request/response schemas.

---

## 5. Clients (later phase — Task #20)

- Patient lists **inherit** membership automatically (they call the API, which
  now returns shared patients). Verify only.
- Web + iOS: add a **"Manage access / Share"** screen (list members, invite by
  email, remove). Small CRUD ×2.
- Realtime (web VideoGrid `cameras` subscription via authed client) works once
  RLS is updated — no client change.

---

## 6. Migration / rollout order

1. Apply schema delta (table + fn + RLS + backfill) to the DB **and** deploy the
   backend repo changes **together** (app-layer check and DB must agree).
2. Then ship the client share screens.
- Local: apply the delta to local Supabase (:54341); prod: Supabase Management
  API delta (never reset prod), then `railway up`.

## 7. Local testing plan (Task #19)

- Confirm local backend (:8002) + Supabase (:54341) up.
- Apply the delta locally; verify table/fn/policies/backfill.
- Create **two** caregiver accounts (A owns Patient P; B is a stranger).
- Before share: B gets 404 on P and its cameras/detectors/events; B can't mint a
  LiveKit token for P's room.
- A shares P with B (by email). After: B lists P, reads its children, mints a
  token; A still owns. A non-member C still blocked.
- Remove B; B loses access immediately (and Realtime stops delivering P's rows).

## 8. Future roles layer (no migration)

Map role → capability set; enforce per-patient in `has_patient_access`-style
helpers + API for specific actions; clients hide features from the capabilities
returned per patient. Because role already lives on the membership row, this is
purely additive.

## 9. Risks / gotchas

- **Two layers must agree** — the Python repos are the real gate (service-role
  bypasses RLS); don't stop at RLS.
- **RLS test matrix** — member sees / non-member blocked, per patient-scoped
  table + Realtime. Centralizing via `has_patient_access()` shrinks the surface.
- **INSERT semantics** — new patient's creator must become owner (backfill +
  create-patient path inserts an owner membership row).
- **Last-owner protection** — can't unshare/remove the final owner; delete-patient
  is owner-only.
- **Sensor/camera api_key auth is separate** — unchanged; "who can share" is an
  owner action only.
- **Backfill idempotency** — `on conflict do nothing`, safe to re-run.
