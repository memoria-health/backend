-- Camera ownership: a camera belongs to an ACCOUNT, and covers 0..N patients.
--
-- Today a camera row carries exactly one patient_id, so a physical camera
-- watching a shared room has to be duplicated per resident. Gateway
-- provisioning did exactly that (one row per patient of the caregiver),
-- which on 2026-08-19 had one Trueview camera publishing three separate
-- streams, paying three times for ingest and model calls on identical
-- footage, and firing duplicate events. It also could not express "this
-- camera arrived before anyone decided who it watches", which is the
-- normal case when the installer sets the hardware up.
--
-- After this delta:
--   cameras.caregiver_id  = the owning account (always known at provision)
--   camera_patients       = every camera->patient link, one flagged primary
--   coverage(camera)      = its camera_patients rows; none = unassigned
--
-- ADDITIVE ONLY. `cameras.patient_id` and `cameras.livekit_room` stay and
-- stay authoritative until the application no longer reads them; dropping
-- them is a SECOND delta after this one has soaked, because a rollback
-- with those columns already gone is unrecoverable.

-- ── 1. Owning account ───────────────────────────────────────────────
alter table public.cameras
  add column if not exists caregiver_id uuid
    references public.caregivers(id) on delete cascade;

-- Backfill from the patient that currently owns the row. Safe to re-run.
update public.cameras c
   set caregiver_id = p.caregiver_id
  from public.patients p
 where p.id = c.patient_id
   and c.caregiver_id is null;

-- Deliberately NULLABLE for now. This is the expand step: the column
-- exists and is backfilled, but nothing is forced to set it yet, so code
-- and tests that predate it keep working. NOT NULL is enforced in the
-- follow-up delta that drops patient_id, once every insert sets it.

create index if not exists cameras_caregiver_idx
  on public.cameras (caregiver_id);

-- A camera the installer has set up but nobody has assigned yet has no
-- patient at all, so the column has to allow it. Relaxing NOT NULL is
-- reversible (it can be re-imposed while every row still has a value);
-- DROPPING the column is not, which is why that waits for the contract
-- delta. Until then patient_id is kept in sync with the primary, so
-- readers that have not migrated yet keep seeing what they expect.
alter table public.cameras alter column patient_id drop not null;

-- ── 2. Coverage ─────────────────────────────────────────────────────
create table if not exists public.camera_patients (
  camera_id  uuid not null references public.cameras(id)  on delete cascade,
  patient_id uuid not null references public.patients(id) on delete cascade,
  -- The primary is the patient whose LiveKit room the camera publishes
  -- into. A camera streams to exactly one room; the other covered
  -- patients receive the RESULTS, never a second copy of the video.
  is_primary boolean not null default false,
  added_at   timestamptz not null default now(),
  primary key (camera_id, patient_id)
);

-- One primary per camera, enforced by the database rather than by hope.
create unique index if not exists camera_patients_one_primary
  on public.camera_patients (camera_id) where is_primary;

create index if not exists camera_patients_patient_idx
  on public.camera_patients (patient_id);

-- Backfill: today's single patient becomes the primary.
insert into public.camera_patients (camera_id, patient_id, is_primary)
select c.id, c.patient_id, true
  from public.cameras c
 where c.patient_id is not null
on conflict (camera_id, patient_id) do nothing;

-- ── 3. Row-level security ───────────────────────────────────────────
-- A caregiver reaches a camera two ways: they own the account it belongs
-- to (which is the ONLY way to see one nobody has assigned yet), or they
-- have access to a patient it covers (which is how a shared caregiver
-- keeps seeing the cameras they always saw).
drop policy if exists cameras_member on public.cameras;
create policy cameras_member on public.cameras
  for all using (
    caregiver_id = auth.uid()
    or exists (
      select 1 from public.camera_patients cp
       where cp.camera_id = cameras.id
         and public.has_patient_access(cp.patient_id)
    )
  )
  with check (
    caregiver_id = auth.uid()
    or exists (
      select 1 from public.camera_patients cp
       where cp.camera_id = cameras.id
         and public.has_patient_access(cp.patient_id)
    )
  );

alter table public.camera_patients enable row level security;
drop policy if exists camera_patients_member on public.camera_patients;
create policy camera_patients_member on public.camera_patients
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));
