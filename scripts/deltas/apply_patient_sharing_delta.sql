-- Patient sharing delta — apply to an EXISTING database (local dev + prod).
-- Introduces patient_members + has_patient_access() and swaps the patient-scoped
-- RLS policies from ownership → membership. Idempotent (safe to re-run).
--
-- Local:  psql "postgresql://postgres:postgres@127.0.0.1:54342/postgres" -f scripts/deltas/apply_patient_sharing_delta.sql
-- Prod:   via Supabase Management API (never reset prod).

begin;

-- 1. Membership table + indexes -----------------------------------------------
create table if not exists public.patient_members (
  id           uuid primary key default gen_random_uuid(),
  patient_id   uuid not null references public.patients(id)   on delete cascade,
  caregiver_id uuid not null references public.caregivers(id) on delete cascade,
  role         text not null default 'caregiver'
                 check (role in ('owner','caregiver','viewer')),
  invited_by   uuid references public.caregivers(id) on delete set null,
  created_at   timestamptz not null default now(),
  unique (patient_id, caregiver_id)
);
create index if not exists patient_members_caregiver_idx on public.patient_members(caregiver_id);
create index if not exists patient_members_patient_idx   on public.patient_members(patient_id);

-- 2. Backfill: existing owner → 'owner' membership row -------------------------
insert into public.patient_members (patient_id, caregiver_id, role)
select id, caregiver_id, 'owner' from public.patients
on conflict (patient_id, caregiver_id) do nothing;

-- 3. Access helper ------------------------------------------------------------
create or replace function public.has_patient_access(pid uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.patient_members m
    where m.patient_id = pid and m.caregiver_id = auth.uid()
  );
$$;

-- 4. Swap ownership policies → membership policies ----------------------------
drop policy if exists patients_owner                  on public.patients;
drop policy if exists cameras_owner                   on public.cameras;
drop policy if exists detectors_owner                 on public.detectors;
drop policy if exists cameras_detectors_owner         on public.cameras_detectors;
drop policy if exists events_owner                    on public.events;
drop policy if exists patient_face_embeddings_owner   on public.patient_face_embeddings;
drop policy if exists analyses_owner                  on public.analyses;

alter table public.patient_members enable row level security;
drop policy if exists patient_members_read on public.patient_members;
create policy patient_members_read on public.patient_members
  for select using (public.has_patient_access(patient_id));

drop policy if exists patients_member on public.patients;
create policy patients_member on public.patients
  for all using (public.has_patient_access(id)) with check (caregiver_id = auth.uid());

drop policy if exists cameras_member on public.cameras;
create policy cameras_member on public.cameras
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

drop policy if exists detectors_member on public.detectors;
create policy detectors_member on public.detectors
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

drop policy if exists cameras_detectors_member on public.cameras_detectors;
create policy cameras_detectors_member on public.cameras_detectors
  for all using (
    camera_id in (select c.id from public.cameras c where public.has_patient_access(c.patient_id))
  ) with check (
    camera_id in (select c.id from public.cameras c where public.has_patient_access(c.patient_id))
  );

drop policy if exists events_member on public.events;
create policy events_member on public.events
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

drop policy if exists patient_face_embeddings_member on public.patient_face_embeddings;
create policy patient_face_embeddings_member on public.patient_face_embeddings
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

drop policy if exists analyses_member on public.analyses;
create policy analyses_member on public.analyses
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

commit;
