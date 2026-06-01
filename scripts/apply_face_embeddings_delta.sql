-- One-shot, idempotent schema delta to add the Re-ID face-embeddings
-- table to a running Supabase WITHOUT a full db reset. Companion to
-- scripts/apply_pi_schema_delta.sql; together they bring an older prod
-- schema up to main's supabase/schema.sql. Apply via:
--   psql "$(supabase status -o env | grep ^DB_URL= | cut -d= -f2- | tr -d '\"')" \
--     -f scripts/apply_face_embeddings_delta.sql
-- (or `railway run psql "$NEMARA_DB_URL" -f scripts/apply_face_embeddings_delta.sql`
--  once psql is available; prod has no source_type/face tables until this runs.)

begin;

create table if not exists public.patient_face_embeddings (
  id              uuid primary key default gen_random_uuid(),
  patient_id      uuid not null references public.patients(id) on delete cascade,
  embedding       bytea not null,
  det_score       real not null,
  source          text not null check (source in ('selfie', 'harvested')),
  source_path     text,
  status          text not null default 'ready'
                  check (status in ('processing', 'ready', 'failed')),
  failure_reason  text,
  created_at      timestamptz not null default now()
);

create index if not exists patient_face_embeddings_patient_ready_idx
  on public.patient_face_embeddings(patient_id)
  where status = 'ready';

create index if not exists patient_face_embeddings_patient_harvested_idx
  on public.patient_face_embeddings(patient_id, created_at)
  where source = 'harvested' and status = 'ready';

-- RLS: rows visible only to the caregiver who owns the patient
-- (matches schema.sql's patient_face_embeddings_owner policy).
alter table public.patient_face_embeddings enable row level security;

do $$ begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'patient_face_embeddings'
      and policyname = 'patient_face_embeddings_owner'
  ) then
    create policy patient_face_embeddings_owner on public.patient_face_embeddings
      for all using (
        patient_id in (select id from public.patients where caregiver_id = auth.uid())
      ) with check (
        patient_id in (select id from public.patients where caregiver_id = auth.uid())
      );
  end if;
end $$;

commit;
