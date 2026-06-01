-- One-shot, idempotent schema delta to add the APNs push-token table to
-- a running Supabase WITHOUT a full db reset. Companion to
-- scripts/apply_pi_schema_delta.sql + scripts/apply_face_embeddings_delta.sql.
--
-- caregiver_devices predates the pi-ingest merge, so a prod DB that already
-- has source_type almost certainly already has this table — in which case
-- every statement below is a no-op. Run it anyway; it is safe either way.
-- Apply via:
--   psql "<prod connection string>" -f scripts/apply_caregiver_devices_delta.sql
-- or paste into the Supabase SQL editor (prod project).

begin;

create table if not exists public.caregiver_devices (
  id              uuid primary key default gen_random_uuid(),
  caregiver_id    uuid not null references public.caregivers(id) on delete cascade,
  apns_token      text not null,
  bundle_id       text not null,
  -- "sandbox" for Xcode-built debug builds, "production" for TestFlight /
  -- App Store. The push dispatcher uses this to pick the right APNs endpoint.
  environment     text not null check (environment in ('sandbox','production')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create unique index if not exists caregiver_devices_token_uniq
  on public.caregiver_devices(apns_token);
create index if not exists caregiver_devices_caregiver_idx
  on public.caregiver_devices(caregiver_id);

alter table public.caregiver_devices enable row level security;

do $$ begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'caregiver_devices'
      and policyname = 'caregiver_devices_owner'
  ) then
    create policy caregiver_devices_owner on public.caregiver_devices
      for all using (caregiver_id = auth.uid()) with check (caregiver_id = auth.uid());
  end if;
end $$;

commit;
