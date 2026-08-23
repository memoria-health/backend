-- One-shot, idempotent schema delta to add the ui_events analytics table
-- to a running Supabase WITHOUT a full db reset. Companion to
-- scripts/apply_feedback_delta.sql + apply_caregiver_devices_delta.sql.
--
-- Backs the iOS caregiver app's UI-interaction analytics: the app flushes
-- batches of button/screen/toggle/lifecycle events straight into this
-- table via the authed Supabase client (no backend endpoint). RLS scopes
-- every insert/select to the caregiver who generated the events.
--
-- Every statement is guarded (IF NOT EXISTS / pg_policies check) so
-- re-running is a no-op. Apply via:
--   psql "<connection string>" -f scripts/apply_ui_events_delta.sql
-- or paste into the Supabase SQL editor.

begin;

create table if not exists public.ui_events (
  id            uuid primary key default gen_random_uuid(),
  caregiver_id  uuid not null references public.caregivers(id) on delete cascade,
  session_id    uuid not null,
  seq           bigint not null,
  event_type    text not null,
  name          text not null,
  screen        text,
  occurred_at   timestamptz not null,
  received_at   timestamptz not null default now(),
  app_build     text,
  os_version    text,
  device_model  text,
  metadata      jsonb
);

create index if not exists ui_events_session_idx
  on public.ui_events(caregiver_id, session_id, seq);

create index if not exists ui_events_occurred_idx
  on public.ui_events(occurred_at);

alter table public.ui_events enable row level security;

do $$ begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public'
      and tablename = 'ui_events'
      and policyname = 'ui_events_owner'
  ) then
    create policy ui_events_owner on public.ui_events
      for all using (caregiver_id = auth.uid()) with check (caregiver_id = auth.uid());
  end if;
end $$;

commit;
