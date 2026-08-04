-- Review-gate (additive, backward compatible; default OFF everywhere)
alter table public.patients add column if not exists require_review boolean not null default false;
alter table public.events   add column if not exists review_gate text; -- null|pending|approved|rejected|auto
create index if not exists events_review_pending on public.events (patient_id) where review_gate = 'pending';
alter table public.events add column if not exists review_clip_key text;
create table if not exists public.review_devices (
  apns_token text primary key, user_id uuid not null,
  environment text not null default 'sandbox', created_at timestamptz not null default now());

-- RLS (2026-08-03). Supabase grants anon/authenticated full DML on every
-- table in public and PostgREST exposes it, so RLS is the only control —
-- and anon is the PUBLISHABLE key that ships in the app binary. Enabling
-- it with no policy denies all non-owner access (fail closed); the backend
-- reaches these through the service role, which bypasses RLS.
-- Must live in THIS file: deltas are applied one at a time, so fixing it in
-- a later delta leaves a live exposure window in between.
alter table public.review_devices enable row level security;
