-- Review-gate (additive, backward compatible; default OFF everywhere)
alter table public.patients add column if not exists require_review boolean not null default false;
alter table public.events   add column if not exists review_gate text; -- null|pending|approved|rejected|auto
create index if not exists events_review_pending on public.events (patient_id) where review_gate = 'pending';
alter table public.events add column if not exists review_clip_key text;
create table if not exists public.review_devices (
  apns_token text primary key, user_id uuid not null,
  environment text not null default 'sandbox', created_at timestamptz not null default now());
