-- One-shot, idempotent schema delta to add Pi-camera ingest support
-- to a running Supabase WITHOUT a full db reset. Apply via:
--   psql "$(supabase status -o env | grep ^DB_URL= | cut -d= -f2- | tr -d '\"')" \
--     -f scripts/apply_pi_schema_delta.sql

begin;

alter table public.cameras
  add column if not exists source_type text not null default 'iphone',
  add column if not exists pi_stream_url text,
  add column if not exists pi_audio_url text,
  add column if not exists ingest_token_hash text,
  add column if not exists ingest_token_created_at timestamptz;

do $$ begin
  if not exists (
    select 1 from pg_constraint where conname = 'cameras_source_type_check'
  ) then
    alter table public.cameras
      add constraint cameras_source_type_check
      check (source_type in ('iphone', 'pi'));
  end if;
end $$;

-- The push refactor removes the requirement that Pi rows carry a
-- pi_stream_url (backend never pulls from the Pi anymore). Drop the
-- old constraint if it exists from an earlier delta apply.
alter table public.cameras
  drop constraint if exists cameras_pi_stream_url_required;

do $$ begin
  if not exists (
    select 1 from pg_constraint where conname = 'cameras_pi_stream_url_length'
  ) then
    alter table public.cameras
      add constraint cameras_pi_stream_url_length
      check (pi_stream_url is null or length(pi_stream_url) <= 2048);
  end if;
end $$;

-- Audio URL constraint also relaxed; the source-type check is no
-- longer enforced now that pi_audio_url is vestigial.
alter table public.cameras
  drop constraint if exists cameras_pi_audio_url_source;

do $$ begin
  if not exists (
    select 1 from pg_constraint where conname = 'cameras_pi_audio_url_length'
  ) then
    alter table public.cameras
      add constraint cameras_pi_audio_url_length
      check (pi_audio_url is null or length(pi_audio_url) <= 2048);
  end if;
end $$;

-- Pi rows MUST carry an ingest_token_hash so the push endpoint can
-- authenticate them. iPhone rows MUST NOT carry one (they use LiveKit
-- session tokens). Hash is SHA-256 hex, exactly 64 chars.
do $$ begin
  if not exists (
    select 1 from pg_constraint where conname = 'cameras_ingest_token_required'
  ) then
    -- Skip the not-valid scan: we have zero source_type='pi' rows yet,
    -- so the constraint is satisfied by definition for the existing
    -- iphone rows.
    alter table public.cameras
      add constraint cameras_ingest_token_required
      check (
        (source_type = 'pi' and ingest_token_hash is not null)
        or (source_type = 'iphone' and ingest_token_hash is null)
      );
  end if;
end $$;

do $$ begin
  if not exists (
    select 1 from pg_constraint where conname = 'cameras_ingest_token_hash_length'
  ) then
    alter table public.cameras
      add constraint cameras_ingest_token_hash_length
      check (ingest_token_hash is null or length(ingest_token_hash) = 64);
  end if;
end $$;

create table if not exists public.pi_discoveries (
  id                uuid primary key default gen_random_uuid(),
  hardware_id       text not null,
  device_label      text not null,
  current_url       text not null,
  current_audio_url text,
  first_seen_at     timestamptz not null default now(),
  last_seen_at      timestamptz not null default now(),
  claimed_at        timestamptz,
  claimed_camera_id uuid references public.cameras(id) on delete set null,
  constraint pi_discoveries_url_length
    check (length(current_url) <= 2048),
  constraint pi_discoveries_audio_url_length
    check (current_audio_url is null or length(current_audio_url) <= 2048),
  unique (hardware_id)
);

alter table public.pi_discoveries
  add column if not exists pending_ingest_token text;

do $$ begin
  if not exists (
    select 1 from pg_constraint where conname = 'pi_discoveries_pending_token_length'
  ) then
    alter table public.pi_discoveries
      add constraint pi_discoveries_pending_token_length
      check (pending_ingest_token is null or length(pending_ingest_token) <= 128);
  end if;
end $$;

create index if not exists pi_discoveries_last_seen_idx
  on public.pi_discoveries(last_seen_at);
create index if not exists pi_discoveries_unclaimed_idx
  on public.pi_discoveries(last_seen_at)
  where claimed_at is null;

commit;

-- RLS (2026-08-03). Supabase grants anon/authenticated full DML on every
-- table in public and PostgREST exposes it, so RLS is the only control —
-- and anon is the PUBLISHABLE key that ships in the app binary. Enabling
-- it with no policy denies all non-owner access (fail closed); the backend
-- reaches these through the service role, which bypasses RLS.
-- Must live in THIS file: deltas are applied one at a time, so fixing it in
-- a later delta leaves a live exposure window in between.
alter table public.pi_discoveries enable row level security;
