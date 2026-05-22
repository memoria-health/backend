-- nemara-backend schema · single-file, edited in place, applied via `make db-reset`.
--
-- Per feedback_no_db_migrations_yet: pre-MVP, no migration files. The
-- entire schema lives here; `supabase db reset` re-runs it.
--
-- Layout: extensions → enums → trigger fns → tables → indexes → RLS
-- policies → Realtime publication.

-- ── extensions ───────────────────────────────────────────────────
create extension if not exists pgcrypto;       -- gen_random_uuid()


-- ══════════════════════════════════════════════════════════════════
-- ENUMS
-- ══════════════════════════════════════════════════════════════════

create type detector_category    as enum ('safety', 'routine');
create type detector_source      as enum ('preset', 'custom');
create type detector_shape       as enum ('point_in_time', 'state');
create type detector_sensitivity as enum ('low', 'standard', 'high', 'off');
create type review_choice        as enum ('correct', 'wrong', 'unsure');
create type patient_state        as enum ('calm', 'critical', 'offline');
create type relation_kind        as enum (
  'parent', 'spouse', 'grandparent', 'sibling', 'client', 'other'
);


-- ══════════════════════════════════════════════════════════════════
-- TRIGGER FUNCTIONS
-- ══════════════════════════════════════════════════════════════════

-- Auto-update `updated_at` on every UPDATE. Applied per-table by name.
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- Mirror auth.users.email → caregivers.email when Supabase confirms a
-- new address. The iOS app updates email via Supabase auth's own
-- verification flow (`client.auth.update(user: UserAttributes(email:))`);
-- this trigger keeps our copy in lock-step with the auth.users source
-- of truth without needing a backend endpoint.
create or replace function public.sync_caregiver_email_from_auth()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  -- Only fire when email actually changed (no-op for password / metadata updates).
  if new.email is distinct from old.email then
    update public.caregivers
       set email = lower(new.email)
     where id = new.id;
  end if;
  return new;
end;
$$;


-- ══════════════════════════════════════════════════════════════════
-- TABLES
-- ══════════════════════════════════════════════════════════════════

-- ── caregivers ───────────────────────────────────────────────────
-- One row per signed-up user. id == auth.users.id so RLS policies
-- pin everything to auth.uid().
create table public.caregivers (
  id           uuid primary key references auth.users(id) on delete cascade,
  display_name text not null,
  email        text not null unique,                  -- mirrored from auth.users.email
  phone        text,                                  -- optional; caregiver-editable
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create trigger caregivers_set_updated_at
  before update on public.caregivers
  for each row execute function public.set_updated_at();

-- email-sync trigger fires on auth.users; one trigger here, not per-row.
drop trigger if exists sync_caregiver_email on auth.users;
create trigger sync_caregiver_email
  after update on auth.users
  for each row execute function public.sync_caregiver_email_from_auth();


-- ── patients ─────────────────────────────────────────────────────
-- 1:N from caregiver. No care_team join — one patient has exactly
-- one caregiver. Multi-caregiver-per-patient is a future addition.
create table public.patients (
  id              uuid primary key default gen_random_uuid(),
  caregiver_id    uuid not null references public.caregivers(id) on delete cascade,
  full_name       text not null,
  preferred_name  text not null,
  date_of_birth   date,
  gender          text,
  relation        relation_kind not null,
  home_address    text,
  -- Server / wall-phone-managed (not writable from iOS — see PATCH /v1/patients).
  state           patient_state not null default 'calm',
  status_line     text not null default 'All quiet',
  last_event_at   timestamptz,
  battery_percent int  check (battery_percent between 0 and 100),
  -- Caregiver-editable.
  cameras_paused  boolean not null default false,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index patients_caregiver_id_idx on public.patients(caregiver_id);

create trigger patients_set_updated_at
  before update on public.patients
  for each row execute function public.set_updated_at();


-- ── cameras ──────────────────────────────────────────────────────
-- One row per wall-mounted phone. Rows are inserted by the caregiver
-- app's "Pair a new phone" flow (POST /v1/patients/{id}/cameras/pairing-code),
-- which mints a 6-digit `pairing_code` valid for 10 minutes. The wall
-- phone then exchanges the code via POST /v1/cameras/pair, receiving a
-- bcrypt-hashed long-lived `api_key` it uses for clip uploads and
-- LiveKit token minting.
create table public.cameras (
  id                  uuid primary key default gen_random_uuid(),
  patient_id          uuid not null references public.patients(id) on delete cascade,
  display_id          int  not null,                  -- per-patient sequence: "Cam 01"
  room                text not null,
  icon_name           text not null default 'tv',
  online              boolean not null default false,
  last_seen_at        timestamptz,
  created_at          timestamptz not null default now(),
  -- ── Pairing flow ────────────────────────────────────────────────
  -- pairing_code is set when the caregiver mints a code; cleared when
  -- the wall phone successfully exchanges it. Unique while non-null so
  -- two open pairing windows can never collide on the same code.
  pairing_code        text,                                              -- 6 digits
  pairing_expires_at  timestamptz,
  -- bcrypt hash of the long-lived API key handed to the wall phone
  -- after successful pairing. NULL until pairing completes.
  api_key_hash        text,
  -- LiveKit identity + room name minted at pairing-code creation time
  -- (so the caregiver app can show a stable identity while waiting).
  -- room = "patient_<patient_id>"; identity = "cam_<short>".
  livekit_identity    text,
  livekit_room        text,
  -- ── Detector state machine ──────────────────────────────────────
  -- Mirrors the dashboard's per-camera open/closed state. `closed`
  -- skips the LLM call until people / motion reappear. Promoted to
  -- nemara-backend so the detector worker (Phase 2) can share the
  -- same persistent state across worker restarts.
  state               text not null default 'open'
        check (state in ('open', 'closed')),
  no_people_count     int  not null default 0,
  state_changed_at    timestamptz not null default now(),
  close_threshold     int  not null default 6,        -- ticks of no people + no motion → closed
  battery_percent     int  check (battery_percent between 0 and 100),
  unique (patient_id, display_id)
);

create index cameras_patient_id_idx on public.cameras(patient_id);

-- Only one camera can hold a given pairing_code at a time (collision
-- avoidance for the 10-min pairing window).
create unique index cameras_pairing_code_uniq
  on public.cameras(pairing_code)
  where pairing_code is not null;


-- ── detectors ────────────────────────────────────────────────────
-- Mix of preset (created by the seed-on-patient-creation service) and
-- custom (created by the caregiver via POST /v1/patients/{id}/detectors).
--
-- The "runtime contract" columns (identifier, prompt_fragment,
-- schema_json, always_on, is_builtin) feed the detector pipeline. The
-- prompt fragments are duplicated across patients (one row per
-- patient × detector) — at the current scale (small N), a bulk UPDATE
-- to retune a prompt is cheap and avoids a parallel "definitions"
-- table per the plan's "extend, don't add" directive.
create table public.detectors (
  id              uuid primary key default gen_random_uuid(),
  patient_id      uuid not null references public.patients(id) on delete cascade,
  name            text not null,
  icon_name       text not null,
  description     text not null,
  category        detector_category    not null,
  source          detector_source      not null,
  shape           detector_shape       not null,
  where_scope     text not null,
  when_scope      text not null,
  sensitivity     detector_sensitivity not null default 'standard',
  is_on           boolean not null default true,
  custom_prompt   text,                              -- only meaningful when source = 'custom'
  -- ── Runtime contract (detector pipeline reads these) ────────────
  -- Stable identifier the LLM JSON-schema reference. For presets:
  -- 'people_present' | 'fall' | 'wandering'. For custom detectors:
  -- 'custom_<short>' generated server-side on POST.
  identifier      text,
  prompt_fragment text,
  schema_json     jsonb,
  -- people_present is always_on=true (included even if is_on=false)
  -- because the state machine requires its output.
  always_on       boolean not null default false,
  is_builtin      boolean not null default true,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index detectors_patient_id_idx on public.detectors(patient_id);

-- A patient can't have two detectors with the same identifier
-- ("fall" appears at most once per patient).
create unique index detectors_patient_identifier_uniq
  on public.detectors(patient_id, identifier)
  where identifier is not null;

create trigger detectors_set_updated_at
  before update on public.detectors
  for each row execute function public.set_updated_at();


-- ── events ───────────────────────────────────────────────────────
-- Rows are written by wall phones (a later plan). For v0, populated
-- by the seed script. Caregiver app only reads + PATCHes them.
create table public.events (
  id            uuid primary key default gen_random_uuid(),
  patient_id    uuid not null references public.patients(id) on delete cascade,
  detector_id   uuid references public.detectors(id) on delete set null,
  camera_id     uuid references public.cameras(id)   on delete set null,
  category      detector_category not null,
  source        detector_source   not null,
  icon_name     text not null,
  title         text not null,
  location      text not null,
  occurred_at   timestamptz not null,
  has_clip      boolean not null default false,
  narrative     text not null default '',
  opened        boolean not null default false,
  reviewed      boolean not null default false,
  review_choice review_choice,
  review_notes  text,
  reviewed_at   timestamptz,
  created_at    timestamptz not null default now()
);

-- Compound index supports the Events tab's "newest first" listing and
-- cursor pagination on `(occurred_at desc, id)`.
create index events_patient_occurred_at_idx
  on public.events(patient_id, occurred_at desc, id);


-- ── analyses ─────────────────────────────────────────────────────
-- Per-tick LLM output from the detector pipeline. One row per ~10s
-- per camera. Not user-facing (events are the user-facing layer);
-- consumed by the detector worker (state machine, active-camera
-- resolution for Talk) and Supabase Studio for debugging. Not added
-- to the Realtime publication — too noisy for the caregiver app.
create table public.analyses (
  id              uuid primary key default gen_random_uuid(),
  caregiver_id    uuid not null references public.caregivers(id) on delete cascade,
  patient_id      uuid not null references public.patients(id)   on delete cascade,
  camera_id       uuid not null references public.cameras(id)    on delete cascade,
  -- `skipped` is the closed-state short-circuit (no LLM was called).
  -- The row is still persisted so Studio + the detector worker (Phase 2)
  -- can see the tick. See `src/api/clips.py`.
  provider        text not null check (provider in ('gemini', 'openai', 'skipped')),
  description     text not null,
  detector_fields jsonb not null default '{}'::jsonb,
  people_present  boolean not null,
  motion_in_clip  boolean not null,
  state_after     text not null check (state_after in ('open', 'closed')),
  captured_at     timestamptz not null,
  created_at      timestamptz not null default now()
);

-- Newest-first per camera (active-camera resolution + debugging UIs).
create index analyses_camera_created_idx
  on public.analyses(camera_id, created_at desc);
-- Patient-scoped active-camera resolution scans by patient + recency.
create index analyses_patient_captured_idx
  on public.analyses(patient_id, captured_at desc);


-- ══════════════════════════════════════════════════════════════════
-- ROW-LEVEL SECURITY
-- ══════════════════════════════════════════════════════════════════
-- Every table is RLS-on. The FastAPI repo layer also filters by
-- caregiver_id (defense in depth) but RLS is the load-bearing check
-- that lets the iOS Supabase Realtime subscription only see the
-- caregiver's own rows. Without it, Realtime would broadcast every
-- table change to every subscriber.

alter table public.caregivers enable row level security;
create policy caregivers_self on public.caregivers
  for all using (id = auth.uid()) with check (id = auth.uid());

alter table public.patients enable row level security;
create policy patients_owner on public.patients
  for all using (caregiver_id = auth.uid()) with check (caregiver_id = auth.uid());

alter table public.cameras enable row level security;
create policy cameras_owner on public.cameras
  for all using (
    patient_id in (select id from public.patients where caregiver_id = auth.uid())
  ) with check (
    patient_id in (select id from public.patients where caregiver_id = auth.uid())
  );

alter table public.detectors enable row level security;
create policy detectors_owner on public.detectors
  for all using (
    patient_id in (select id from public.patients where caregiver_id = auth.uid())
  ) with check (
    patient_id in (select id from public.patients where caregiver_id = auth.uid())
  );

alter table public.events enable row level security;
create policy events_owner on public.events
  for all using (
    patient_id in (select id from public.patients where caregiver_id = auth.uid())
  ) with check (
    patient_id in (select id from public.patients where caregiver_id = auth.uid())
  );

alter table public.analyses enable row level security;
create policy analyses_owner on public.analyses
  for all using (caregiver_id = auth.uid()) with check (caregiver_id = auth.uid());


-- ══════════════════════════════════════════════════════════════════
-- REALTIME PUBLICATION
-- ══════════════════════════════════════════════════════════════════
-- iOS subscribes to these tables via the Supabase Swift SDK's
-- `RealtimeChannel`. RLS policies above filter rows automatically — a
-- caregiver only sees changes for their own patients (+ that patient's
-- cameras / detectors / events).
--
-- `caregivers` is intentionally NOT published — a caregiver doesn't
-- subscribe to changes on their own row.
--
-- REPLICA IDENTITY FULL is required so UPDATE/DELETE events carry the
-- full old row in the broadcast payload (default REPLICA IDENTITY only
-- includes the primary key, which strands the iOS-side cache mutation
-- logic without enough data to find the matching local row).

alter publication supabase_realtime add table
  public.patients,
  public.cameras,
  public.detectors,
  public.events;

alter table public.patients   replica identity full;
alter table public.cameras    replica identity full;
alter table public.detectors  replica identity full;
alter table public.events     replica identity full;
