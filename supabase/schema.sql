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
  -- Bucket-relative key in `patient-photos` (e.g. "{patient_id}.jpg").
  -- NULL until the caregiver uploads an avatar; the roster + header
  -- fall back to initials while NULL. The signed URL is minted at
  -- read time by GET /v1/patients/{id}/photo-url.
  photo_storage_key text,
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
  -- ── User-facing "off" toggle ─────────────────────────────────────
  -- Per-camera disable controlled from the caregiver app / web
  -- dashboard. The LiveKit subscriber-bot stays in the room and the
  -- frame buffer keeps filling, but the per-tick loop in
  -- `services/detection/camera_ingest.py:_run_tick` short-circuits
  -- BEFORE the LLM call when this is true (or when the patient's
  -- `cameras_paused` is true). Caregiver-web hides disabled cameras
  -- from the live grid + analyses feed.
  disabled            boolean not null default false,
  -- ── Desired connection state (privacy / battery / cost) ─────────
  -- Stronger than `disabled`: when 'disconnected', the wall phone
  -- itself LEAVES the LiveKit room. No video bytes are encoded,
  -- uploaded, or seen by LiveKit Cloud. The phone polls
  -- /v1/cameras/me/desired-state every 2s and reconciles by calling
  -- room.connect() / room.disconnect() to match. Reconnect latency
  -- ~ 2-5s after the caregiver flips this back to 'connected'.
  -- See NemaraCam/.../DesiredStateMonitor.swift.
  desired_state       text not null default 'connected'
    check (desired_state in ('connected', 'disconnected')),
  -- ── Camera source type (iPhone vs Pi ingest path) ───────────────
  -- 'iphone': wall phone running the NemaraCam app. Publishes
  --   directly to LiveKit, motion detection runs on-device, audio
  --   co-streamed via LiveKit. This is the only source type the
  --   pairing flow supports for v0.
  -- 'pi': Raspberry Pi running the MJPEG-over-HTTP streaming
  --   daemon. The backend's pi_ingest worker process pulls frames,
  --   runs motion detection backend-side, and re-publishes to
  --   LiveKit as a cam_* participant so the rest of the system
  --   (caregiver viewer, detector tick, clip archival) treats the
  --   Pi camera identically to an iPhone camera. See
  --   src/services/ingest/pi/ for the worker.
  -- Default 'iphone' makes this backwards-compatible with existing
  -- rows; no data migration required.
  source_type         text not null default 'iphone'
    check (source_type in ('iphone', 'pi')),
  -- Required when source_type='pi'. The HTTP MJPEG endpoint the
  -- pi_ingest worker connects to (e.g. http://10.0.0.44:8000/stream.mjpg).
  -- NULL is enforced for source_type='iphone' to keep the schema
  -- self-consistent. Length cap of 2048 matches the Pydantic
  -- boundary so a row that survives a write can always be read.
  pi_stream_url       text,
  constraint cameras_pi_stream_url_required
    check (
      (source_type = 'pi' and pi_stream_url is not null)
      or (source_type = 'iphone' and pi_stream_url is null)
    ),
  constraint cameras_pi_stream_url_length
    check (pi_stream_url is null or length(pi_stream_url) <= 2048),
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


-- ── cameras_detectors (per-camera detector assignment) ──────────────
-- Many-to-many between cameras and detectors. Lets caregivers turn a
-- detector ON for some cameras and OFF for others — e.g., "wandering"
-- only on the hallway camera, "medication" only on the kitchen camera.
--
-- Before this table existed, every detector with `is_on=true` ran on
-- every camera under the patient. To preserve that behavior on
-- existing data, the seed below inserts (camera, detector) pairs for
-- every existing combination; new detectors auto-assign to every
-- camera of their patient (and vice-versa) via the API layer.
--
-- The runtime contract:
--   - `detectors.is_on=false`  → never runs anywhere (master kill switch)
--   - `detectors.always_on=true` → runs everywhere regardless of this table
--                                  (people_present today; not stored as a row
--                                   so doesn't appear here)
--   - row in this table        → detector runs on that specific camera
--   - no row                   → detector does NOT run on that camera
create table public.cameras_detectors (
  camera_id   uuid not null references public.cameras(id)   on delete cascade,
  detector_id uuid not null references public.detectors(id) on delete cascade,
  created_at  timestamptz not null default now(),
  primary key (camera_id, detector_id)
);

create index cameras_detectors_camera_id_idx   on public.cameras_detectors(camera_id);
create index cameras_detectors_detector_id_idx on public.cameras_detectors(detector_id);


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
  -- Bucket-relative key in `incident-clips` (e.g.
  -- "events/{event_id}.mp4"). NULL until the post-event capture +
  -- concat completes; `has_clip` flips to TRUE in the same UPDATE.
  -- Read at iOS-fetch time by `GET /v1/events/{id}/clip-url`, which
  -- mints a short-lived Supabase Storage signed URL.
  clip_storage_key text,
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
  created_at      timestamptz not null default now(),
  -- Idempotency guard: at most one analysis per camera per moment.
  -- The in-process per-camera asyncio.Lock prevents overlap on a
  -- live worker, but a worker restart at the wrong moment can
  -- otherwise produce a duplicate row for the same tick. This
  -- constraint is belt-and-braces: the second INSERT fails cleanly
  -- with sqlalchemy.exc.IntegrityError, which the tick body's
  -- exception handler swallows.
  constraint analyses_camera_captured_at_uniq unique (camera_id, captured_at)
);

-- Newest-first per camera (active-camera resolution + debugging UIs).
create index analyses_camera_created_idx
  on public.analyses(camera_id, created_at desc);
-- Patient-scoped active-camera resolution scans by patient + recency.
create index analyses_patient_captured_idx
  on public.analyses(patient_id, captured_at desc);


-- ── caregiver_devices ────────────────────────────────────────────
-- One row per APNs device token the caregiver has registered. A
-- caregiver may have multiple tokens (phone + iPad + a re-installed
-- app on the same phone) — `(caregiver_id)` is NOT unique. The token
-- itself IS unique so that a re-registration from the same device
-- (token regenerated post-install) idempotently replaces the prior row.
create table public.caregiver_devices (
  id              uuid primary key default gen_random_uuid(),
  caregiver_id    uuid not null references public.caregivers(id) on delete cascade,
  apns_token      text not null,
  bundle_id       text not null,
  -- "sandbox" for Xcode-built debug builds, "production" for TestFlight /
  -- App Store. The push dispatcher uses this to pick the right APNs
  -- endpoint (`api.sandbox.push.apple.com` vs `api.push.apple.com`).
  environment     text not null check (environment in ('sandbox','production')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);
create unique index caregiver_devices_token_uniq on public.caregiver_devices(apns_token);
create index caregiver_devices_caregiver_idx on public.caregiver_devices(caregiver_id);

create trigger caregiver_devices_set_updated_at
  before update on public.caregiver_devices
  for each row execute function public.set_updated_at();


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

-- cameras_detectors: scoped through the camera's patient → caregiver.
-- Joining via cameras (not detectors) is fine — both FKs point to the
-- same patient by construction, and going through cameras matches the
-- pattern used by `cameras_owner` above.
alter table public.cameras_detectors enable row level security;
create policy cameras_detectors_owner on public.cameras_detectors
  for all using (
    camera_id in (
      select c.id from public.cameras c
      join public.patients p on p.id = c.patient_id
      where p.caregiver_id = auth.uid()
    )
  ) with check (
    camera_id in (
      select c.id from public.cameras c
      join public.patients p on p.id = c.patient_id
      where p.caregiver_id = auth.uid()
    )
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

alter table public.caregiver_devices enable row level security;
create policy caregiver_devices_owner on public.caregiver_devices
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
  public.cameras_detectors,
  public.events,
  public.analyses;

alter table public.patients          replica identity full;
alter table public.cameras           replica identity full;
alter table public.detectors         replica identity full;
alter table public.cameras_detectors replica identity full;
alter table public.events            replica identity full;
alter table public.analyses          replica identity full;
