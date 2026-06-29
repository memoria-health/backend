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
-- 'scheduled' is the absence-style detector (e.g. "medication not
-- taken by 5pm"): the scheduler evaluates a daily/weekly window at a
-- deadline and fires if the signal was never seen. 'state' stays in
-- the enum for the agitation preset but is no longer offered in the
-- create UI; everything reactive is point_in_time.
create type detector_shape       as enum ('point_in_time', 'state', 'scheduled');
create type detector_sensitivity as enum ('low', 'standard', 'high');
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
  -- Per-account flag for the web onboarding walkthrough. False until
  -- the caregiver dismisses or completes the modal on the web
  -- dashboard; PATCH /v1/me flips it true. iOS gets its own column
  -- when we ship the iOS onboarding (deliberately separate so a user
  -- who learned the web flow still sees the iOS walkthrough on first
  -- use of the mobile app, and vice versa).
  onboarded_web boolean not null default false,
  -- IANA tz name (e.g. "America/New_York"). Scheduled detectors
  -- evaluate their window in the caregiver's local time. The web
  -- client auto-detects browser tz and PATCHes /v1/me; this default
  -- is the fallback for accounts that never loaded the web app.
  timezone     text not null default 'America/New_York',
  -- Per-account VLM override: 'gemini' | 'openai' | 'openrouter'.
  -- NULL = use the server's default_vlm_provider. Lets us move a single
  -- family onto a different model (e.g. Qwen3-VL via OpenRouter) by
  -- flipping this column — no deploy. Read on the detection hot path.
  vlm_provider text,
  -- Per-account toggle for the dashboard's "Live analyses" feed. Default
  -- true (everyone sees it); flip false to hide it for specific accounts
  -- (e.g. demo logins). Web reads it from /v1/me and gates the section.
  show_live_analyses boolean not null default true,
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
  -- Caregiver-editable.
  cameras_paused  boolean not null default false,
  -- Per-patient email alert prefs. Push notifications cover the iOS
  -- app; email_alerts_enabled is the master on/off for THIS patient,
  -- email_alert_recipients is additional recipients beyond the owning
  -- caregiver's account email (which always receives this patient's
  -- alerts when enabled). JSON array of strings.
  email_alerts_enabled    boolean not null default false,
  email_alert_recipients  jsonb   not null default '[]'::jsonb,
  -- Data-collection consent (research opt-in). `data_collection_enabled`
  -- is the master switch to store short video+audio clips of this
  -- patient for research, only when someone is present; default OFF
  -- until the caregiver explicitly opts in. `data_collection_decided_at`
  -- records WHEN the caregiver decided: NULL = never asked, which drives
  -- the first-open consent banner (web + iOS). Both "Share" and "Not
  -- now" stamp it so the banner never re-nags.
  data_collection_enabled    boolean not null default false,
  data_collection_decided_at timestamptz,
  -- Privacy mode (dignity override). While ON it overrides data
  -- collection (no clips stored), severs live video+audio to caregivers
  -- at the SFU, and strips event clips from insights — detection + text
  -- insights keep flowing. A live switch (no decided_at): it SUSPENDS
  -- data_collection_enabled without mutating it. Enforced server-side.
  privacy_mode_enabled       boolean not null default false,
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
  -- ── AI paused (LLM-only pause, video keeps streaming) ────────────
  -- Lighter than `disabled`: the tick still short-circuits before
  -- the LLM call (synthetic "skipped" row with skipped_reason='ai_paused')
  -- but the tile STAYS VISIBLE in the live grid and video keeps
  -- streaming. Lets the caregiver watch a room without paying for
  -- Gemini calls.
  ai_paused           boolean not null default false,
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
  -- Vestigial: was the URL the backend pulled MJPEG from when Pis
  -- served their stream. Push refactor (Pi POSTs frames outbound)
  -- made it unused but column is kept nullable so historical /
  -- diagnostic queries still work. NOT referenced by the relay path.
  pi_stream_url       text,
  -- Vestigial: was the URL the backend pulled PCM audio from in the
  -- pull-mode design. Same status as pi_stream_url after the push
  -- refactor; nullable, unused by the relay path.
  pi_audio_url        text,
  -- Bearer token (hashed) that the Pi presents on POST /v1/cameras/me/frames
  -- when pushing its MJPEG stream up. Generated at provision time,
  -- returned ONCE in plaintext to the Pi via the announce response,
  -- and stored only as a SHA-256 hex digest server-side. Required for
  -- source_type='pi' (Pis are the only callers of the push ingest
  -- endpoint), NULL for source_type='iphone' (iPhones publish to
  -- LiveKit directly with their own per-session tokens).
  ingest_token_hash       text,
  ingest_token_created_at timestamptz,
  -- ── Memoria Sensor app-present heartbeat (iPhone only) ───────────
  -- Answers: is the Memoria Sensor app actually RUNNING on this
  -- camera's phone right now? Distinct from `online` (publishing) and
  -- `desired_state` (caregiver intent). An iPhone sensor polls
  -- GET /v1/cameras/me/desired-state every 1s the whole time the app
  -- is open; the backend's PresenceSweeper records that heartbeat
  -- in-memory and flips this column on transitions. false = app
  -- closed / force-quit / phone off → caregiver must OPEN the app
  -- (Reconnect can't help, the phone isn't polling). true = app alive
  -- (so "Disconnected" means reconnectable). Default true so a live
  -- sensor never flashes "closed" on rollout. ONLY meaningful for
  -- source_type='iphone' — Pi cameras never poll this endpoint, so the
  -- sweeper never writes them and they stay true (a Pi has no app to
  -- close). See services/detection/presence_sweeper.py.
  app_present             boolean not null default true,
  constraint cameras_pi_stream_url_length
    check (pi_stream_url is null or length(pi_stream_url) <= 2048),
  constraint cameras_pi_audio_url_length
    check (pi_audio_url is null or length(pi_audio_url) <= 2048),
  constraint cameras_ingest_token_required
    check (
      (source_type = 'pi' and ingest_token_hash is not null)
      or (source_type = 'iphone' and ingest_token_hash is null)
    ),
  constraint cameras_ingest_token_hash_length
    check (ingest_token_hash is null or length(ingest_token_hash) = 64),
  unique (patient_id, display_id)
);

create index cameras_patient_id_idx on public.cameras(patient_id);

-- Only one camera can hold a given pairing_code at a time (collision
-- avoidance for the 10-min pairing window).
create unique index cameras_pairing_code_uniq
  on public.cameras(pairing_code)
  where pairing_code is not null;


-- ── pi_discoveries ───────────────────────────────────────────────
-- Holds "I am a Nemara camera at <url>" announcements from Pi
-- devices that have joined home WiFi but have not yet been paired
-- to a patient. Pairing happens when a caregiver picks a discovery
-- from the "Add a camera" screen, which triggers
-- POST /v1/cameras/pi-provision {discovery_id} and inserts a row
-- in `cameras` referencing this discovery.
--
-- One row per device, keyed by hardware_id (stable MAC address).
-- A device that re-announces just updates `current_url` and
-- `last_seen_at`. The caregiver-facing list endpoint filters to
-- recently-seen unclaimed rows so stale devices fall off
-- automatically.
--
-- Security note: there is intentionally NO authentication on the
-- announce path. The Pi has no credentials before pairing. This is
-- acceptable for v1 because BLE-based provisioning (Phase 4.5b)
-- requires physical proximity to set up the device, and our
-- customer scale is low enough that cross-customer announcement
-- collisions are extremely unlikely. v2 will add a per-device
-- secret negotiated over BLE that the announcement carries.
create table public.pi_discoveries (
  id                uuid primary key default gen_random_uuid(),
  -- Stable hardware identifier (the WiFi NIC's MAC address). Used
  -- both as the de-dupe key (one record per device) and as the
  -- public-facing "ID" the caregiver picks from in the UI.
  hardware_id       text not null,
  -- Human-readable label the Pi sends (e.g. "Nemara Camera A4B7C2"
  -- with the last 6 of MAC). Surfaces in the caregiver's pick list
  -- so they can match a discovered record to the physical device.
  device_label      text not null,
  -- Current MJPEG endpoint. Updated on every re-announce; a Pi that
  -- gets a new DHCP lease pushes a new URL within ~60 s.
  current_url       text not null,
  -- Optional audio endpoint, same shape contract as cameras.pi_audio_url.
  current_audio_url text,
  first_seen_at     timestamptz not null default now(),
  last_seen_at      timestamptz not null default now(),
  -- When a caregiver pairs this discovery to a camera row, both
  -- fields are populated. `claimed_camera_id` is the FK; nulled on
  -- camera delete so the discovery can be re-paired without manual
  -- DB intervention.
  claimed_at        timestamptz,
  claimed_camera_id uuid references public.cameras(id) on delete set null,
  -- One-shot plaintext ingest bearer token for the Pi to pick up on
  -- its next announce response. See src/db/pi_discovery.py for the
  -- lifecycle rule (set on provision, nulled on first read).
  pending_ingest_token text,
  -- Length caps mirror cameras.pi_stream_url for consistency.
  constraint pi_discoveries_url_length
    check (length(current_url) <= 2048),
  constraint pi_discoveries_audio_url_length
    check (current_audio_url is null or length(current_audio_url) <= 2048),
  constraint pi_discoveries_pending_token_length
    check (pending_ingest_token is null or length(pending_ingest_token) <= 128),
  unique (hardware_id)
);

create index pi_discoveries_last_seen_idx
  on public.pi_discoveries(last_seen_at);
-- Index the "unclaimed list" query path: GET /v1/cameras/discovered
-- filters `WHERE claimed_at IS NULL AND last_seen_at > now() - 5 min`.
create index pi_discoveries_unclaimed_idx
  on public.pi_discoveries(last_seen_at)
  where claimed_at is null;


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
  -- Scheduled-detector config (null for reactive detectors). Shape:
  --   { "cadence": "daily"|"weekly",
  --     "days_of_week": [0..6],     -- weekly only; 0=Mon..6=Sun
  --     "window_start": "HH:MM",     -- local time
  --     "window_end":   "HH:MM",
  --     "deadline":     "HH:MM" }    -- when to evaluate + fire
  -- Read by the scheduler runtime (separate from the per-tick path).
  schedule_config            jsonb,
  -- Continuous ("state") detectors: seconds the activity must be absent
  -- before a resumption counts as a NEW event (rising-edge episode gap).
  -- Null for point_in_time / scheduled. LLM-inferred at creation.
  episode_gap_seconds        integer,
  -- Single-fire guard for the scheduled runtime: the date the
  -- detector last evaluated its window. The 60s scheduler loop
  -- skips a detector if today's date matches this; NULL = never
  -- evaluated.
  last_scheduled_eval_date   date,
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


-- ── patient_face_embeddings (Re-ID gallery) ─────────────────────────
-- One row per face embedding used for patient identification at tick time.
-- Two kinds of rows, distinguished by `source`:
--
--   'selfie'    Uploaded by the caregiver from the iOS onboarding flow
--               (10-15 angle-prompted captures). One row per uploaded
--               photo that passed SCRFD's face detection threshold.
--               `source_path` is the Supabase Storage object key.
--
--   'harvested' Inserted automatically when Re-ID locks a track with
--               very high confidence (cosine >= 0.80). Closes the
--               selfie-to-wall-cam domain gap over time. `source_path`
--               is NULL — we don't keep the source frame, just the
--               aligned ArcFace embedding.
--
-- Status field exists because selfie processing is async (background
-- task downloads + runs SCRFD + ArcFace). Harvested rows are written
-- inline with status='ready'.
create table public.patient_face_embeddings (
  id              uuid primary key default gen_random_uuid(),
  patient_id      uuid not null references public.patients(id) on delete cascade,
  -- 512-dim float32 ArcFace embedding, L2-normalized. Stored as bytea
  -- (~2KB per row); small enough to scan in-Python without pgvector.
  embedding       bytea not null,
  -- SCRFD detection confidence at the moment the embedding was extracted.
  -- Surfaced for diagnostics; not used as a gate at match time (the
  -- adaptive threshold uses it internally instead).
  det_score       real not null,
  source          text not null check (source in ('selfie', 'harvested')),
  source_path     text,                              -- Storage object key; selfies only
  status          text not null default 'ready'
                  check (status in ('processing', 'ready', 'failed')),
  failure_reason  text,
  created_at      timestamptz not null default now()
);

-- Hot-path index: gallery lookups always filter to ready rows for one patient.
create index patient_face_embeddings_patient_ready_idx
  on public.patient_face_embeddings(patient_id)
  where status = 'ready';

-- Used by the FIFO-evict-harvested logic when the per-patient gallery
-- cap is exceeded. Partial index keeps it tight (selfies never evict).
create index patient_face_embeddings_patient_harvested_idx
  on public.patient_face_embeddings(patient_id, created_at)
  where source = 'harvested' and status = 'ready';


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
  -- True when the clip was intentionally not archived because the patient
  -- was in privacy mode. Lets the apps render "No clip — Privacy mode"
  -- instead of an endless "Generating clip…" for has_clip=false events.
  clip_suppressed boolean not null default false,
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
  provider        text not null check (provider in ('gemini', 'openai', 'openrouter', 'skipped')),
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


-- ── feedback ──────────────────────────────────────────────────────
-- Caregiver-submitted product feedback from the iOS app's Settings →
-- Send feedback sheet (POST /v1/feedback). Storing the row is the
-- source of truth; the handler also fires a best-effort founder email
-- whose failure does NOT roll back the row. All free-text fields are
-- nullable — the API requires at least a rating OR one non-blank field.
create table public.feedback (
  id                uuid primary key default gen_random_uuid(),
  caregiver_id      uuid not null references public.caregivers(id) on delete cascade,
  rating            smallint,
  whats_working     text,
  whats_frustrating text,
  category          text,
  created_at        timestamptz not null default now()
);
create index feedback_caregiver_idx on public.feedback(caregiver_id);


-- ── ui_events ─────────────────────────────────────────────────────
-- Raw UI-interaction analytics from the iOS caregiver app: one row per
-- button tap, screen view, toggle, and app-lifecycle transition. The app
-- buffers events in memory + on disk and flushes batches directly into
-- this table via the authed Supabase client (no backend endpoint) — RLS
-- scopes every insert/select to the caregiver who generated it.
--
-- This is a write-mostly append log used for product analysis (which
-- buttons users press, in what order, with what timing). Logging is a
-- synchronous, non-throwing, fire-and-forget side effect in the app, so a
-- failed flush never affects app behavior — events just retry next flush.
--
--   session_id  — one UUID per app launch; groups a session's events.
--   seq         — monotonic per-session counter; gives EXACT ordering even
--                 when two occurred_at timestamps collide.
--   occurred_at — CLIENT clock (ISO8601, ms precision) — real event time.
--   received_at — SERVER clock (default now()) — lets us spot a device with
--                 a badly-skewed clock by comparing the two.
--   event_type  — 'button' | 'screen' | 'toggle' | 'lifecycle'.
--   name        — button id / screen name / lifecycle name (e.g.
--                 'app_background'); button id is an explicit track label
--                 when set, else the button title.
--   screen      — the screen the event happened on (button/toggle context).
create table public.ui_events (
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
-- Per-caregiver, per-session, in event order — the shape every analysis
-- query reads (reconstruct a session as an ordered stream).
create index ui_events_session_idx
  on public.ui_events(caregiver_id, session_id, seq);
-- Time-range scans across all caregivers (e.g. "all taps last 24h").
create index ui_events_occurred_idx on public.ui_events(occurred_at);


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

-- patient_face_embeddings: scoped through the embedding's patient → caregiver.
-- Mirrors the events_owner pattern: rows visible only to the caregiver who owns
-- the patient.
alter table public.patient_face_embeddings enable row level security;
create policy patient_face_embeddings_owner on public.patient_face_embeddings
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

alter table public.feedback enable row level security;
create policy feedback_owner on public.feedback
  for all using (caregiver_id = auth.uid()) with check (caregiver_id = auth.uid());

-- ui_events: a caregiver can only write/read their own interaction events.
-- The insert check (caregiver_id = auth.uid()) is what makes direct
-- client-side inserts safe — a caregiver physically cannot write a row
-- attributed to anyone else.
alter table public.ui_events enable row level security;
create policy ui_events_owner on public.ui_events
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
