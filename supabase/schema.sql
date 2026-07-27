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
  -- Terms of Service + Privacy Notice acceptance. tos_accepted_at is the
  -- server-stamped moment the caregiver accepted (NULL = never accepted → the
  -- signup gate blocks the app until they do); tos_version is the document
  -- version they accepted, so a later material update can bump it and
  -- re-prompt. Mirrors the patient data-collection consent. Existing rows come
  -- out NULL (re-prompted on next login) — do not backfill as accepted.
  tos_accepted_at timestamptz,
  tos_version     text,
  -- Data-residency region for this account: 'US' | 'IN'. This is the
  -- "India key" — the single authoritative axis for splitting all tenant
  -- data by geography. Every sensitive row (patients, cameras, analyses,
  -- patient_face_embeddings, events, gateways) resolves ITS region through
  -- this column via its caregiver_id/patient_id FK, so the tag lives only
  -- here. It makes three future operations one query (`where region = ?`):
  -- lifting a region's data to in-region infra, scoping a breach
  -- notification, and answering a data-subject access request. Residency is
  -- trust-sensitive: it is set by the backend/admin and is IMMUTABLE from
  -- the client path (see the caregivers_region_immutable trigger below).
  region       text not null default 'US' check (region in ('US', 'IN')),
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now()
);

create trigger caregivers_set_updated_at
  before update on public.caregivers
  for each row execute function public.set_updated_at();

-- Data-residency guard: `region` must never change once set — moving it
-- would silently strand an account's data in the wrong geography. Clients
-- reach caregivers only through the backend, but as defence-in-depth we
-- hard-block any region change arriving on the RLS/PostgREST client path
-- (which carries JWT claims / runs as authenticated|anon). A direct
-- admin/service connection (no JWT claims, bypasses RLS) may still set or
-- correct it — that is the authoritative path used to tag India accounts
-- until the onboarding flow sets region at insert time. Fires before
-- caregivers_set_updated_at (alphabetical order), which is harmless.
create or replace function public.enforce_caregiver_region_immutable()
returns trigger
language plpgsql
as $$
begin
  if new.region is distinct from old.region then
    if current_setting('request.jwt.claims', true) is not null
       or current_user in ('authenticated', 'anon') then
      raise exception
        'caregivers.region is immutable from the client path (data-residency attribute); change it only via an admin/service connection'
        using errcode = 'check_violation';
    end if;
  end if;
  return new;
end;
$$;

create trigger caregivers_region_immutable
  before update on public.caregivers
  for each row execute function public.enforce_caregiver_region_immutable();

-- Authoritative resolver: the region a caregiver's data belongs to. Every
-- downstream residency decision (storage-key prefixing, region-scoped
-- queries, breach/DSAR scoping) should call this rather than re-deriving
-- the join, so there is exactly one source of truth. STABLE ⇒ usable in
-- read queries and index expressions.
create or replace function public.region_of(p_caregiver_id uuid)
returns text
language sql
stable
as $$
  select region from public.caregivers where id = p_caregiver_id;
$$;

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
  -- Data-residency region for THIS patient's data: 'US' | 'IN'. This is
  -- the authoritative residency key — it follows the monitored home /
  -- camera location, NOT the caregiver (a US-based caregiver can manage a
  -- patient living in India, whose video/audio/biometrics are Indian
  -- data). All heavy media (research clips → Backblaze, selfies + incident
  -- clips → Supabase Storage) and every per-patient row route by this
  -- column, so a region's data is liftable to in-region infra as one
  -- `where region = ?` slice. Captured at patient onboarding (India → 'IN',
  -- else 'US'); defaults 'US' so pre-region clients keep working. IMMUTABLE
  -- from the client path (see patients_region_immutable trigger).
  region          text not null default 'US' check (region in ('US', 'IN')),
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create index patients_caregiver_id_idx on public.patients(caregiver_id);

-- Same data-residency guard as caregivers.region: block any region change
-- arriving on the RLS/PostgREST client path (JWT claims / authenticated|
-- anon). The backend/admin (direct connection) may still set it at insert
-- or correct it. Residency must not silently move.
create or replace function public.enforce_patient_region_immutable()
returns trigger
language plpgsql
as $$
begin
  if new.region is distinct from old.region then
    if current_setting('request.jwt.claims', true) is not null
       or current_user in ('authenticated', 'anon') then
      raise exception
        'patients.region is immutable from the client path (data-residency attribute); change it only via an admin/service connection'
        using errcode = 'check_violation';
    end if;
  end if;
  return new;
end;
$$;

create trigger patients_region_immutable
  before update on public.patients
  for each row execute function public.enforce_patient_region_immutable();

-- Authoritative resolver: the region a patient's data belongs to. Storage
-- routing (Backblaze bucket, Supabase key prefix) and any region-scoped
-- query should call this, so there is exactly one source of truth.
create or replace function public.region_of_patient(p_patient_id uuid)
returns text
language sql
stable
as $$
  select region from public.patients where id = p_patient_id;
$$;

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
-- ── gateways ─────────────────────────────────────────────────────
-- One row per mini-PC gateway box (bridges LAN RTSP cameras into
-- LiveKit as cam_* participants). Each box holds ONE stable site token;
-- the backend hashes it and looks it up here to resolve the box ->
-- caregiver, then serves that box's roster. `patient_id` is kept
-- nullable for back-compat (the box binds to a CAREGIVER, and each of
-- its cameras binds to a patient via cameras.gateway_id). Backend-
-- internal (accessed by the service role via the gateway roster/
-- provision endpoints); no RLS — mirrors apply_gateway_delta.sql.
create table public.gateways (
  id           uuid primary key default gen_random_uuid(),
  caregiver_id uuid references public.caregivers(id) on delete cascade,
  patient_id   uuid,
  token_hash   text not null,
  created_at   timestamptz default now(),
  last_seen_at timestamptz
);

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
  -- 'gateway': a LAN RTSP camera bridged by a mini-PC gateway box.
  --   Provisioned via /v1/gateway/provision-camera; the box publishes
  --   to LiveKit as a cam_* participant. Carries an ingest token like a
  --   Pi and is owned by a specific box (gateway_id below).
  source_type         text not null default 'iphone'
    check (source_type in ('iphone', 'pi', 'gateway')),
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
  -- ── Gateway ownership + liveness ─────────────────────────────────
  -- source_type='gateway' rows. The box's roster poll doubles as a
  -- liveness heartbeat, stamped here. `gateway_id` is the specific box
  -- that runs this camera (a caregiver may have several); the roster +
  -- heartbeat scope by it so liveness is per-box and two boxes can't
  -- both claim one camera. NULL for iphone/pi and not-yet-assigned rows.
  last_heartbeat_at       timestamptz,
  gateway_id              uuid references public.gateways(id) on delete set null,
  constraint cameras_pi_stream_url_length
    check (pi_stream_url is null or length(pi_stream_url) <= 2048),
  constraint cameras_pi_audio_url_length
    check (pi_audio_url is null or length(pi_audio_url) <= 2048),
  constraint cameras_ingest_token_required
    check (
      (source_type in ('pi', 'gateway') and ingest_token_hash is not null)
      or (source_type = 'iphone' and ingest_token_hash is null)
    ),
  constraint cameras_ingest_token_hash_length
    check (ingest_token_hash is null or length(ingest_token_hash) = 64),
  unique (patient_id, display_id)
);

create index cameras_patient_id_idx on public.cameras(patient_id);
create index idx_cameras_gateway_id on public.cameras(gateway_id);

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
-- schema_json, always_on) feed the detector pipeline. The
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
  -- Room scope (canonical). JSON array of room names the detector is scoped
  -- to; null or ["ALL ROOMS"] = every room. Source of truth for the rooms
  -- selector + camera auto-assignment. The legacy where_scope scalar is
  -- derived from this on serialize (2026-07-26 collapse), not stored.
  where_rooms     jsonb,
  when_scope      text not null,
  sensitivity     detector_sensitivity not null default 'standard',
  is_on           boolean not null default true,
  custom_prompt   text,                              -- only meaningful when source = 'custom'
  -- Alerting spine (2026-07-26): which moment fires + internal temporal
  -- nature. Replaces `shape` (deprecated bridge below, dropped after the
  -- runtime is migrated). duration_seconds is the "too long" threshold for
  -- alert_trigger='duration'.
  alert_trigger   text not null default 'leading'
                  check (alert_trigger in ('leading','trailing','duration','absence')),
  is_sustained    boolean not null default false,
  duration_seconds integer,
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
  -- Wandering exit-door selection (only meaningful on the 'wandering'
  -- detector). The caregiver's choice from the setup picker:
  --   { "box": [x,y,w,h] (0-1) | null, "label": text,
  --     "source": "vlm"|"manual", "no_exit_door": bool }
  -- The wandering prompt_fragment is regenerated from this so the per-tick
  -- VLM only counts leaving THROUGH that doorway as wandering. no_exit_door
  -- => detector is turned off (nowhere to wander to).
  door_config     jsonb,
  -- people_present is always_on=true (included even if is_on=false)
  -- because the state machine requires its output.
  always_on       boolean not null default false,
  -- (is_builtin was dropped 2026-07-13 — it was 100% redundant with
  -- source: every 'preset' row was true, every 'custom' row false.
  -- Catalog membership checks key on source = 'preset'.)
  -- ── Detection v2 dimensions ────────────────────────────────────────
  -- Spec: docs/architecture/detection-architecture-v2.md. Python source of
  -- truth for built-in values: src/services/detection/catalog_v2.py (keep
  -- the backfill block in scripts/apply_detector_v2_delta.sql in sync).
  -- NULL = not yet classified (pre-backfill rows / legacy customs).
  -- Nothing before P3 reads these except severity (promotion prefers the
  -- row value, falls back to the in-code catalog map when NULL).
  requires_motion boolean,   -- collapse of evidence_branch: does the per-track motion floor apply
  subject_scope   text check (subject_scope in ('patient', 'anyone', 'environment')),
  bucket          text check (bucket in ('scene', 'person', 'fine_motion', 'rapid', 'face', 'snapshot')),
  needs_audio     boolean not null default false,
  evidence_unit   text check (evidence_unit in ('verdict', 'instance', 'sample')),
  eventing        jsonb,    -- {shape, k, n, M, T, absence_deadline}
  duty_cycle      text check (duty_cycle in ('every_eligible_tick', 'sampled', 'scheduled_only')),
  severity        integer,  -- within-tick precedence; moves out of catalog._SEVERITY
  cooldown_seconds integer, -- per-detector re-fire spacing for point shapes
  degradation_warning text check (degradation_warning in ('none', 'distance_class')),
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
  -- Per-SENSOR wandering exit-door config (only the 'wandering' detector).
  -- The exit door is per camera, so it lives here, not on the detector row:
  --   { "doors": [{"box":[x,y,w,h],"label":..,"source":"vlm"|"manual"}],
  --     "no_exit_door": bool }
  -- Runtime builds THIS camera's wandering prompt from it; no_exit_door=true
  -- skips wandering on this camera.
  door_config jsonb,
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

-- ── Patient sharing (multi-caregiver membership) ──────────────────
-- A caregiver's access to a patient is a row in patient_members, NOT the single
-- patients.caregiver_id FK. `role` is unused in the MVP (every member has full
-- access) but present so per-patient permission levels drop in later with NO
-- migration. The original owner is mirrored as a role='owner' row (backfill).
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

-- Backfill: every existing patient's owner becomes an 'owner' membership row.
insert into public.patient_members (patient_id, caregiver_id, role)
select id, caregiver_id, 'owner' from public.patients
on conflict (patient_id, caregiver_id) do nothing;

-- Single source of truth for "can this caller access this patient?" Referenced
-- by every patient-scoped RLS policy below. SECURITY DEFINER so the membership
-- lookup isn't itself re-filtered by RLS on patient_members.
create or replace function public.has_patient_access(pid uuid)
returns boolean language sql stable security definer set search_path = public as $$
  select exists (
    select 1 from public.patient_members m
    where m.patient_id = pid and m.caregiver_id = auth.uid()
  );
$$;

alter table public.caregivers enable row level security;
create policy caregivers_self on public.caregivers
  for all using (id = auth.uid()) with check (id = auth.uid());

-- patient_members: a caregiver may READ membership rows for patients they can
-- access. Inserts/deletes go through the owner-guarded service-role API, so the
-- authed client only needs read.
alter table public.patient_members enable row level security;
create policy patient_members_read on public.patient_members
  for select using (public.has_patient_access(patient_id));

-- Patient + children: gated by MEMBERSHIP (has_patient_access), not ownership.
-- The patients write-check stays caregiver_id = auth.uid() so a direct authed-
-- client write can't reassign ownership (patient writes actually go through the
-- service-role API).
alter table public.patients enable row level security;
create policy patients_member on public.patients
  for all using (public.has_patient_access(id)) with check (caregiver_id = auth.uid());

alter table public.cameras enable row level security;
create policy cameras_member on public.cameras
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

-- gateways is backend-internal: ONLY the postgres/service role (which
-- bypasses RLS) touches it, via the gateway roster/provision endpoints. No
-- client should ever reach it through PostgREST. So unlike the tenant
-- tables here it gets RLS with NO policy — deny-all to anon/authenticated —
-- plus a revoke of the default API grants as a second, independent lock.
-- Without this the public anon key could read the (caregiver_id,
-- token_hash) rows or forge/tamper with gateway rows.
alter table public.gateways enable row level security;
revoke all on public.gateways from anon, authenticated;

-- pi_discoveries is likewise backend-internal (the Pi announce/provision
-- flow). The app reads discoveries via the backend API, never the table
-- directly, and there's no caregiver_id to scope by. It holds a plaintext
-- one-shot ingest token (`pending_ingest_token`), so deny-all is essential.
alter table public.pi_discoveries enable row level security;
revoke all on public.pi_discoveries from anon, authenticated;

alter table public.detectors enable row level security;
create policy detectors_member on public.detectors
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

-- cameras_detectors: scoped through the camera's patient.
alter table public.cameras_detectors enable row level security;
create policy cameras_detectors_member on public.cameras_detectors
  for all using (
    camera_id in (select c.id from public.cameras c where public.has_patient_access(c.patient_id))
  ) with check (
    camera_id in (select c.id from public.cameras c where public.has_patient_access(c.patient_id))
  );

alter table public.events enable row level security;
create policy events_member on public.events
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

alter table public.patient_face_embeddings enable row level security;
create policy patient_face_embeddings_member on public.patient_face_embeddings
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

alter table public.analyses enable row level security;
create policy analyses_member on public.analyses
  for all using (public.has_patient_access(patient_id))
  with check (public.has_patient_access(patient_id));

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
