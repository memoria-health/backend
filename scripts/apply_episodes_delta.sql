-- Episodes: the annotation layer (2026-08-05).
--
-- Things that happened over a SPAN, on a per-camera wall-clock timeline.
-- Deliberately NOT attached to a clip: a sleep episode crosses ~30
-- recording rolls plus gaps where nothing recorded, so the link to video
-- is computed at read time from (started_at, ended_at) against a clip's
-- started_at + duration.
--
-- Idempotent. Additive only — nothing here changes existing behaviour,
-- and nothing reads this table until the writers land.

create table if not exists public.episodes (
  id            uuid primary key,
  caregiver_id  uuid not null references public.caregivers(id) on delete cascade,
  patient_id    uuid not null references public.patients(id)   on delete cascade,
  camera_id     uuid not null references public.cameras(id)    on delete cascade,

  -- detector slug ("fall", "wandering") or derived ("sleep"). Free text
  -- so a new detector needs no migration.
  kind          text not null,
  -- open | closed | retracted
  state         text not null default 'open',

  started_at    timestamptz not null,
  -- NULL while it is still happening. Sleep stays open all night.
  ended_at      timestamptz,

  -- Everything the detector was thinking: reasoning, k-of-N counters,
  -- latch state, quality flags. Filtered at EXPORT, where the audience is
  -- known — deciding at write time loses it forever.
  payload       jsonb not null default '{}'::jsonb,

  -- Revision without destruction. A re-cut episode points at its
  -- replacement instead of being overwritten, so a verdict already
  -- attached to it still refers to the span that was judged.
  superseded_by uuid references public.episodes(id) on delete set null,

  -- Human adjudication. NULL = nobody has looked, which IS the labelling
  -- backlog and is queryable as exactly that.
  reviewer_verdict boolean,
  reviewed_at      timestamptz,

  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

-- The projection query: everything overlapping a window for one camera.
create index if not exists episodes_camera_span
  on public.episodes (camera_id, started_at desc);

-- Episodes still running, so a writer can find the one to extend without
-- scanning history.
create index if not exists episodes_open
  on public.episodes (camera_id, kind) where ended_at is null;

-- The labelling backlog.
create index if not exists episodes_unreviewed
  on public.episodes (patient_id, started_at desc)
  where reviewer_verdict is null and superseded_by is null;

-- RLS. Supabase grants anon/authenticated full DML on every table in
-- public and PostgREST exposes it, so RLS is the only control here — and
-- anon is the PUBLISHABLE key that ships in the app binary. This table
-- carries detector reasoning text about a named patient, so it must not
-- be world-readable.
--
-- Backend-only: written and read through the service role, which bypasses
-- RLS. Enabling RLS with NO policy therefore denies every client role
-- while leaving the backend unaffected. Do NOT add a permissive policy
-- "to make it work" — that would open it, not secure it.
alter table public.episodes enable row level security;
