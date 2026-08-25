-- Sleep + activity summary tables (2026-07-28). Written nightly/daily by
-- tools/run_daily_summaries.py from the pure summarizers
-- (sleep_summary.py / activity_summary.py). RECORD-ONLY for pilots — no
-- notifications read these yet. Idempotent.
-- record_only alert trigger (sleep/activity): the detectors' fields feed
-- the summary tables below, never event promotion. The CHECK must admit
-- the new value or the rows can't insert.
ALTER TABLE public.detectors DROP CONSTRAINT IF EXISTS detectors_alert_trigger_check;
ALTER TABLE public.detectors ADD CONSTRAINT detectors_alert_trigger_check
    CHECK (alert_trigger = ANY (ARRAY[
        'leading'::text, 'trailing'::text, 'duration'::text,
        'absence'::text, 'record_only'::text]));

CREATE TABLE IF NOT EXISTS public.sleep_nights (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id    uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    night_date    date NOT NULL,             -- the evening's calendar date
    bedtime       timestamptz,
    sleep_onset   timestamptz,
    wake_time     timestamptz,
    duration_minutes  integer,
    disturbances      integer NOT NULL DEFAULT 0,
    out_of_bed_events jsonb   NOT NULL DEFAULT '[]'::jsonb,
    flags             jsonb   NOT NULL DEFAULT '[]'::jsonb,
    computed_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (patient_id, night_date)
);

-- v2 (same day): activity moved from motion-tick counts to time-weighted
-- VLM state minutes (lying/sitting/standing/active). Legacy count columns
-- retained with defaults; new columns below.
ALTER TABLE IF EXISTS public.activity_days
    ADD COLUMN IF NOT EXISTS state_minutes   jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS labeled_minutes real,
    ADD COLUMN IF NOT EXISTS label_ticks     integer NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS public.activity_days (
    id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id    uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
    day_date      date NOT NULL,
    per_camera    jsonb NOT NULL DEFAULT '{}'::jsonb,
    occupied_ticks    integer NOT NULL DEFAULT 0,
    moving_ticks      integer NOT NULL DEFAULT 0,
    active_fraction   real,
    room_transitions  integer NOT NULL DEFAULT 0,
    coverage_ticks    integer NOT NULL DEFAULT 0,
    state_minutes     jsonb NOT NULL DEFAULT '{}'::jsonb,
    labeled_minutes   real,
    label_ticks       integer NOT NULL DEFAULT 0,
    computed_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (patient_id, day_date)
);

-- RLS (2026-08-03). Supabase grants anon/authenticated full DML on every
-- table in public and PostgREST exposes it, so RLS is the only control —
-- and anon is the PUBLISHABLE key that ships in the app binary. Enabling
-- it with no policy denies all non-owner access (fail closed); the backend
-- reaches these through the service role, which bypasses RLS.
-- Must live in THIS file: deltas are applied one at a time, so fixing it in
-- a later delta leaves a live exposure window in between.
alter table public.sleep_nights enable row level security;
alter table public.activity_days enable row level security;
