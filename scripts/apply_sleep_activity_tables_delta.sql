-- Sleep + activity summary tables (2026-07-28). Written nightly/daily by
-- tools/run_daily_summaries.py from the pure summarizers
-- (sleep_summary.py / activity_summary.py). RECORD-ONLY for pilots — no
-- notifications read these yet. Idempotent.
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
    computed_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (patient_id, day_date)
);
