-- Single-fire guard for the scheduled-detector runtime. The scheduler
-- evaluates each scheduled detector once per applicable day after its
-- deadline; this records the last date it ran (whether or not it
-- fired an absence event), so a 60s scheduler loop doesn't re-evaluate
-- the same window repeatedly. NULL = never evaluated.

alter table public.detectors
  add column last_scheduled_eval_date date;
