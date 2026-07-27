-- Type-change-via-prompt (2026-07-26): a distinct plain-words timing/type
-- directive from /retype, separate from the semantic `refinements`. Appended
-- to custom_prompt when re-deriving shape + eventing so the caregiver can
-- change a detector's type in words ("only alert if it's been going 10 min").
-- Idempotent; LOCAL/soak only — do NOT run against prod without the deploy
-- runbook (DB delta before backend deploy).
alter table public.detectors add column if not exists timing_instruction text;
