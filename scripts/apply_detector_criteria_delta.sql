-- Customs visibility (2026-07-18): caregiver-facing summary of the
-- compiled prompt + accumulated plain-words refinements. Idempotent.
alter table public.detectors add column if not exists user_criteria jsonb;
alter table public.detectors add column if not exists refinements jsonb;
