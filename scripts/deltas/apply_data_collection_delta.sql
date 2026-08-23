-- Adds the per-patient data-collection consent columns to an EXISTING
-- database. Idempotent. Apply BEFORE deploying the recorder backend
-- (DB-before-deploy — see docs/PROD_MIGRATION_RUNBOOK.md):
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_data_collection_delta.sql
--
-- data_collection_enabled    — master opt-in to store short video+audio
--   clips of the patient for research (only while someone is present).
--   Default false: nothing records until the caregiver opts in.
-- data_collection_decided_at — when the caregiver decided (NULL = never
--   asked). Drives the first-open consent banner on web + iOS; both
--   "Share" and "Not now" stamp it so the banner never re-nags.

alter table public.patients
  add column if not exists data_collection_enabled boolean not null default false;

alter table public.patients
  add column if not exists data_collection_decided_at timestamptz;
