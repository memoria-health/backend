-- Adds the per-patient privacy-mode column to an EXISTING database.
-- Idempotent. Apply BEFORE deploying the privacy-mode backend
-- (DB-before-deploy — see docs/PROD_MIGRATION_RUNBOOK.md):
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_privacy_mode_delta.sql
--
-- privacy_mode_enabled — per-patient dignity override. While ON it
--   overrides data collection (no clips stored, in-flight sessions
--   aborted without upload), severs all live video+audio to caregivers
--   at the SFU, and strips event clips from insights — while detection
--   and TEXT insights keep flowing. A live switch (no decided_at): it
--   SUSPENDS data_collection_enabled without mutating it. Default false,
--   so existing patients are unaffected until a caregiver opts in.

alter table public.patients
  add column if not exists privacy_mode_enabled boolean not null default false;

-- events.clip_suppressed — true when an event's clip was intentionally
-- NOT archived because the patient was in privacy mode. Records WHY the
-- clip is absent so the apps show "No clip — Privacy mode" instead of an
-- endless "Generating clip…" for has_clip=false events.
alter table public.events
  add column if not exists clip_suppressed boolean not null default false;
