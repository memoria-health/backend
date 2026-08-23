-- Research-consent policy versioning (2026-08-05). Additive, idempotent.
--
-- `patients.data_collection_decided_at` records THAT a decision was made,
-- but not what wording was on screen when it was. Once the research policy
-- text changes there is no way to tell who agreed to what, and no way to
-- re-prompt only the people who saw the old version.
--
-- Same pattern `caregivers.tos_version` already uses for the Terms: bump
-- the version string on a material change and re-prompt anyone below it.
--
-- Nullable, no default and NO BACKFILL: existing opt-ins come out NULL,
-- meaning "decided before versioning existed". Stamping them with the
-- current version would assert they saw wording that did not exist yet.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_data_collection_policy_version_delta.sql
-- Run DB-before-deploy (see docs/PROD_MIGRATION_RUNBOOK.md).

alter table public.patients
  add column if not exists data_collection_policy_version text;
