-- Terms of Service + Privacy Notice acceptance on the caregiver account.
--
-- Adds caregivers.tos_accepted_at + caregivers.tos_version to an EXISTING
-- database (local dev, or prod). schema.sql includes them for fresh creates;
-- this delta is for a DB that predates the columns. Idempotent — safe to re-run.
--
-- tos_accepted_at: server-stamped moment of acceptance. NULL = never accepted,
--   which the signup gate uses to block the app until the caregiver agrees.
-- tos_version: the document version string the caregiver accepted, so a later
--   material update can bump the version and re-prompt for consent.
--
-- Both nullable, no default: EXISTING accounts come out NULL (not accepted), so
-- the gate re-prompts them on next login — the desired behavior. Do NOT
-- backfill them as accepted.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_tos_acceptance_delta.sql
-- or paste into the Supabase SQL editor. Run DB-before-deploy (see docs/PROD_MIGRATION_RUNBOOK.md).

alter table public.caregivers
  add column if not exists tos_accepted_at timestamptz;
alter table public.caregivers
  add column if not exists tos_version text;
