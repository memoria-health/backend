-- Continuous ("state") detector hysteresis gap.
--
-- Adds detectors.episode_gap_seconds to an EXISTING database (local dev or
-- prod). schema.sql already includes it for fresh creates; this delta is for
-- a DB that predates the column. Idempotent — safe to re-run.
--
-- When shape = 'state', this is the number of seconds the activity must be
-- ABSENT before a resumption counts as a NEW event (rising-edge episode
-- promotion). Null for point_in_time / scheduled detectors.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_episode_gap_delta.sql
-- or paste into the Supabase SQL editor / run via the Management API.

alter table public.detectors
  add column if not exists episode_gap_seconds integer;
