-- Multi-room detector scope (2026-07-26).
--
-- Adds detectors.where_rooms to an EXISTING database (local dev, or prod).
-- schema.sql includes it for fresh creates; this delta is for a DB that
-- predates the column. Idempotent — safe to re-run.
--
-- where_rooms: JSON array of room names the detector is scoped to. NULL or
--   ["ALL ROOMS"] means every room (the back-compat default). where_scope
--   stays populated as a scalar fallback; where_rooms is the source of truth
--   for the rooms multi-select + camera auto-assignment.
--
-- Nullable, no default: EXISTING detectors come out NULL, which the code
-- treats as "all rooms" — matching their current single where_scope behavior,
-- so nothing changes for them until a caregiver edits the rooms.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_where_rooms_delta.sql
-- or paste into the Supabase SQL editor. Run DB-before-deploy (see docs/PROD_MIGRATION_RUNBOOK.md).

alter table public.detectors
  add column if not exists where_rooms jsonb;
