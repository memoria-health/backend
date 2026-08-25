-- Per-account "Live analyses" visibility.
--
-- Adds caregivers.show_live_analyses to an EXISTING database (local dev, or
-- prod). schema.sql already includes it for fresh creates; this delta is for
-- a DB that predates the column. Idempotent — safe to re-run.
--
-- Default TRUE: every existing and future account sees the Live analyses feed.
-- The web ALSO gates on a global SHOW_LIVE_ANALYSES kill-switch, so this column
-- is the per-account dial under that master switch.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/deltas/apply_live_analyses_delta.sql
-- or paste into the Supabase SQL editor / run via the Management API.

alter table public.caregivers
  add column if not exists show_live_analyses boolean not null default true;

-- Per request: the demo login must NOT see the Live analyses feed.
update public.caregivers set show_live_analyses = false where email = 'demo@test.com';
