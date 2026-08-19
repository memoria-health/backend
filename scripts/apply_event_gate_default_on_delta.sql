-- Event gate defaults ON: review is the point of production.
--
-- Three sources disagreed about this column's default:
--   ORM (src/db/patient.py)  server_default true, default True
--   prod patients column     default false
--   supabase/schema.sql      column absent entirely (delta-only, never
--                            folded back), so a fresh database created
--                            from schema.sql had no gate column at all.
--
-- The ORM sends the value explicitly on insert, so new patients created
-- through the API already got TRUE regardless of the column default —
-- the drift was invisible until something inserted a patient by raw SQL
-- (seeds, deltas, the e2e helper), which silently got FALSE instead.
-- This makes the database agree with the code.
--
-- Idempotent — safe to re-run.
--
-- DOES NOT TOUCH EXISTING ROWS. A column default only applies to future
-- inserts. Flipping live households is a separate, deliberate act:
--
--   update public.patients set event_gate_enabled = true where id = '<id>';
--
-- or the audited route, POST /v1/event-gate/toggle, which logs the
-- before/after at WARNING because it changes a household's privacy
-- posture from the outside.
--
-- BEFORE ENABLING FOR A REAL HOUSEHOLD, know what holding means:
-- only category='safety' auto-releases (60s). Everything else waits for
-- a human verdict with NO timeout, so an unattended queue means those
-- events never reach the caregiver at all. Requires NEMARA_PAGERDUTY_
-- ROUTING_KEY + NEMARA_OPS_API_TOKEN + NEMARA_EVENT_GATE_LINK_SECRET +
-- NEMARA_EVENT_GATE_PUBLIC_BASE_URL to be set (done in prod 2026-08-19).
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_event_gate_default_on_delta.sql
-- or paste into the Supabase SQL editor. Run DB-before-deploy (see docs/PROD_MIGRATION_RUNBOOK.md).

alter table public.patients
  add column if not exists event_gate_enabled boolean not null default true;

alter table public.patients
  alter column event_gate_enabled set default true;

comment on column public.patients.event_gate_enabled is
  'Hold this patient''s alerts for an OPERATOR yes/no before any caregiver '
  'is told (src/services/event_gate.py). Defaults ON. Unrelated to '
  'events.reviewed, which is the caregiver''s own retrospective review.';
