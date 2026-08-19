-- Per-home timezone: the day boundary belongs to the PATIENT, not the reader.
--
-- Adds patients.timezone to an EXISTING database (local dev or prod).
-- schema.sql includes it for fresh creates; this delta is for a DB that
-- predates the column. Idempotent — safe to re-run.
--
-- Why: every day boundary the caregiver sees — the journal, scheduled
-- ("medication by 5pm") windows, the per-tick when_scope gate — resolved to
-- `caregivers.timezone`. That is the reader's zone, not the home's, so a
-- daughter in New York watching a father in California saw his day roll over
-- at 9pm his time, and "by 5pm" meant 2pm to him. Resolution order is now
-- patient → caregiver → America/New_York (src/services/patient_tz.py).
--
-- NULLABLE ON PURPOSE, no backfill. NULL means "never set", which falls back
-- to the caregiver's zone — exactly the behaviour every existing home has
-- today. Backfilling caregivers.timezone into every patient would look
-- identical right up until a caregiver moves or corrects their own zone, at
-- which point their patients would silently stop following it. Homes adopt a
-- real value when a client sends one; new patients get the creator's zone at
-- insert time (src/api/patients.py).
--
-- No validation constraint here: the server validates the IANA name on write
-- (schemas/patient.py:_known_zone) and degrades safely on read, and a check
-- constraint would need this database's tz table to agree with the app's.
--
-- Apply locally:  psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/apply_patient_timezone_delta.sql
-- or paste into the Supabase SQL editor. Run DB-before-deploy (see docs/PROD_MIGRATION_RUNBOOK.md).

alter table public.patients
  add column if not exists timezone text;

comment on column public.patients.timezone is
  'IANA timezone of this home. NULL = inherit the caregiver''s zone. '
  'Drives the journal day boundary and scheduled-detector windows.';
