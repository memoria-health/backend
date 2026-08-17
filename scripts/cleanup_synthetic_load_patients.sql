-- Removes the five synthetic load-test patients (F2-Syn-*/F3-Syn-*)
-- created for the 2026-08-13/14 GPU capacity ladders
-- (docs/IDENTITY-GPU-ARCHITECTURE.md). RUN ONLY when GPU testing is
-- finished — they are the load generators for the ladder and the
-- v2.3.1 stall investigation. Local/dev database only; these patients
-- never existed in production.
--
--   psql "$NEMARA_DB_URL_without_+asyncpg" -f scripts/cleanup_synthetic_load_patients.sql
--
-- Deletes dependents first so the patient rows go cleanly regardless of
-- FK cascade settings.

BEGIN;

DELETE FROM public.cameras
 WHERE patient_id IN (
    SELECT id FROM public.patients WHERE preferred_name LIKE 'F%-Syn-%');

DELETE FROM public.patients WHERE preferred_name LIKE 'F%-Syn-%';

COMMIT;
