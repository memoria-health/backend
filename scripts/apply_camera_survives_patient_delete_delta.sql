-- Deleting a patient must not destroy a sensor that watches someone else.
--
-- `cameras.patient_id` is a compatibility shim since 2026-08-19 — the
-- earliest-covered patient, authoritative for nothing. Except it kept the
-- FK it was born with, `ON DELETE CASCADE`, so it was authoritative for
-- one thing after all: whether the row exists.
--
-- Sensor S covers {A, B}, shim = A. Deleting patient A cascaded away the
-- whole `cameras` row, and with it B's coverage, B's detector links and
-- the sensor's ingest token — while the bridge went on publishing into a
-- camera that no longer existed. B lost a physical sensor still pointed
-- at them, silently.
--
-- SET NULL instead: the patient's own coverage row goes (camera_patients
-- cascades correctly and always did), the sensor stays on the account,
-- and the application re-points the shim at whoever is still covered.
-- An unassigned sensor is now a legal, supported state, so landing there
-- is safe rather than a corruption.

BEGIN;

ALTER TABLE public.cameras DROP CONSTRAINT IF EXISTS cameras_patient_id_fkey;

ALTER TABLE public.cameras
  ADD CONSTRAINT cameras_patient_id_fkey
  FOREIGN KEY (patient_id) REFERENCES public.patients(id) ON DELETE SET NULL;

COMMIT;

-- ── Rollback note ───────────────────────────────────────────────────
-- Re-create the constraint with ON DELETE CASCADE. Nothing to restore:
-- this changes only what happens on a FUTURE delete. Note that rolling
-- back reinstates the data loss above, so it is only correct alongside a
-- revert of the coverage model itself.
