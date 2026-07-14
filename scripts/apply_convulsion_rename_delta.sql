-- Rename the seizure preset's DISPLAY name to "Convulsion"
-- (Aryaman 2026-07-14: the camera detects convulsive movement; "seizure"
-- is a diagnosis — many seizures never convulse — so the caregiver-facing
-- name must claim only what was measured).
--
-- The internal identifier stays 'seizure' (analyses fields seizure_*,
-- catalog key, dedup/absence continuity) — invisible to caregivers.
-- New detectors pick the name up from the catalog; this back-fills
-- existing rows. Idempotent; safe to re-run.

update public.detectors
   set name = 'Convulsion'
 where identifier = 'seizure'
   and name = 'Seizure';
