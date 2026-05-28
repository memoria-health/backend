-- Per-patient email alert preferences. Push notifications cover the
-- iOS app; this lets caregivers route email alerts per patient — each
-- patient can notify a different set of family/aides, and some
-- patients can have email off entirely.
--
-- email_alerts_enabled  — master on/off for THIS patient.
-- email_alert_recipients — ADDITIONAL recipients beyond the owning
--   caregiver's account email (which always receives this patient's
--   alerts when enabled). JSON array of strings.

alter table public.patients
  add column email_alerts_enabled boolean not null default false;

alter table public.patients
  add column email_alert_recipients jsonb not null default '[]'::jsonb;
