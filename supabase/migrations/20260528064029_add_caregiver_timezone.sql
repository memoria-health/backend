-- Per-account timezone. Scheduled ("absence") detectors evaluate their
-- daily/weekly window in the owning caregiver's local time, so a
-- "medication by 5pm" deadline means 5pm in the caregiver's zone. The
-- web client auto-detects the browser timezone and PATCHes /v1/me, so
-- this default is only a fallback for accounts that never loaded the
-- web app. IANA tz name (e.g. "America/New_York").

alter table public.caregivers
  add column timezone text not null default 'America/New_York';
