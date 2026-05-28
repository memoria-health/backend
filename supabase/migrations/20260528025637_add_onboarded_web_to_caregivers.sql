-- Per-account flag for the web onboarding walkthrough.
-- False until the caregiver dismisses or completes the modal on the
-- web dashboard; PATCH /v1/me flips it true. iOS gets its own column
-- when we ship the iOS onboarding (deliberately separate so a user
-- who learned the web flow still sees iOS walkthrough on first use
-- of the mobile app, and vice versa).

alter table public.caregivers
  add column onboarded_web boolean not null default false;
