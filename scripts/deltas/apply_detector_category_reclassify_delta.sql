-- Reclassify built-in detectors into the four categories (2026-08-05).
--
-- Run AFTER apply_detector_category_behaviour_care_delta.sql, which adds the
-- enum values. This one only moves rows.
--
-- Before: 20 of 25 built-ins were 'safety'. That is what
-- scheduling/scheduler.py:68 meant by "the category axis proved unusable —
-- everything is 'safety' except medication". A label that 80% of rows share
-- tells a caregiver nothing.
--
-- SAFE BY CONSTRUCTION: category is semantic only. Treatment rides on
-- `severity`, which this delta does not touch, so no detector changes how it
-- DETECTS. `agitation` keeps severity 60 and its existing prompt bias; only
-- its label moves.
--
-- What the label DOES decide is how loudly the clients present it: both
-- mobile apps paint the patient card red only for an unopened 'safety'
-- event (iOS PatientRow.swift:31, Android RosterUiState.kt:181). That is
-- intended — safety is the only bucket that shouts, the other three are
-- informational. So moving a detector out of 'safety' is a deliberate
-- de-escalation and must only be done where that is wanted.
--
-- THE RULE (Aryaman, 2026-08-05): safety means it warrants immediate
-- MEDICAL attention. Nothing else qualifies. That is deliberately narrow —
-- a category 80% of rows shared was worthless, and a red card that fires
-- for everything trains people to ignore it.
--
-- `distress` (sustained crying, child cohort) fails that test: it is worth
-- surfacing, it is not medical. It keeps severity 82, so we still DETECT it
-- generously via the inclusive prompt — we just stop shouting about it.
-- Detect broadly, escalate narrowly.
--
-- Scoped to identifier, so a caregiver's own custom detector is never
-- touched. Their category was their (or the classifier's) choice and silently
-- rewriting it is worse than an inconsistent catalogue.
--
-- Idempotent: re-running is a no-op once the rows already hold the value.

-- how the person seems in themselves, rather than a hazard in the room
update public.detectors
   set category = 'behaviour'
 where identifier in ('agitation', 'distress', 'child_device_usage')
   and source = 'preset'
   and category is distinct from 'behaviour';

-- Verify:
--   select category, count(*) from public.detectors
--    where source = 'preset' group by 1 order by 2 desc;
--
-- Expect (per patient with the full library): safety 17, routine 5,
-- behaviour 3, care 0.
--
-- NOTE — 'care' is deliberately empty. The product pitches Care as "help to
-- the bathroom, company at meals, long stretches alone", and no built-in
-- detector watches for any of that today. The category exists so custom
-- detectors and the classifier can use it; filling it is product work, not a
-- migration. Flagged rather than papered over with a mislabelled detector.
