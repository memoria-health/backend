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
-- behaves. `distress` is the worked example — it moves to 'behaviour' but
-- keeps severity 82, stays above the compiler's inclusive threshold (80), and
-- goes on getting the generous prompt. If category still drove the bias, this
-- delta would silently make it conservative.
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
