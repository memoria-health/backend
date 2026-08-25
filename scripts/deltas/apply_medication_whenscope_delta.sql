-- Medication adherence: drop the hardcoded "8 AM, 8 PM ±1H" when_scope
-- (2026-07-28, Aryaman): presets ship ALL DAY; schedules are caregiver
-- choices made through the editor. Idempotent; only touches preset rows
-- still carrying the authored string.
UPDATE public.detectors
SET when_scope = 'ALWAYS'
WHERE identifier = 'medication'
  AND when_scope = '8 AM, 8 PM ±1H';
