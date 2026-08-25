-- Convulsion removal (2026-07-28, Aryaman): the benchmark showed VLM video
-- sampling sits below tremor frequency (3-6 Hz) — a temporal-sampling
-- ceiling — so Convulsion is removed as a BUILT-IN. The rapid lane stays
-- for custom detectors, which now compile with a "rapid_frequency"
-- degradation warning. Successor: computed-motion FFT detector, post-pilot.
--
-- Idempotent. Deletes only PRESET seizure rows (customs untouched).
-- Cascades clean up events/links per existing FKs.
DELETE FROM public.detectors
WHERE identifier = 'seizure'
  AND source = 'preset';
