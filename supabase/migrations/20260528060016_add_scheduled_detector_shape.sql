-- Scheduled ("absence") detectors. A scheduled detector runs the same
-- per-tick reactive detection as any other (produces `<id>_detected`),
-- but instead of firing the moment it's seen, a scheduler evaluates a
-- time window at its deadline and fires an event if the signal was
-- NEVER seen in that window ("medication not taken today").
--
-- shape gains 'scheduled' alongside the existing 'point_in_time' /
-- 'state'. ('state' stays in the enum for the agitation preset but is
-- no longer offered in the create UI — everything reactive is treated
-- as point_in_time.)
--
-- schedule_config (JSONB, null for reactive detectors) holds:
--   { "cadence": "daily"|"weekly",
--     "days_of_week": [0..6],          -- weekly only; 0=Mon..6=Sun
--     "window_start": "HH:MM",          -- local time
--     "window_end":   "HH:MM",
--     "deadline":     "HH:MM" }         -- when to evaluate + fire
-- The runtime scheduler (separate, later) reads this; storing it now
-- so the create UI can capture + persist it.

alter type detector_shape add value if not exists 'scheduled';

alter table public.detectors
  add column schedule_config jsonb;
