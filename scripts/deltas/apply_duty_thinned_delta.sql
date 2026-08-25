-- duty_cycle 'sampled' renamed to 'thinned' (2026-07-18): "sampled"
-- collided with accumulate-style firing semantics. Pure rename, same
-- behavior (run every k-th eligible tick). Constraint swap + row
-- update; idempotent.
alter table public.detectors drop constraint if exists detectors_duty_cycle_check;
update public.detectors set duty_cycle = 'thinned' where duty_cycle = 'sampled';
alter table public.detectors add constraint detectors_duty_cycle_check
  check (duty_cycle = any (array['every_eligible_tick'::text, 'thinned'::text, 'scheduled_only'::text]));
