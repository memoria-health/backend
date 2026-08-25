-- Phase E refinement (2026-07-26): the alert_trigger 'schedule' is really the
-- ABSENCE condition (fire if NOT detected within its when_scope window); the
-- window itself is when_scope, not a trigger. Rename the value + drop the
-- positive-polarity concept (positive folds into 'leading'). LOCAL/soak only.
alter table public.detectors drop constraint if exists detectors_alert_trigger_check;
update public.detectors set alert_trigger = 'absence' where alert_trigger = 'schedule';
alter table public.detectors
  add constraint detectors_alert_trigger_check
  check (alert_trigger in ('leading','trailing','duration','absence'));
